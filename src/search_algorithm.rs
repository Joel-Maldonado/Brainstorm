use crate::utils::{encode_board_features, order_captures, order_moves, HistoryTable};
use pleco::core::GenTypes;
use pleco::{BitMove, Board, PieceType, Player};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tch::{CModule, Device};

const INF: i32 = 32_000;
const MATE_SCORE: i32 = 31_000;
const MATE_THRESHOLD: i32 = 30_000;
const EVAL_SCALE_CP: f32 = 2_500.0;
const MAX_PLY: usize = 128;
const TIME_CHECK_INTERVAL: u64 = 1_024;
const DEFAULT_HASH_MB: usize = 64;
const DEFAULT_EVAL_CACHE_MB: usize = 16;
const DEFAULT_THREADS: usize = 1;
const DEFAULT_MAX_DEPTH: u32 = 64;
const EVAL_CACHE_EMPTY_KEY: u64 = u64::MAX;
const EVAL_LARGE_KEY_MIX: u64 = 0x9e37_79b9_7f4a_7c15;
const Q_DELTA_MARGIN_CP: i32 = 120;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelMode {
    Small,
    Large,
    HybridRoot,
}

impl ModelMode {
    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "small" => Some(Self::Small),
            "large" => Some(Self::Large),
            "hybrid_root" => Some(Self::HybridRoot),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub hash_mb: usize,
    pub threads: usize,
    pub model_mode: ModelMode,
    pub debug_log: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            hash_mb: DEFAULT_HASH_MB,
            threads: DEFAULT_THREADS,
            model_mode: ModelMode::Small,
            debug_log: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SearchRequest {
    pub max_depth: u32,
    pub soft_time_ms: Option<u64>,
    pub hard_time_ms: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    pub best_move: BitMove,
    pub score_cp: i32,
    pub depth: u32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub stats: SearchStats,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SearchStats {
    pub eval_calls: u64,
    pub eval_cache_hits: u64,
    pub tt_probes: u64,
    pub tt_hits: u64,
    pub q_nodes: u64,
    pub beta_cutoffs: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct HashTableInfo {
    pub entries: usize,
    pub effective_mb: usize,
}

#[derive(Clone)]
pub struct SearchAlgorithm {
    small_evaluator: Arc<CModule>,
    large_evaluator: Arc<CModule>,
    eval_device: Device,
    pub should_stop: Arc<AtomicBool>,
    tt: Arc<Mutex<TranspositionTable>>,
    eval_cache: Arc<Mutex<EvalCache>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug)]
struct TTEntry {
    key: u64,
    depth: i16,
    score: i32,
    bound: Bound,
    best_move: BitMove,
    generation: u8,
}

impl Default for TTEntry {
    fn default() -> Self {
        Self {
            key: 0,
            depth: -1,
            score: 0,
            bound: Bound::Exact,
            best_move: BitMove::null(),
            generation: 0,
        }
    }
}

struct TranspositionTable {
    entries: Vec<TTEntry>,
    mask: usize,
    generation: u8,
    configured_hash_mb: usize,
}

impl TranspositionTable {
    fn new(hash_mb: usize) -> Self {
        let entries = tt_entries_from_mb(hash_mb);
        Self {
            entries: vec![TTEntry::default(); entries],
            mask: entries - 1,
            generation: 1,
            configured_hash_mb: hash_mb.max(1),
        }
    }

    fn ensure_size(&mut self, hash_mb: usize) {
        let requested = hash_mb.max(1);
        if requested == self.configured_hash_mb {
            return;
        }
        let entries = tt_entries_from_mb(requested);
        self.entries = vec![TTEntry::default(); entries];
        self.mask = entries - 1;
        self.generation = 1;
        self.configured_hash_mb = requested;
    }

    fn next_generation(&mut self) -> u8 {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.entries.fill(TTEntry::default());
            self.generation = 1;
        }
        self.generation
    }

    fn info_for_hash(hash_mb: usize) -> HashTableInfo {
        let entries = tt_entries_from_mb(hash_mb.max(1));
        let effective_mb = entries.saturating_mul(std::mem::size_of::<TTEntry>()) / (1024 * 1024);
        HashTableInfo {
            entries,
            effective_mb,
        }
    }
}

#[derive(Clone, Copy)]
struct EvalCacheEntry {
    key: u64,
    score: i32,
}

impl Default for EvalCacheEntry {
    fn default() -> Self {
        Self {
            key: EVAL_CACHE_EMPTY_KEY,
            score: 0,
        }
    }
}

struct EvalCache {
    entries: Vec<EvalCacheEntry>,
    mask: usize,
    configured_mb: usize,
}

impl EvalCache {
    fn new(cache_mb: usize) -> Self {
        let entries = eval_cache_entries_from_mb(cache_mb);
        Self {
            entries: vec![EvalCacheEntry::default(); entries],
            mask: entries - 1,
            configured_mb: cache_mb.max(1),
        }
    }

    fn ensure_size(&mut self, cache_mb: usize) {
        let requested = cache_mb.max(1);
        if requested == self.configured_mb {
            return;
        }
        let entries = eval_cache_entries_from_mb(requested);
        self.entries = vec![EvalCacheEntry::default(); entries];
        self.mask = entries - 1;
        self.configured_mb = requested;
    }

    fn probe(&self, key: u64) -> Option<i32> {
        let idx = (key as usize) & self.mask;
        let entry = self.entries[idx];
        if entry.key == key {
            Some(entry.score)
        } else {
            None
        }
    }

    fn store(&mut self, key: u64, score: i32) {
        let idx = (key as usize) & self.mask;
        self.entries[idx] = EvalCacheEntry { key, score };
    }
}

#[derive(Debug, Clone, Copy)]
struct RootOutcome {
    score: i32,
    best_move: BitMove,
    completed: bool,
}

impl SearchAlgorithm {
    pub fn new(
        small_evaluator: Arc<CModule>,
        large_evaluator: Arc<CModule>,
        eval_device: Device,
        should_stop: Arc<AtomicBool>,
    ) -> Self {
        Self {
            small_evaluator,
            large_evaluator,
            eval_device,
            should_stop,
            tt: Arc::new(Mutex::new(TranspositionTable::new(DEFAULT_HASH_MB))),
            eval_cache: Arc::new(Mutex::new(EvalCache::new(DEFAULT_EVAL_CACHE_MB))),
        }
    }

    pub fn hash_table_info(hash_mb: usize) -> HashTableInfo {
        TranspositionTable::info_for_hash(hash_mb)
    }

    pub fn search(
        &self,
        board: &Board,
        request: SearchRequest,
        options: &SearchOptions,
        game_history: &[u64],
    ) -> SearchResult {
        self.search_with_info(board, request, options, game_history, true)
    }

    pub fn search_quiet(
        &self,
        board: &Board,
        request: SearchRequest,
        options: &SearchOptions,
        game_history: &[u64],
    ) -> SearchResult {
        self.search_with_info(board, request, options, game_history, false)
    }

    fn search_with_info(
        &self,
        board: &Board,
        request: SearchRequest,
        options: &SearchOptions,
        game_history: &[u64],
        emit_info: bool,
    ) -> SearchResult {
        self.should_stop.store(false, Ordering::Relaxed);
        let start = Instant::now();
        let _no_grad = tch::no_grad_guard();

        let requested_threads = options.threads.max(1).min(i32::MAX as usize) as i32;
        tch::set_num_threads(requested_threads);

        let soft_deadline = request
            .soft_time_ms
            .map(|ms| start + Duration::from_millis(ms));
        let hard_deadline = request
            .hard_time_ms
            .map(|ms| start + Duration::from_millis(ms));

        let mut tt_guard = self.tt.lock().unwrap();
        tt_guard.ensure_size(options.hash_mb);
        let tt_generation = tt_guard.next_generation();
        let mut eval_cache_guard = self.eval_cache.lock().unwrap();
        eval_cache_guard.ensure_size(DEFAULT_EVAL_CACHE_MB);

        let mut ctx = SearchContext::new(
            board,
            &self.small_evaluator,
            &self.large_evaluator,
            self.eval_device,
            self.should_stop.as_ref(),
            options,
            &mut tt_guard,
            tt_generation,
            &mut eval_cache_guard,
            soft_deadline,
            hard_deadline,
            game_history,
        );

        let mut root_board = board.shallow_clone();
        let legal_moves = root_board.generate_moves();
        let fallback_move = if legal_moves.is_empty() {
            BitMove::null()
        } else {
            legal_moves[0]
        };

        if fallback_move.is_null() {
            return SearchResult {
                best_move: BitMove::null(),
                score_cp: 0,
                depth: 0,
                nodes: 0,
                elapsed: start.elapsed(),
                stats: SearchStats::default(),
            };
        }

        let max_depth = request.max_depth.max(1).min(DEFAULT_MAX_DEPTH);
        let mut best_move = fallback_move;
        let mut best_score = 0;
        let mut completed_depth = 0_u32;
        let mut prev_score = 0_i32;

        for depth in 1..=max_depth {
            if ctx.should_abort() {
                break;
            }

            let mut alpha = -INF;
            let mut beta = INF;
            let mut window = 50;
            if depth >= 3 {
                alpha = prev_score - window;
                beta = prev_score + window;
            }

            let outcome = loop {
                let outcome = ctx.search_root(&mut root_board, depth as i32, alpha, beta);
                if !outcome.completed {
                    break outcome;
                }

                if outcome.score <= alpha {
                    alpha = (alpha - window).max(-INF);
                    window = (window * 2).min(8_000);
                    continue;
                }

                if outcome.score >= beta {
                    beta = (beta + window).min(INF);
                    window = (window * 2).min(8_000);
                    continue;
                }

                break outcome;
            };

            if !outcome.completed {
                break;
            }

            best_move = outcome.best_move;
            best_score = outcome.score;
            prev_score = best_score;
            completed_depth = depth;

            if emit_info {
                let (score_kind, score_value) = score_to_uci(best_score);
                println!(
                    "info depth {} score {} {} nodes {} time {} pv {}",
                    depth,
                    score_kind,
                    score_value,
                    ctx.nodes,
                    start.elapsed().as_millis(),
                    best_move
                );
            }

            if ctx.soft_deadline_reached() {
                break;
            }
        }

        SearchResult {
            best_move,
            score_cp: best_score,
            depth: completed_depth,
            nodes: ctx.nodes,
            elapsed: start.elapsed(),
            stats: ctx.stats,
        }
    }
}

struct SearchContext<'a> {
    small_eval: &'a CModule,
    large_eval: &'a CModule,
    eval_device: Device,
    should_stop: &'a AtomicBool,
    model_mode: ModelMode,
    _debug_log: bool,
    tt: &'a mut TranspositionTable,
    tt_generation: u8,
    eval_cache: &'a mut EvalCache,
    soft_deadline: Option<Instant>,
    hard_deadline: Option<Instant>,
    killers: Vec<[BitMove; 2]>,
    history: HistoryTable,
    repetition_counts: HashMap<u64, u8>,
    eval_features: [f32; 775],
    stats: SearchStats,
    nodes: u64,
}

impl<'a> SearchContext<'a> {
    fn new(
        board: &Board,
        small_eval: &'a CModule,
        large_eval: &'a CModule,
        eval_device: Device,
        should_stop: &'a AtomicBool,
        options: &SearchOptions,
        tt: &'a mut TranspositionTable,
        tt_generation: u8,
        eval_cache: &'a mut EvalCache,
        soft_deadline: Option<Instant>,
        hard_deadline: Option<Instant>,
        game_history: &[u64],
    ) -> Self {
        let mut repetition_counts = HashMap::new();
        for &key in game_history {
            let entry = repetition_counts.entry(key).or_insert(0u8);
            *entry = (*entry).saturating_add(1);
        }
        let root_key = board.zobrist();
        let entry = repetition_counts.entry(root_key).or_insert(0u8);
        if *entry == 0 {
            *entry = 1;
        }

        Self {
            small_eval,
            large_eval,
            eval_device,
            should_stop,
            model_mode: options.model_mode,
            _debug_log: options.debug_log,
            tt,
            tt_generation,
            eval_cache,
            soft_deadline,
            hard_deadline,
            killers: vec![[BitMove::null(), BitMove::null()]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            repetition_counts,
            eval_features: [0.0; 775],
            stats: SearchStats::default(),
            nodes: 0,
        }
    }

    fn search_root(
        &mut self,
        board: &mut Board,
        depth: i32,
        mut alpha: i32,
        beta: i32,
    ) -> RootOutcome {
        let alpha_orig = alpha;
        let mut moves = board.generate_moves().to_vec();
        if moves.is_empty() {
            if board.in_check() {
                return RootOutcome {
                    score: -mate_in(0),
                    best_move: BitMove::null(),
                    completed: true,
                };
            }
            return RootOutcome {
                score: 0,
                best_move: BitMove::null(),
                completed: true,
            };
        }

        let tt_move = self.tt_best_move(board.zobrist());
        let side_to_move = board.turn();
        let killers = self.killers[0];
        order_moves(
            board,
            &mut moves,
            tt_move,
            killers,
            &self.history,
            side_to_move,
        );

        let mut best_score = -INF;
        let mut best_move = moves[0];
        let mut completed = true;

        for (idx, mv) in moves.into_iter().enumerate() {
            if self.should_abort() {
                completed = false;
                break;
            }

            let is_quiet = !board.is_capture_or_promotion(mv);
            board.apply_move(mv);
            let child_key = board.zobrist();
            self.push_repetition(child_key);

            let mut score;
            if idx == 0 {
                score = -self.negamax(board, depth - 1, 1, -beta, -alpha, true);
            } else {
                score = -self.negamax(board, depth - 1, 1, -alpha - 1, -alpha, true);
                if score > alpha && score < beta {
                    score = -self.negamax(board, depth - 1, 1, -beta, -alpha, true);
                }
            }

            self.pop_repetition(child_key);
            board.undo_move();

            if self.should_abort() {
                completed = false;
                break;
            }

            if score > best_score {
                best_score = score;
                best_move = mv;
            }

            if score > alpha {
                alpha = score;
            }

            if alpha >= beta {
                self.stats.beta_cutoffs = self.stats.beta_cutoffs.saturating_add(1);
                if is_quiet {
                    self.store_killer(0, mv);
                    self.update_history(side_to_move, mv, depth);
                }
                break;
            }
        }

        if completed {
            let bound = if best_score <= alpha_orig {
                Bound::Upper
            } else if best_score >= beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            self.store_tt(
                board.zobrist(),
                depth as i16,
                best_score,
                bound,
                best_move,
                0,
            );
        }

        RootOutcome {
            score: best_score,
            best_move,
            completed,
        }
    }

    fn negamax(
        &mut self,
        board: &mut Board,
        depth: i32,
        ply: usize,
        mut alpha: i32,
        beta: i32,
        allow_null: bool,
    ) -> i32 {
        if self.bump_node_and_check_stop() {
            return alpha;
        }

        if ply >= MAX_PLY - 1 {
            return self.evaluate(board, ply);
        }

        if self.is_draw(board) {
            return 0;
        }

        let in_check = board.in_check();
        let mut depth = depth;
        if in_check && depth > 0 {
            depth += 1;
        }

        if depth <= 0 {
            return self.quiescence(board, ply, alpha, beta);
        }

        if let Some(tt_value) = self.probe_tt(board.zobrist(), depth as i16, alpha, beta, ply) {
            return tt_value;
        }

        let alpha_orig = alpha;
        let mut best_move = BitMove::null();
        let mut best_score = -INF;

        if allow_null && depth >= 3 && !in_check && board.non_pawn_material(board.turn()) > 0 {
            unsafe { board.apply_null_move() };
            let score = -self.negamax(board, depth - 3, ply + 1, -beta, -beta + 1, false);
            unsafe { board.undo_null_move() };

            if self.should_abort() {
                return alpha;
            }

            if score >= beta {
                self.stats.beta_cutoffs = self.stats.beta_cutoffs.saturating_add(1);
                return beta;
            }
        }

        let mut moves = board.generate_moves().to_vec();
        if moves.is_empty() {
            return if in_check { -mate_in(ply) } else { 0 };
        }

        let side_to_move = board.turn();
        let killers = self
            .killers
            .get(ply)
            .copied()
            .unwrap_or([BitMove::null(), BitMove::null()]);
        let tt_move = self.tt_best_move(board.zobrist());
        order_moves(
            board,
            &mut moves,
            tt_move,
            killers,
            &self.history,
            side_to_move,
        );

        for (idx, mv) in moves.into_iter().enumerate() {
            if self.should_abort() {
                break;
            }

            let is_quiet = !board.is_capture_or_promotion(mv);
            board.apply_move(mv);
            let child_key = board.zobrist();
            self.push_repetition(child_key);

            let mut score;
            if idx == 0 {
                score = -self.negamax(board, depth - 1, ply + 1, -beta, -alpha, true);
            } else {
                let mut reduction = 0;
                if depth >= 3 && idx >= 4 && is_quiet && !in_check {
                    reduction = 1;
                }
                let reduced_depth = (depth - 1 - reduction).max(0);
                score = -self.negamax(board, reduced_depth, ply + 1, -alpha - 1, -alpha, true);

                if reduction > 0 && score > alpha {
                    score = -self.negamax(board, depth - 1, ply + 1, -alpha - 1, -alpha, true);
                }
                if score > alpha && score < beta {
                    score = -self.negamax(board, depth - 1, ply + 1, -beta, -alpha, true);
                }
            }

            self.pop_repetition(child_key);
            board.undo_move();

            if self.should_abort() {
                break;
            }

            if score > best_score {
                best_score = score;
                best_move = mv;
            }

            if score > alpha {
                alpha = score;
            }

            if alpha >= beta {
                self.stats.beta_cutoffs = self.stats.beta_cutoffs.saturating_add(1);
                if is_quiet {
                    self.store_killer(ply, mv);
                    self.update_history(side_to_move, mv, depth);
                }
                break;
            }
        }

        let bound = if best_score <= alpha_orig {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.store_tt(
            board.zobrist(),
            depth as i16,
            best_score,
            bound,
            best_move,
            ply,
        );

        best_score
    }

    fn quiescence(&mut self, board: &mut Board, ply: usize, mut alpha: i32, beta: i32) -> i32 {
        self.stats.q_nodes = self.stats.q_nodes.saturating_add(1);
        if self.bump_node_and_check_stop() {
            return alpha;
        }

        if self.is_draw(board) {
            return 0;
        }

        if board.in_check() {
            let mut evasions = board.generate_moves().to_vec();
            if evasions.is_empty() {
                return -mate_in(ply);
            }
            let side_to_move = board.turn();
            let killers = self
                .killers
                .get(ply)
                .copied()
                .unwrap_or([BitMove::null(), BitMove::null()]);
            order_moves(
                board,
                &mut evasions,
                None,
                killers,
                &self.history,
                side_to_move,
            );

            for mv in evasions {
                if self.should_abort() {
                    break;
                }

                board.apply_move(mv);
                let child_key = board.zobrist();
                self.push_repetition(child_key);
                let score = -self.quiescence(board, ply + 1, -beta, -alpha);
                self.pop_repetition(child_key);
                board.undo_move();

                if self.should_abort() {
                    break;
                }

                if score >= beta {
                    self.stats.beta_cutoffs = self.stats.beta_cutoffs.saturating_add(1);
                    return beta;
                }
                if score > alpha {
                    alpha = score;
                }
            }

            return alpha;
        }

        let stand_pat = self.evaluate(board, ply);
        if stand_pat >= beta {
            self.stats.beta_cutoffs = self.stats.beta_cutoffs.saturating_add(1);
            return beta;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        let mut captures = board.generate_moves_of_type(GenTypes::Captures).to_vec();
        if captures.is_empty() {
            return alpha;
        }
        order_captures(board, &mut captures);

        for mv in captures {
            if self.should_abort() {
                break;
            }

            let immediate_gain = capture_move_gain(board, mv);
            if stand_pat
                .saturating_add(immediate_gain)
                .saturating_add(Q_DELTA_MARGIN_CP)
                < alpha
            {
                continue;
            }

            board.apply_move(mv);
            let child_key = board.zobrist();
            self.push_repetition(child_key);
            let score = -self.quiescence(board, ply + 1, -beta, -alpha);
            self.pop_repetition(child_key);
            board.undo_move();

            if self.should_abort() {
                break;
            }

            if score >= beta {
                self.stats.beta_cutoffs = self.stats.beta_cutoffs.saturating_add(1);
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }

    fn evaluate(&mut self, board: &Board, ply: usize) -> i32 {
        self.stats.eval_calls = self.stats.eval_calls.saturating_add(1);

        let use_large = match self.model_mode {
            ModelMode::Small => false,
            ModelMode::Large => true,
            ModelMode::HybridRoot => ply <= 1,
        };

        let mut cache_key = board.zobrist() ^ if use_large { EVAL_LARGE_KEY_MIX } else { 0 };
        if cache_key == EVAL_CACHE_EMPTY_KEY {
            cache_key ^= 1;
        }
        if let Some(score) = self.eval_cache.probe(cache_key) {
            self.stats.eval_cache_hits = self.stats.eval_cache_hits.saturating_add(1);
            return score;
        }

        encode_board_features(board, &mut self.eval_features);
        let input = tch::Tensor::from_slice(&self.eval_features)
            .view([1, 775])
            .to_device(self.eval_device);
        let raw = if use_large {
            self.large_eval
                .forward_ts(&[input])
                .map(|tensor| tensor.double_value(&[]) as f32)
                .unwrap_or(0.0)
        } else {
            self.small_eval
                .forward_ts(&[input])
                .map(|tensor| tensor.double_value(&[]) as f32)
                .unwrap_or(0.0)
        };
        let mut score = (raw * EVAL_SCALE_CP).round() as i32;
        if board.turn() == Player::Black {
            score = -score;
        }
        self.eval_cache.store(cache_key, score);
        score
    }

    fn is_draw(&self, board: &Board) -> bool {
        if board.rule_50() >= 100 {
            return true;
        }
        self.repetition_counts
            .get(&board.zobrist())
            .copied()
            .unwrap_or(0)
            >= 3
    }

    fn push_repetition(&mut self, key: u64) {
        let entry = self.repetition_counts.entry(key).or_insert(0u8);
        *entry = entry.saturating_add(1);
    }

    fn pop_repetition(&mut self, key: u64) {
        if let Some(entry) = self.repetition_counts.get_mut(&key) {
            if *entry <= 1 {
                self.repetition_counts.remove(&key);
            } else {
                *entry -= 1;
            }
        }
    }

    fn side_index(side: Player) -> usize {
        if side == Player::White {
            0
        } else {
            1
        }
    }

    fn update_history(&mut self, side: Player, mv: BitMove, depth: i32) {
        let from = mv.get_src_u8() as usize;
        let to = mv.get_dest_u8() as usize;
        let bonus = (depth.max(1) * depth.max(1)).min(64);
        let slot = &mut self.history[Self::side_index(side)][from][to];
        *slot = (*slot + bonus).min(20_000);
    }

    fn store_killer(&mut self, ply: usize, mv: BitMove) {
        if ply >= self.killers.len() {
            return;
        }
        let killers = &mut self.killers[ply];
        if killers[0] != mv {
            killers[1] = killers[0];
            killers[0] = mv;
        }
    }

    fn should_abort(&self) -> bool {
        if self.should_stop.load(Ordering::Relaxed) {
            return true;
        }
        if let Some(deadline) = self.hard_deadline {
            if Instant::now() >= deadline {
                self.should_stop.store(true, Ordering::Relaxed);
                return true;
            }
        }
        false
    }

    fn soft_deadline_reached(&self) -> bool {
        if let Some(deadline) = self.soft_deadline {
            return Instant::now() >= deadline;
        }
        false
    }

    fn bump_node_and_check_stop(&mut self) -> bool {
        self.nodes = self.nodes.saturating_add(1);
        if self.nodes % TIME_CHECK_INTERVAL == 0 {
            return self.should_abort();
        }
        self.should_stop.load(Ordering::Relaxed)
    }

    fn tt_best_move(&self, key: u64) -> Option<BitMove> {
        let idx = self.tt_index(key);
        let entry = self.tt.entries[idx];
        if entry.key == key && entry.generation == self.tt_generation && !entry.best_move.is_null()
        {
            Some(entry.best_move)
        } else {
            None
        }
    }

    fn probe_tt(&mut self, key: u64, depth: i16, alpha: i32, beta: i32, ply: usize) -> Option<i32> {
        self.stats.tt_probes = self.stats.tt_probes.saturating_add(1);
        let idx = self.tt_index(key);
        let entry = self.tt.entries[idx];
        if entry.key != key || entry.generation != self.tt_generation || entry.depth < depth {
            return None;
        }

        self.stats.tt_hits = self.stats.tt_hits.saturating_add(1);
        let score = score_from_tt(entry.score, ply);
        match entry.bound {
            Bound::Exact => Some(score),
            Bound::Lower if score >= beta => Some(score),
            Bound::Upper if score <= alpha => Some(score),
            _ => None,
        }
    }

    fn store_tt(
        &mut self,
        key: u64,
        depth: i16,
        score: i32,
        bound: Bound,
        best_move: BitMove,
        ply: usize,
    ) {
        let idx = self.tt_index(key);
        let existing = self.tt.entries[idx];

        let replace = existing.key != key
            || existing.generation != self.tt_generation
            || depth >= existing.depth
            || (bound == Bound::Exact && existing.bound != Bound::Exact);

        if replace {
            self.tt.entries[idx] = TTEntry {
                key,
                depth,
                score: score_to_tt(score, ply),
                bound,
                best_move,
                generation: self.tt_generation,
            };
        }
    }

    fn tt_index(&self, key: u64) -> usize {
        (key as usize) & self.tt.mask
    }
}

fn tt_entries_from_mb(hash_mb: usize) -> usize {
    let bytes = hash_mb.max(1).saturating_mul(1024 * 1024);
    let mut entries = bytes / std::mem::size_of::<TTEntry>().max(1);
    entries = entries.max(1);
    let rounded = entries.next_power_of_two();
    if rounded > entries {
        (rounded / 2).max(1)
    } else {
        rounded
    }
}

fn eval_cache_entries_from_mb(cache_mb: usize) -> usize {
    let bytes = cache_mb.max(1).saturating_mul(1024 * 1024);
    let mut entries = bytes / std::mem::size_of::<EvalCacheEntry>().max(1);
    entries = entries.max(1);
    let rounded = entries.next_power_of_two();
    if rounded > entries {
        (rounded / 2).max(1)
    } else {
        rounded
    }
}

fn piece_value(piece_type: PieceType) -> i32 {
    match piece_type {
        PieceType::P => 100,
        PieceType::N => 320,
        PieceType::B => 330,
        PieceType::R => 500,
        PieceType::Q => 900,
        PieceType::K => 20_000,
        PieceType::None | PieceType::All => 0,
    }
}

fn capture_move_gain(board: &Board, mv: BitMove) -> i32 {
    let captured_value = if mv.is_en_passant() {
        piece_value(PieceType::P)
    } else {
        piece_value(board.piece_at_sq(mv.get_dest()).type_of())
    };

    let promotion_gain = if mv.is_promo() {
        piece_value(mv.promo_piece()).saturating_sub(piece_value(PieceType::P))
    } else {
        0
    };

    captured_value.saturating_add(promotion_gain)
}

fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score > MATE_THRESHOLD {
        score + ply as i32
    } else if score < -MATE_THRESHOLD {
        score - ply as i32
    } else {
        score
    }
}

fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score > MATE_THRESHOLD {
        score - ply as i32
    } else if score < -MATE_THRESHOLD {
        score + ply as i32
    } else {
        score
    }
}

fn mate_in(ply: usize) -> i32 {
    MATE_SCORE - ply as i32
}

fn score_to_uci(score: i32) -> (&'static str, i32) {
    if score > MATE_THRESHOLD {
        let plies_to_mate = (MATE_SCORE - score).max(0);
        let mate_moves = (plies_to_mate + 1) / 2;
        ("mate", mate_moves)
    } else if score < -MATE_THRESHOLD {
        let plies_to_mate = (MATE_SCORE + score).max(0);
        let mate_moves = -((plies_to_mate + 1) / 2);
        ("mate", mate_moves)
    } else {
        ("cp", score)
    }
}
