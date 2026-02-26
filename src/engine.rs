use crate::search_algorithm::{ModelMode, SearchAlgorithm, SearchOptions, SearchRequest};
use pleco::{BitMove, Board, Player};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use tch::{CModule, Device};

const SMALL_MODEL_PATH: &str = "models/eval_params264k_norm_mse0.117666_jit.pt";
const LARGE_MODEL_PATH: &str = "models/eval_660k_norm_mse_0.026550_jit.pt";
const MAX_HASH_MB: usize = 4096;
const DEFAULT_FALLBACK_MOVETIME_MS: u64 = 2_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvalDeviceChoice {
    Auto,
    Cpu,
    Cuda,
}

impl EvalDeviceChoice {
    fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }
}

pub struct Engine {
    pub board: Board,
    pub search_algorithm: SearchAlgorithm,
    options: SearchOptions,
    device_choice: EvalDeviceChoice,
    active_device: Device,
    position_history: Vec<u64>,
    search_handle: Option<JoinHandle<()>>,
}

impl Engine {
    pub fn new() -> Self {
        let board = Board::default();
        let should_stop = Arc::new(AtomicBool::new(false));
        let device_choice = EvalDeviceChoice::Auto;
        let (search_algorithm, active_device, startup_message) =
            Self::build_search_algorithm_for_choice(device_choice, Arc::clone(&should_stop))
                .expect("failed to initialize evaluators");
        if let Some(message) = startup_message {
            eprintln!("[engine] {message}");
        }

        let options = SearchOptions {
            hash_mb: 64,
            threads: SearchAlgorithm::default_threads(),
            model_mode: ModelMode::Small,
            debug_log: false,
        };

        let root_key = board.zobrist();

        Self {
            board,
            search_algorithm,
            options,
            device_choice,
            active_device,
            position_history: vec![root_key],
            search_handle: None,
        }
    }

    pub fn uci(&self) {
        println!("id name Brainstorm");
        println!("id author JoelM");
        println!("option name Hash type spin default 64 min 1 max 4096");
        println!(
            "option name Threads type spin default {} min 1 max {}",
            self.options.threads,
            std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1)
        );
        println!("option name Model type combo default fast var fast var balanced var accurate");
        println!("option name Device type combo default auto var auto var cpu var cuda");
        println!("option name DebugLog type check default false");
        println!("uciok");
    }

    pub fn isready(&self) {
        println!("readyok");
    }

    pub fn ucinewgame(&mut self) {
        self.stop_and_join_search();
        self.board = Board::default();
        self.position_history.clear();
        self.position_history.push(self.board.zobrist());
    }

    pub fn setoption(&mut self, command: &str) {
        if let Some((name, value)) = parse_option_parts(command) {
            match name.to_ascii_lowercase().as_str() {
                "hash" => {
                    if let Ok(hash_mb) = value.parse::<usize>() {
                        self.options.hash_mb = hash_mb.clamp(1, MAX_HASH_MB);
                        let info = SearchAlgorithm::hash_table_info(self.options.hash_mb);
                        println!(
                            "info string hash_config requested_mb={} effective_mb={} entries={}",
                            self.options.hash_mb, info.effective_mb, info.entries
                        );
                    }
                }
                "threads" => {
                    if let Ok(threads) = value.parse::<usize>() {
                        let max_threads = std::thread::available_parallelism()
                            .map(|count| count.get())
                            .unwrap_or(1);
                        self.options.threads = threads.clamp(1, max_threads);
                    }
                }
                "model" => {
                    if let Some((mode, alias_message)) = ModelMode::parse_with_alias_info(&value) {
                        self.options.model_mode = mode;
                        if let Some(message) = alias_message {
                            println!("info string {}", message);
                        }
                    }
                }
                "debuglog" => {
                    self.options.debug_log = parse_bool(&value).unwrap_or(self.options.debug_log);
                }
                "device" => {
                    if let Some(choice) = EvalDeviceChoice::from_str(&value) {
                        self.stop_and_join_search();
                        let should_stop = Arc::clone(&self.search_algorithm.should_stop);
                        match Self::build_search_algorithm_for_choice(choice, should_stop) {
                            Ok((search_algorithm, active_device, warning)) => {
                                self.search_algorithm = search_algorithm;
                                self.device_choice = choice;
                                self.active_device = active_device;
                                if let Some(message) = warning {
                                    println!("info string {}", message);
                                }
                                println!(
                                    "info string evaluator_device={}",
                                    device_to_label(self.active_device)
                                );
                            }
                            Err(err) => {
                                println!("info string failed to set Device: {}", err);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn position(&mut self, command: &str) {
        self.stop_and_join_search();

        let tokens: Vec<&str> = command.split_whitespace().collect();
        if tokens.len() < 2 {
            return;
        }

        match tokens[1] {
            "startpos" => {
                self.board = Board::default();
                self.position_history.clear();
                self.position_history.push(self.board.zobrist());

                let mut idx = 2;
                if idx < tokens.len() && tokens[idx] == "moves" {
                    idx += 1;
                    self.apply_moves_and_track(&tokens[idx..]);
                }
            }
            "fen" => {
                let mut idx = 2;
                let mut fen_parts = Vec::new();
                while idx < tokens.len() && tokens[idx] != "moves" {
                    fen_parts.push(tokens[idx]);
                    idx += 1;
                }
                let fen = fen_parts.join(" ");
                match Board::from_fen(&fen) {
                    Ok(board) => {
                        self.board = board;
                        self.position_history.clear();
                        self.position_history.push(self.board.zobrist());
                    }
                    Err(_) => {
                        println!("info string Invalid FEN");
                        return;
                    }
                }

                if idx < tokens.len() && tokens[idx] == "moves" {
                    idx += 1;
                    self.apply_moves_and_track(&tokens[idx..]);
                }
            }
            _ => {}
        }
    }

    pub fn go(&mut self, command: &str, tx: Sender<BitMove>) {
        self.stop_and_join_search();

        let go_options = self.parse_go_options(command);
        let request = self.build_search_request(&go_options);

        let board_clone = self.board.parallel_clone();
        let history = self.position_history.clone();
        let search_algorithm = self.search_algorithm.clone();
        let options = self.options.clone();

        self.search_handle = Some(thread::spawn(move || {
            let result = search_algorithm.search(&board_clone, request, &options, &history);
            if options.debug_log {
                println!(
                    "info string depth={} score_cp={} nodes={} elapsed_ms={} eval_calls={} eval_cache_hits={} tt_probes={} tt_hits={} q_nodes={} beta_cutoffs={}",
                    result.depth,
                    result.score_cp,
                    result.nodes,
                    result.elapsed.as_millis(),
                    result.stats.eval_calls,
                    result.stats.eval_cache_hits,
                    result.stats.tt_probes,
                    result.stats.tt_hits,
                    result.stats.q_nodes,
                    result.stats.beta_cutoffs
                );
            }
            let _ = tx.send(result.best_move);
        }));
    }

    pub fn make_move(&mut self, best_move: BitMove) {
        if !best_move.is_null() {
            self.board.apply_move(best_move);
            self.position_history.push(self.board.zobrist());
        }
        self.finish_search_if_done();
    }

    pub fn stop(&mut self) {
        self.search_algorithm
            .should_stop
            .store(true, Ordering::Relaxed);
    }

    pub fn quit(&mut self) {
        self.stop_and_join_search();
    }

    pub fn finish_search_if_done(&mut self) {
        if self
            .search_handle
            .as_ref()
            .map(|handle| handle.is_finished())
            .unwrap_or(false)
        {
            if let Some(handle) = self.search_handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn stop_and_join_search(&mut self) {
        self.stop();
        if let Some(handle) = self.search_handle.take() {
            let _ = handle.join();
        }
        self.search_algorithm
            .should_stop
            .store(false, Ordering::Relaxed);
    }

    fn apply_moves_and_track(&mut self, moves: &[&str]) {
        for mv in moves {
            if self.board.apply_uci_move(mv) {
                self.position_history.push(self.board.zobrist());
            } else {
                println!("info string Invalid move {}", mv);
                break;
            }
        }
    }

    fn parse_go_options(&self, command: &str) -> GoOptions {
        let mut options = GoOptions::default();
        let mut iter = command.split_whitespace().skip(1);

        while let Some(token) = iter.next() {
            match token {
                "depth" => options.depth = iter.next().and_then(|value| value.parse::<u32>().ok()),
                "movetime" => {
                    options.movetime_ms = iter.next().and_then(|value| value.parse::<u64>().ok())
                }
                "wtime" => {
                    options.wtime_ms = iter.next().and_then(|value| value.parse::<u64>().ok())
                }
                "btime" => {
                    options.btime_ms = iter.next().and_then(|value| value.parse::<u64>().ok())
                }
                "winc" => options.winc_ms = iter.next().and_then(|value| value.parse::<u64>().ok()),
                "binc" => options.binc_ms = iter.next().and_then(|value| value.parse::<u64>().ok()),
                "movestogo" => {
                    options.movestogo = iter.next().and_then(|value| value.parse::<u32>().ok())
                }
                "infinite" => options.infinite = true,
                _ => {}
            }
        }

        options
    }

    fn build_search_request(&self, options: &GoOptions) -> SearchRequest {
        let max_depth = options.depth.unwrap_or(64).max(1);

        if options.infinite {
            return SearchRequest {
                max_depth,
                soft_time_ms: None,
                hard_time_ms: None,
            };
        }

        if let Some(movetime_ms) = options.movetime_ms {
            let adjusted = movetime_ms
                .saturating_sub(safety_margin(movetime_ms))
                .max(1);
            return SearchRequest {
                max_depth,
                soft_time_ms: Some(adjusted),
                hard_time_ms: Some(adjusted),
            };
        }

        if let Some(allocated) = self.allocate_clock_time_ms(options) {
            let hard = allocated.saturating_sub(safety_margin(allocated)).max(1);
            let soft = hard.saturating_mul(9) / 10;
            return SearchRequest {
                max_depth,
                soft_time_ms: Some(soft.max(1)),
                hard_time_ms: Some(hard),
            };
        }

        if options.depth.is_some() {
            return SearchRequest {
                max_depth,
                soft_time_ms: None,
                hard_time_ms: None,
            };
        }

        let adjusted = DEFAULT_FALLBACK_MOVETIME_MS
            .saturating_sub(safety_margin(DEFAULT_FALLBACK_MOVETIME_MS))
            .max(1);
        SearchRequest {
            max_depth,
            soft_time_ms: Some(adjusted),
            hard_time_ms: Some(adjusted),
        }
    }

    fn allocate_clock_time_ms(&self, options: &GoOptions) -> Option<u64> {
        let (remaining_ms, increment_ms) = match self.board.turn() {
            Player::White => (options.wtime_ms?, options.winc_ms.unwrap_or(0)),
            Player::Black => (options.btime_ms?, options.binc_ms.unwrap_or(0)),
        };

        let moves_to_go = options.movestogo.unwrap_or(30).max(1) as u64;
        let base = remaining_ms / moves_to_go;
        let increment_bonus = increment_ms.saturating_mul(4) / 5;
        let mut allocated = base.saturating_add(increment_bonus).max(20);

        let cap = remaining_ms.saturating_mul(7) / 10;
        allocated = allocated.min(cap.max(20));
        Some(allocated)
    }

    fn build_search_algorithm_for_choice(
        choice: EvalDeviceChoice,
        should_stop: Arc<AtomicBool>,
    ) -> Result<(SearchAlgorithm, Device, Option<String>), String> {
        let target = resolve_device_for_choice(choice);
        let mut warning = device_choice_warning(choice, target);

        match Self::build_search_algorithm_on_device(target, Arc::clone(&should_stop)) {
            Ok(search_algorithm) => Ok((search_algorithm, target, warning)),
            Err(err) if target != Device::Cpu => {
                warning = Some(match warning {
                    Some(existing) => {
                        format!("{existing}; fallback to cpu because load failed: {err}")
                    }
                    None => format!(
                        "falling back to cpu because loading models on {} failed: {}",
                        device_to_label(target),
                        err
                    ),
                });
                let search_algorithm =
                    Self::build_search_algorithm_on_device(Device::Cpu, should_stop)?;
                Ok((search_algorithm, Device::Cpu, warning))
            }
            Err(err) => Err(err),
        }
    }

    fn build_search_algorithm_on_device(
        device: Device,
        should_stop: Arc<AtomicBool>,
    ) -> Result<SearchAlgorithm, String> {
        let small_evaluator = CModule::load_on_device(SMALL_MODEL_PATH, device).map_err(|err| {
            format!(
                "small model load failed on {}: {err}",
                device_to_label(device)
            )
        })?;
        let large_evaluator = CModule::load_on_device(LARGE_MODEL_PATH, device).map_err(|err| {
            format!(
                "large model load failed on {}: {err}",
                device_to_label(device)
            )
        })?;

        Self::probe_model_forward(&small_evaluator, device, "small")?;
        Self::probe_model_forward(&large_evaluator, device, "large")?;

        let small_evaluator = Arc::new(small_evaluator);
        let large_evaluator = Arc::new(large_evaluator);
        Ok(SearchAlgorithm::new(
            small_evaluator,
            large_evaluator,
            device,
            should_stop,
        ))
    }

    fn probe_model_forward(model: &CModule, device: Device, label: &str) -> Result<(), String> {
        let probe = tch::Tensor::from_slice(&[0_f32; 775])
            .view([1, 775])
            .to_device(device);
        let _ = model.forward_ts(&[probe]).map_err(|err| {
            format!(
                "{} model forward probe failed on {}: {}",
                label,
                device_to_label(device),
                err
            )
        })?;
        Ok(())
    }
}

#[derive(Default)]
struct GoOptions {
    depth: Option<u32>,
    movetime_ms: Option<u64>,
    wtime_ms: Option<u64>,
    btime_ms: Option<u64>,
    winc_ms: Option<u64>,
    binc_ms: Option<u64>,
    movestogo: Option<u32>,
    infinite: bool,
}

fn parse_option_parts(command: &str) -> Option<(String, String)> {
    let mut tokens = command.split_whitespace().peekable();
    if tokens.next()? != "setoption" {
        return None;
    }

    let mut name_parts = Vec::new();
    let mut value_parts = Vec::new();
    let mut in_name = false;
    let mut in_value = false;

    for token in tokens {
        match token {
            "name" => {
                in_name = true;
                in_value = false;
            }
            "value" => {
                in_name = false;
                in_value = true;
            }
            _ if in_name => name_parts.push(token),
            _ if in_value => value_parts.push(token),
            _ => {}
        }
    }

    let name = name_parts.join(" ");
    let value = value_parts.join(" ");
    if name.is_empty() {
        None
    } else {
        Some((name, value))
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "on" | "yes" => Some(true),
        "false" | "0" | "off" | "no" => Some(false),
        _ => None,
    }
}

fn safety_margin(time_ms: u64) -> u64 {
    (time_ms / 25).clamp(5, 50)
}

fn resolve_device_for_choice(choice: EvalDeviceChoice) -> Device {
    match choice {
        EvalDeviceChoice::Auto => {
            if tch::Cuda::is_available() {
                Device::Cuda(0)
            } else {
                Device::Cpu
            }
        }
        EvalDeviceChoice::Cpu => Device::Cpu,
        EvalDeviceChoice::Cuda => {
            if tch::Cuda::is_available() {
                Device::Cuda(0)
            } else {
                Device::Cpu
            }
        }
    }
}

fn device_choice_warning(choice: EvalDeviceChoice, resolved: Device) -> Option<String> {
    match choice {
        EvalDeviceChoice::Cuda if !resolved.is_cuda() => {
            Some("cuda requested but unavailable; using cpu".to_string())
        }
        _ => None,
    }
}

fn device_to_label(device: Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Mps => "mps",
        Device::Vulkan => "vulkan",
    }
}
