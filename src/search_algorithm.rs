use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use pleco::BitMove;
use tch::nn::Module;
use pleco::Player;
use rayon::prelude::*;
use crate::utils::{board_to_bitboard, order_moves};

pub struct SearchAlgorithm {
    pub evaluator: Arc<tch::CModule>,
    pub transposition_table: Arc<Mutex<HashMap<String, (f32, u32)>>>,
    killer_moves: Arc<Mutex<HashMap<u32, BitMove>>>,
    pub should_stop: Arc<AtomicBool>,
}

impl SearchAlgorithm {
    pub fn new(evaluator: Arc<tch::CModule>, should_stop: Option<Arc<AtomicBool>>) -> Self {
        let transposition_table = Arc::new(Mutex::new(HashMap::new()));
        let killer_moves = Arc::new(Mutex::new(HashMap::new()));
        let should_stop = should_stop.unwrap_or(Arc::new(AtomicBool::new(false)));

        SearchAlgorithm {
            evaluator,
            transposition_table,
            killer_moves,
            should_stop,
        }
    }

    pub fn minimax(&self, board: &mut pleco::Board, depth: u32, max_time: f32, start_time: Instant, mut alpha: f32, mut beta: f32, maximizing: bool, best_move: Option<BitMove>) -> f32 {
        if self.should_stop.load(Ordering::Relaxed) || (start_time.elapsed().as_secs_f32() > max_time) {
            return match board.turn() {
                Player::White => f32::MAX,
                Player::Black => f32::MIN,
            };
        }

        let hash_key = board.fen();

        if let Some(t) = self.transposition_table.lock().unwrap().get(&hash_key) {
            // t is a tuple of (value, depth)
            if t.1 >= depth {
                return t.0;
            }
        }

        if board.checkmate() {
            match board.turn() {
                pleco::Player::White => return f32::MIN,
                pleco::Player::Black => return f32::MAX,
            }
        }

        if board.stalemate() {
            return 0.0;
        }

        if depth == 0 {
            return self.evaluator.forward(&board_to_bitboard(&board)).double_value(&[]) as f32;
        }

        let killer_moves_guard = self.killer_moves.lock().unwrap().clone();
        let killer_move = killer_moves_guard.get(&depth);
        let ordered_moves = order_moves(board, killer_move, best_move);

        if maximizing {
            let mut value = f32::MIN;

            for m in ordered_moves {
                if self.should_stop.load(Ordering::Relaxed) || (start_time.elapsed().as_secs_f32() > max_time) {
                    return f32::MIN;
                }

                board.apply_move(m);
                let eval = self.minimax(board, depth - 1, max_time, start_time, alpha, beta, false, best_move);
                board.undo_move();

                value = value.max(eval);
                alpha = alpha.max(eval);
                if beta <= alpha {
                    self.killer_moves.lock().unwrap().insert(depth, m);
                    break;
                }
            }

            self.transposition_table.lock().unwrap().clone().insert(hash_key, (value, depth));
            return value;
        } else {
            let mut value = f32::MAX;

            for m in ordered_moves {
                if self.should_stop.load(Ordering::Relaxed) || (start_time.elapsed().as_secs_f32() > max_time) {
                    return f32::MAX;
                }

                board.apply_move(m);
                let eval = self.minimax(board, depth - 1, max_time, start_time, alpha, beta, true, best_move);
                board.undo_move();

                value = value.min(eval);
                beta = beta.min(eval);

                if beta <= alpha {
                    self.killer_moves.lock().unwrap().insert(depth, m);
                    break;
                }
            }

            self.transposition_table.lock().unwrap().clone().insert(hash_key, (value, depth));
            return value;
        }
    }

    pub fn search(&self, board: &mut pleco::Board, max_depth: u32, max_time: f32) -> BitMove {
        let start_time = Instant::now();
        let maximizing = match board.turn() {
            Player::White => true,
            Player::Black => false,
        };

        let mut best_move: Option<BitMove> = None;

        if maximizing {
            for depth in 1..=max_depth {
                if self.should_stop.load(Ordering::Relaxed) || (start_time.elapsed().as_secs_f32() > max_time){
                    break;
                }

                let moves: Vec<BitMove> = board.generate_moves().to_vec();

                let results: Vec<(f32, BitMove)> = moves.par_iter().map(|&m| {
                    let mut local_board = board.clone();

                    local_board.apply_move(m);
                    let eval = self.minimax(&mut local_board, depth - 1, max_time, start_time, f32::MIN, f32::MAX, false, best_move);
                    local_board.undo_move();

                    (eval, m)
                }).collect();

                if self.should_stop.load(Ordering::Relaxed) || (start_time.elapsed().as_secs_f32() > max_time) {
                    break;
                }
                
                let mut best_score = if maximizing { f32::MIN } else { f32::MAX };
                for (eval, m) in results {
                    if (maximizing && eval > best_score) || (!maximizing && eval < best_score) {
                        best_score = eval;
                        best_move = Some(m);  // Update best_move
                    }
                }
    
                
                let cp_score = best_score * 25.0 * 100.0;
                println!("info depth {} score cp {} currmove {}", depth, cp_score, best_move.unwrap());
            }
        } else {
            for depth in 1..=max_depth {
                let moves: Vec<BitMove> = board.generate_moves().to_vec();

                let results: Vec<(f32, BitMove)> = moves.par_iter().map(|&m| {
                    let mut local_board = board.clone();

                    local_board.apply_move(m);
                    let eval = self.minimax(&mut local_board, depth - 1, max_time, start_time, f32::MIN, f32::MAX, true, best_move);
                    local_board.undo_move();

                    (eval, m)

                }).collect();

                if self.should_stop.load(Ordering::Relaxed) || (start_time.elapsed().as_secs_f32() > max_time){
                    break;
                }

                let mut best_score = if maximizing { f32::MIN } else { f32::MAX };
                for (eval, m) in results {
                    if (maximizing && eval > best_score) || (!maximizing && eval < best_score) {
                        best_score = eval;
                        best_move = Some(m);  // Update best_move
                    }
                }

                let cp_score = best_score * 25.0 * 100.0;
                println!("info depth {} score cp {} currmove {}", depth, cp_score, best_move.unwrap());
            }
        }

        println!("Finished in {} seconds", start_time.elapsed().as_secs_f32());

        best_move.unwrap_or_else(|| {
            let moves = board.generate_moves();
            moves[0]
        })
    }

}
