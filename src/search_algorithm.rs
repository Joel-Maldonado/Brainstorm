use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Instant, Duration};
use pleco::BitMove;
use pleco::Player;
use rayon::prelude::*;
use tch::nn::Module;
use crate::utils::{board_to_bitboard, order_moves};
use dashmap::DashMap;  // Import DashMap

pub struct SearchAlgorithm {
    pub evaluator: Arc<tch::CModule>,
    pub transposition_table: Arc<DashMap<String, (f32, u32)>>,  // Updated to DashMap
    killer_moves: Arc<DashMap<u32, BitMove>>,  // Updated to DashMap
    pub should_stop: Arc<AtomicBool>,
    pub position_count: Arc<DashMap<String, u32>>,  // Updated to DashMap
}


impl SearchAlgorithm {
    pub fn new(evaluator: Arc<tch::CModule>, should_stop: Option<Arc<AtomicBool>>, position_count: Arc<DashMap<String, u32>>) -> Self {
        let transposition_table = Arc::new(DashMap::new());  // Initialize DashMap
        let killer_moves = Arc::new(DashMap::new());  // Initialize DashMap
        let should_stop = should_stop.unwrap_or(Arc::new(AtomicBool::new(false)));

        SearchAlgorithm {
            evaluator,
            transposition_table,
            killer_moves,
            should_stop,
            position_count,
        }
    }

    pub fn minimax(&self, board: &mut pleco::Board, depth: u32, max_time: u128, start_time: Instant, mut alpha: f32, mut beta: f32, maximizing: bool, best_move: Option<BitMove>) -> f32 {
        if self.should_stop.load(Ordering::Relaxed) {
            return match board.turn() {
                Player::White => f32::MAX,
                Player::Black => f32::MIN,
            };
        }


        let hash_key = board.fen();

        if let Some(t) = self.transposition_table.get(&hash_key) {
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

        if let Some(count) = self.position_count.get(&hash_key) {
            if *count >= 3 {
                return 0.0;  // Draw due to threefold repetition
            }
        }
    

        if depth == 0 {
            return self.evaluator.forward(&board_to_bitboard(&board)).double_value(&[]) as f32;
        }


        // let killer_move = self.killer_moves.get(&depth);
        let killer_move = self.killer_moves.get(&depth).map(|r| *r);
        let ordered_moves = order_moves(board, killer_move, best_move);

        if maximizing {
            let mut value = f32::MIN;

            for m in ordered_moves {
                if self.should_stop.load(Ordering::Relaxed) {
                    return f32::MIN;
                }

                board.apply_move(m);
                let eval = self.minimax(board, depth - 1, max_time, start_time, alpha, beta, false, best_move);
                board.undo_move();

                value = value.max(eval);
                alpha = alpha.max(eval);
                if beta <= alpha {
                    self.killer_moves.insert(depth, m);
                    break;
                }
            }

            self.transposition_table.insert(hash_key, (value, depth));
            return value;
        } else {
            let mut value = f32::MAX;

            for m in ordered_moves {
                if self.should_stop.load(Ordering::Relaxed) {
                    return f32::MAX;
                }

                board.apply_move(m);
                let eval = self.minimax(board, depth - 1, max_time, start_time, alpha, beta, true, best_move);
                board.undo_move();

                value = value.min(eval);
                beta = beta.min(eval);

                if beta <= alpha {
                    self.killer_moves.insert(depth, m);
                    break;
                }
            }

            self.transposition_table.insert(hash_key, (value, depth));
            return value;
        }
    }

    pub fn search(&self, board: &mut pleco::Board, max_depth: u32, max_time: u128) -> BitMove {
        self.should_stop.store(false, Ordering::Relaxed);
        let start_time = Instant::now();

        let maximizing = match board.turn() {
            Player::White => true,
            Player::Black => false,
        };

        let mut best_move: Option<BitMove> = None;

        let stop_flag = self.should_stop.clone();  // Clone the Arc<AtomicBool> for the thread
        let max_time_duration = Duration::from_millis(max_time as u64);  // Convert max_time to a Duration
        
        // Spawn the timer thread
        thread::spawn(move || {
            thread::sleep(max_time_duration);
            stop_flag.store(true, Ordering::Relaxed);
        });


        if maximizing {
            for depth in 1..=max_depth {
                if self.should_stop.load(Ordering::Relaxed) || (start_time.elapsed().as_millis() > max_time){
                    break;
                }

                let moves: Vec<BitMove> = board.generate_moves().to_vec();

                let results: Vec<(f32, BitMove)> = moves.par_iter().map(|&m| {
                    let mut local_board = board.parallel_clone();

                    local_board.apply_move(m);
                    let eval = self.minimax(&mut local_board, depth - 1, max_time, start_time, f32::MIN, f32::MAX, false, best_move);
                    local_board.undo_move();

                    (eval, m)
                }).collect();

                if self.should_stop.load(Ordering::Relaxed) {
                    break;
                }
                
                let mut best_score = if maximizing { f32::MIN } else { f32::MAX };
                for (eval, m) in results {
                    if (maximizing && eval > best_score) || (!maximizing && eval < best_score) {
                        best_score = eval;
                        best_move = Some(m);  // Update best_move
                    }
                }
    
                
                let cp_score = best_score * 25.0;
                println!("info depth {} score cp {} pv {}", depth, cp_score, best_move.unwrap());
            }
        } else {
            for depth in 1..=max_depth {
                let moves: Vec<BitMove> = board.generate_moves().to_vec();

                let results: Vec<(f32, BitMove)> = moves.par_iter().map(|&m| {
                    let mut local_board = board.parallel_clone();

                    local_board.apply_move(m);
                    let eval = self.minimax(&mut local_board, depth - 1, max_time, start_time, f32::MIN, f32::MAX, true, best_move);
                    local_board.undo_move();

                    (eval, m)

                }).collect();

                if self.should_stop.load(Ordering::Relaxed) || (start_time.elapsed().as_millis() > max_time){
                    break;
                }

                let mut best_score = if maximizing { f32::MIN } else { f32::MAX };
                for (eval, m) in results {
                    if (maximizing && eval > best_score) || (!maximizing && eval < best_score) {
                        best_score = eval;
                        best_move = Some(m);
                    }
                }

                let cp_score = best_score * 25.0;
                println!("info depth {} score cp {} pv {}", depth, cp_score, best_move.unwrap());
            }
        }

        best_move.unwrap_or_else(|| {
            let moves = board.generate_moves();
            moves[0]
        })
    }

}
