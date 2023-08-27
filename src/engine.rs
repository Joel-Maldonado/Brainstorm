use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use pleco::BitMove;

use crate::search_algorithm::SearchAlgorithm;

pub struct Engine {
    board: pleco::Board,
    search_algorithm: SearchAlgorithm,
    should_stop: Arc<AtomicBool>,
}


impl Engine {
    pub fn new(model_path: &str) -> Self {
        let board = pleco::Board::default();
        let evaluator = Arc::new(tch::CModule::load(model_path).unwrap());
        let search_algorithm = SearchAlgorithm::new(evaluator);
        let should_stop = Arc::new(AtomicBool::new(false));
        
        Engine {
            board,
            search_algorithm,
            should_stop,
        }
    }

    pub fn uci(&self) {
        println!("id name Brainstorm");
        println!("id author JoelM");
        println!("uciok");
    }

    pub fn isready(&self) {
        println!("readyok");
    }
    
    pub fn ucinewgame(&mut self) {
        self.board = pleco::Board::default();
        self.search_algorithm.transposition_table.lock().unwrap().clear();
    }

    pub fn position(&mut self, command: &str) {
        let mut words = command.split_whitespace();
        words.next(); // Skip the "position" word

        match words.next() {
            Some("fen") => {
                let fen: String = words.clone().take_while(|&word| word != "moves").collect::<Vec<_>>().join(" ");
                self.board = pleco::Board::from_fen(&fen).expect("Invalid FEN");
            },
            Some("startpos") => {
                self.board = pleco::Board::default();
                words.next(); // Skip the "moves" word if present
            },
            _ => return, // Malformed command
        }

        // Apply the moves
        for move_str in words {
            if !self.board.apply_uci_move(move_str) {
                println!("Invalid move: {}", move_str);
            }
        }
    }

    pub fn go(&mut self, command: &str, tx: Sender<BitMove>) {
        let options = self.parse_go_options(command);
        
        let mut board_clone = self.board.clone();
        let evaluator_clone = self.search_algorithm.evaluator.clone();
        let search_algo = SearchAlgorithm::new(evaluator_clone);

        let max_time = options.max_time.unwrap_or(999.0);  // set a default
        let max_depth = options.max_depth.unwrap_or(4);  // set a default

        // Spawn a new thread to handle the search
        let handle = thread::spawn(move || {
            let best_move = search_algo.search(&mut board_clone, max_depth, max_time);
            
            // Send the best move back via the channel
            tx.send(best_move).expect("Could not send best move");
        });

        // Optional: wait for the thread to complete
        // let _ = handle.join();
    }

    pub fn stop(&mut self) {
        self.should_stop.store(true, Ordering::Relaxed);
    }

    fn parse_go_options(&self, command: &str) -> GoOptions {
        let mut options = GoOptions {
            max_time: None,
            max_depth: None,
        };

        let mut iter = command.split_whitespace();
        iter.next(); // Skip the "go" command itself

        while let Some(option) = iter.next() {
            match option {
                "depth" => options.max_depth = iter.next().and_then(|s| s.parse().ok()),
                "movetime" => options.max_time = iter.next().and_then(|s| s.parse().ok()),
                _ => {}
            }
        }

        options
    }
}


struct GoOptions {
    max_time: Option<f32>,
    max_depth: Option<u32>,
}
