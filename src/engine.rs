use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use dashmap::DashMap;
use pleco::BitMove;

use crate::search_algorithm::SearchAlgorithm;

pub struct Engine {
    pub board: pleco::Board,
    pub search_algorithm: SearchAlgorithm,
}


impl Engine {
    pub fn new(model_path: &str) -> Self {
        let board = pleco::Board::default();
        let evaluator = Arc::new(tch::CModule::load(model_path).unwrap());
        let should_stop = Arc::new(AtomicBool::new(false));
        let search_algorithm = SearchAlgorithm::new(evaluator, Some(should_stop), Arc::new(DashMap::new()));
        
        Engine {
            board,
            search_algorithm,
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
        self.search_algorithm.transposition_table.clear();
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

        for move_str in words {
            if !self.board.apply_uci_move(move_str) {
                println!("Invalid move: {}", move_str);
            }
        }

        let fen = self.board.fen();
        let mut counter = self.search_algorithm.position_count.entry(fen).or_insert(0);
        *counter += 1;

    }

    pub fn go(&mut self, command: &str, tx: Sender<BitMove>) {
        let options = self.parse_go_options(command);
        
        // Cloning necessary states and objects for the new thread
        let mut board_clone = self.board.clone();
        let evaluator_clone = self.search_algorithm.evaluator.clone();
        let should_stop_clone = self.search_algorithm.should_stop.clone();
        let position_count_clone = self.search_algorithm.position_count.clone();  // New line
    
        // Set default values if options are not provided
        let max_time = options.max_time_ms.unwrap_or(10_000);
        let max_depth = options.max_depth.unwrap_or(5);
    
        // Spawn a new thread to handle the search
        let _ = thread::spawn(move || {
            // Initialize a new SearchAlgorithm instance with cloned states and objects
            let search_algo = SearchAlgorithm::new(evaluator_clone, Some(should_stop_clone), position_count_clone); // Modified line
            let best_move = search_algo.search(&mut board_clone, max_depth, max_time);
            
            // Send the best move back via the channel
            tx.send(best_move).expect("Could not send best move");
        });
    }
    

    pub fn make_move(&mut self, best_move: BitMove) {
        self.board.apply_move(best_move);
        let fen = self.board.fen();
        let mut counter = self.search_algorithm.position_count.entry(fen).or_insert(0);
        *counter += 1;
    }

    pub fn stop(&mut self) {
        self.search_algorithm.should_stop.store(true, Ordering::Relaxed);
    }

    fn parse_go_options(&self, command: &str) -> GoOptions {
        println!("Received GO Command: {}", command);

        let mut options = GoOptions {
            max_time_ms: None,
            max_depth: None,
        };

        let mut iter = command.split_whitespace();
        iter.next(); // Skip the "go" command itself

        while let Some(option) = iter.next() {
            match option {
                "depth" => options.max_depth = iter.next().and_then(|s| s.parse().ok()),
                "movetime" => options.max_time_ms = iter.next().and_then(|s| s.parse().ok()),
                "infinite" =>  {
                    options.max_time_ms = Some(999_000);
                    options.max_depth = Some(999);
                },
                _ => {}
            }
        }

        options
    }
}


struct GoOptions {
    max_time_ms: Option<u128>,
    max_depth: Option<u32>,
}
