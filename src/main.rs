use std::io::{self, BufRead};
use std::sync::mpsc::{channel, TryRecvError};
use std::thread;
use engine::Engine;
use pleco::BitMove;

mod search_algorithm;
mod utils;
mod engine;

fn main() {
    // 1. Initialize the engine
    let mut engine = Engine::new("models/eval_params264k_norm_mse0.117666_jit.pt");

    // Create a channel for communicating best moves
    let (tx, rx) = channel::<BitMove>();

    // Spawn a new thread to read commands from stdin
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let command = line.unwrap();
            match command.split_whitespace().next() {
                Some("uci") => engine.uci(),
                Some("isready") => engine.isready(),
                Some("ucinewgame") => engine.ucinewgame(),
                Some("position") => engine.position(&command),
                Some("go") => engine.go(&command, tx.clone()),  // Pass the Sender to go
                // Some("bestmove") => engine.bestmove(&command),
                Some("stop") => engine.stop(),
                Some("quit") => break,
                _ => {}
            }
        }
        // Clean up
        engine.stop();
    });

    // Main loop to update game state and receive best moves
    loop {
        match rx.try_recv() {
            Ok(best_move) => {
                // Update the board with the best move
                println!("Best move received: {}", best_move);
            },
            Err(TryRecvError::Empty) => {
                // No message received
            },
            Err(TryRecvError::Disconnected) => {
                // The Sender has disconnected, break out of the loop
                break;
            }
        }
        // Here, you can add other game update logic if needed
    }
}
