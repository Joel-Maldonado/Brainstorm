use std::io::{self, BufRead};
use std::sync::{Arc, Mutex, mpsc::{channel, TryRecvError}};
use std::thread;
use pleco::BitMove;

mod search_algorithm;
mod utils;
mod engine;

use engine::Engine;

fn main() {
    // Initialize the engine and wrap it in an Arc<Mutex<>>.
    let engine = Arc::new(Mutex::new(Engine::new("/home/rnoc/Projects/rust/brainstorm/models/eval_params264k_norm_mse0.117666_jit.pt")));

    // Create a channel for communicating best moves
    let (tx, rx) = channel::<BitMove>();

    // Create another Arc for the thread
    let engine_for_thread = Arc::clone(&engine);

    // Spawn a new thread to read commands from stdin (Loops every time a new command is received)
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let command = line.unwrap();
            println!("Received command: {}", command);
            
            let mut engine = engine_for_thread.lock().unwrap();  // Lock the Mutex
            match command.split_whitespace().next() {
                Some("uci") => engine.uci(),
                Some("isready") => engine.isready(),
                Some("ucinewgame") => engine.ucinewgame(),
                Some("position") => engine.position(&command),
                Some("go") => engine.go(&command, tx.clone()),  // Pass the Sender to go
                Some("stop") => engine.stop(),
                Some("quit") => break,
                Some("fen") => println!("{}", engine.board.fen()),
                _ => {}
            }
        }
        // Clean up
        engine_for_thread.lock().unwrap().stop();
    });

    // Main loop to update game state and receive best moves (Constantly updates)
    loop {
        match rx.try_recv() {
            Ok(best_move) => {
                println!("bestmove {}", best_move.to_string());
                let mut engine = engine.lock().unwrap();  // Lock the Mutex
                engine.make_move(best_move);
            },
            Err(TryRecvError::Empty) => {
                // No message received
            },
            Err(TryRecvError::Disconnected) => {
                // The Sender has disconnected, break out of the loop
                break;
            }
        }

        // Sleep to avoid hogging the CPU
        thread::sleep(std::time::Duration::from_millis(10));
    }
}
