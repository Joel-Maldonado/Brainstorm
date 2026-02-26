use pleco::BitMove;
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{
    mpsc::{channel, TryRecvError},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;

mod engine;
mod search_algorithm;
mod utils;

use engine::Engine;

fn main() {
    let engine = Arc::new(Mutex::new(Engine::new()));
    let (tx, rx) = channel::<BitMove>();
    let running = Arc::new(AtomicBool::new(true));

    let engine_for_input = Arc::clone(&engine);
    let running_for_input = Arc::clone(&running);

    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let command = match line {
                Ok(command) => command,
                Err(_) => continue,
            };

            let mut engine = engine_for_input.lock().unwrap();
            match command.split_whitespace().next() {
                Some("uci") => engine.uci(),
                Some("isready") => engine.isready(),
                Some("setoption") => engine.setoption(&command),
                Some("ucinewgame") => engine.ucinewgame(),
                Some("position") => engine.position(&command),
                Some("go") => engine.go(&command, tx.clone()),
                Some("stop") => engine.stop(),
                Some("quit") => {
                    engine.quit();
                    running_for_input.store(false, Ordering::Relaxed);
                    break;
                }
                Some("fen") => println!("{}", engine.board.fen()),
                _ => {}
            }
        }
        running_for_input.store(false, Ordering::Relaxed);
    });

    while running.load(Ordering::Relaxed) {
        match rx.try_recv() {
            Ok(best_move) => {
                if best_move.is_null() {
                    println!("bestmove 0000");
                } else {
                    println!("bestmove {}", best_move);
                }
                let mut engine = engine.lock().unwrap();
                engine.make_move(best_move);
            }
            Err(TryRecvError::Empty) => {
                let mut engine = engine.lock().unwrap();
                engine.finish_search_if_done();
            }
            Err(TryRecvError::Disconnected) => break,
        }

        thread::sleep(Duration::from_millis(2));
    }

    {
        let mut engine = engine.lock().unwrap();
        engine.quit();
    }
}
