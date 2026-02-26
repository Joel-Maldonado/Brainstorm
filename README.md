# Brainstorm Chess Engine

This experimental chess engine, developed in Rust, integrates deep learning with traditional search methods to investigate new approaches to chess AI.

<img width="512" alt="Screenshot 2024-10-23 at 6 04 59 PM" src="[https://github.com/user-attachments/assets/b22f3bfb-d127-4862-a539-0a2b10d19ec0](https://github.com/user-attachments/assets/b22f3bfb-d127-4862-a539-0a2b10d19ec0)">

## The Idea

Traditional chess engines such as Stockfish excel by using fast, hand-crafted evaluation functions and deep search trees. This project takes a different approach by employing a **neural network**, trained on 37 million chess positions, as the evaluation function. This method prioritizes pattern recognition over computational speed.

The central question is whether encoding positional understanding directly into the evaluation function can offset reduced search depth.

## How It Works

The engine combines two main components:

1. **Neural Network Evaluation**
* Trained on 37 million chess positions
* Focuses on pattern recognition and positional understanding
* Exchanges speed for potentially greater positional insight


2. **Classical Chess Search**
* Modified minimax with alpha-beta pruning
* Operates at a lower search depth than traditional engines
* Standard optimizations: transposition tables, killer moves



## Requirements

* **libtorch** (PyTorch C++ library)
* **(Optional)** To play against the engine using the GUI:
* pygame
* python-chess



## Setup & Installation

1. **To install libtorch:**
* Download libtorch from the PyTorch website
* Export the environment variable:


```bash
export LIBTORCH=/path/to/libtorch

```


2. **Build the engine (recommended):**
```bash
./build.sh --install-deps

```


3. **Install the required Python dependencies for the GUI:**
```bash
pip install pygame python-chess

```



## Playing Against the Engine

1. Build the engine as outlined above.
2. Ensure the `brainstorm` executable is located in the root directory.
3. To launch the GUI:
```bash
./run-gui.sh

```



## UCI Compatibility

The compiled engine is **UCI-compatible** and can be used with any chess GUI that supports the UCI protocol, such as Arena or Cutechess.

### UCI Options

The engine now exposes the following UCI options:

* `Hash` (MB)
* `Threads`
* `Model` (`small`, `large`, `hybrid_root`)
* `DebugLog` (`true`/`false`)

It also supports full go-time controls:

* `movetime`
* `wtime`, `btime`, `winc`, `binc`, `movestogo`
* `depth`
* `infinite`

### Benchmark / Regression Scripts

* `scripts/bench_uci.py` for timing and node-rate benchmarks.
* `scripts/strength_suite.py` for a tactical regression suite.
* `tests/test_uci_protocol.py` and `tests/test_time_control.py` for CI smoke tests.

## Current Status

This experimental project aims to explore alternative approaches to chess engine design. While the engine is capable of playing chess, the primary objective is to test concepts related to deep learning in chess, rather than to compete with established engines.

**Current areas of exploration:**

* Board representation techniques
* Neural network architecture experiments
* Search optimization techniques
* Evaluation speed improvements
