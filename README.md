# Brainstorm Chess Engine

An experimental chess engine built using Rust. It combines deep learning with traditional search techniques to explore alternative approaches to chess AI.

<img width="512" alt="Screenshot 2024-10-23 at 6 04 59 PM" src="https://github.com/user-attachments/assets/b22f3bfb-d127-4862-a539-0a2b10d19ec0">

## The Idea

Traditional chess engines like Stockfish excel at calculating variations using fast, hand-crafted evaluation functions and deep search trees. This project explores a different approach: using a neural network trained on 37 million chess positions as the evaluation function, trading computational speed for pattern recognition.

The core question: Can we compensate for lower search depth by encoding positional understanding directly into the evaluation function?

## How It Works

The engine combines two main components:

1. **Neural Network Evaluation**
   - Trained on 37 million chess positions
   - Focuses on pattern recognition and positional understanding
   - Trades speed for potentially deeper positional insight

2. **Classical Chess Search**
   - Modified minimax with alpha-beta pruning
   - Lower search depth than traditional engines
   - Standard optimizations: transposition tables, killer moves

## Requirements
- libtorch (PyTorch C++ library)
- (optional) To play against engine in the GUI:
  - pygame
  - python-chess

## Setup & Installation

1. **Install libtorch:**
   - Download libtorch from PyTorch website
   - Export the environment variable:
   ```bash
   export LIBTORCH=/path/to/libtorch
   ```

2. **Build the engine:**
   ```bash
   cargo build --release
   cp target/release/brainstorm .
   ```

3. **Install Python dependencies for GUI:**
   ```bash
   pip install pygame python-chess
   ```

## Playing Against the Engine

1. Build the engine as described above
2. Make sure the `brainstorm` executable is in the root directory
3. Run the GUI:
   ```bash
   python scripts/gui.py
   ```

## UCI Compatibility

The compiled engine is UCI-compatible, meaning it can be used with any chess GUI that supports the UCI protocol (like Arena, Cutechess, etc.).

## Current Status

This is an experimental project focused on exploring alternative approaches to chess engine design. While it can play chess, the goal isn't to compete with established engines but rather to test ideas about deep learning in chess.

Current areas of exploration:
- Board representation techniques
- Neural network architecture experiments
- Search optimization techniques
- Evaluation speed improvements
