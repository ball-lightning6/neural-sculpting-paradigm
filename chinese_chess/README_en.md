# Chinese Chess

## Overall Architecture

This project builds a complete training loop for Chinese Chess AI. The core philosophy is to transfer knowledge from strong traditional engines (like PikaFish) to neural networks (Policy Transformer) via **"Expert Iteration"** and **"Soft Label Distillation"**.

The workflow consists of three main stages:
1.  **Foundation Training**: Using random legal moves (`generate_chess_positions_by_random_moves.py`) to teach the model basic rules and legal move space.
2.  **Expert Data Generation**: Using `generate_chess_positions_from_engine_self_play.py` to invoke UCI engines for self-play, generating high-quality moves and evaluation data.
3.  **Policy Distillation**: Generating probabilistic Soft Labels via `generate_soft_labels.py` to train the Policy Network to imitate engine search results.

Additionally, `play_with_ai.py` is provided for verifying model performance in real games, and specific tactical training tasks like `generate_chess_resolve_check_task.py` (Resolve Check) are included.

---


## 1. **chinese_chess/generate_chess_positions_by_random_moves.py**

- **Purpose:** Quickly generates a large number of plausible, legal Chinese chess positions by simulating a completely random player making moves.
    
- **Logic:** The script starts from the standard Chinese chess starting position. In a loop, it gets all legal moves in the current position, then randomly selects one to execute. This process repeats max_steps times, finally obtaining a random but legal position.
    
- **I/O Format:**
    
    - Output: FEN format position string.
        
- **Main Parameters:** max_steps, max_capture.

---

## 2. **chinese_chess/generate_chess_positions_by_random_placement.py**

- **Purpose:** Generates a large number of atypical but mostly legal Chinese chess positions by randomly placing pieces on the board (rather than simulating moves), used for stress testing the model's robustness.
    
- **Logic:** Instead of generating positions through moves, this script directly places pieces randomly on the board following piece position constraints and the rule that kings cannot face each other, thereby creating a large number of positions that rarely appear in real games but are syntactically legal.
    
- **I/O Format:**
    
    - Output: FEN format position string.
        
- **Main Parameters:** num_fens.

---

## 3. **chinese_chess/generate_chess_positions_from_engine_self_play.py**

- **Purpose:** Generates a large number of high-quality, combat-logic-compliant Chinese chess positions (FEN format) as basic data source for training chess AI.
    
- **Logic:** Simulates tens of thousands of high-level self-play games by calling a powerful third-party chess engine (PikaFish) through subprocess. During simulation, it records the FEN representation of each move in the game, thereby building a large and realistic position database.
    
- **I/O Format:**
    
    - Output: A .txt file, each line containing a complete FEN string.
        
- **Main Parameters:** num_games, max_steps, depth.

---

## 4. **chinese_chess/generate_preprocess_legal_moves.py**

- **Purpose:** This is a data preprocessing script for converting FEN format position datasets into a "legal move prediction" task that models can directly learn.
    
- **Logic:** Reads a FEN file, for each position uses the cchess library to parse and generate all legal moves. Then, according to a global mapping file, converts each specific move (like 'h2e2') into a unique integer ID.
    
- **I/O Format:**
    
    - Input: .txt file, one FEN per line.
        
    - Output: .jsonl file, each JSON object contains fen and its corresponding legal_move_ids list.
        
- **Main Parameters:** fen_file, output_file.

---

## 5. **chinese_chess/generate_chess_resolve_check_task.py**

- **Purpose:** Generates a dataset specifically targeting the "resolving a check" tactical scenario in Chinese chess. This task requires the model to find all legal moves that can resolve the check when in a checked state.
    
- **Logic:** The script first filters from a large random position library, only keeping positions that satisfy the condition of "currently being checked but not checkmated (not stalemate)." Then, for each filtered position, it calculates all legal moves that can resolve the check and saves the IDs of these moves.
    
- **I/O Format:**
    
    - Output: A .jsonl file. Each JSON object contains fen (position) and legal_move_ids (an integer list representing all legal check-resolving moves).
        
- **Main Parameters:** fen_file, output_file.

---

## 6. **chinese_chess/worker_logic.py**

- **Purpose:** A core **engine interaction and task scheduling module**, bridging the gap between engine and data. It is used for both generating FEN positions and calculating soft labels for moves.

- **Logic:** 
    - **Encapsulate Pikafish Engine:** `PikaFishEngineFinal` class encapsulates the UCI communication protocol with the Pikafish international-level chess engine. It uses multiprocess-safe pipes to interact, sending instructions like `go depth` and robustly parsing `bestmove` and `score` information returned by the engine.
    - **Breaking Determinism:** Dynamically sets Hash size at initialization to break determinism for generating diverse data.
    - **Multiprocess Worker:** 
        - `worker_label_generation`: A Worker function designed for `generate_soft_labels.py`. It receives assigned FEN chunks, starts independent engine processes, calculates MultiPV (multiple candidates) moves and their scores for each position, generates soft labels, and writes to temporary files.

- **Key Features:** Efficient multiprocess support, robust UCI protocol handling, supports MultiPV score retrieval.

---

## 7. **chinese_chess/generate_soft_labels.py**

- **Purpose:** This is a **Knowledge Distillation** data generation script. Its purpose is to transfer knowledge from a powerful traditional chess engine (Teacher) into a format (soft labels) suitable for neural network (Student) learning.

- **Logic:**
    - **Massive Parallelism:** Uses `multiprocessing.Pool` to distribute millions of FEN positions to multiple CPU cores.
    - **Soft Label Calculation:** For each position, calls the engine to calculate scores (CP Score) for Top-K (e.g., MultiPV=5) best moves.
    - **Softmax Normalization:** Converts absolute scores (CP score) from the engine into a probability distribution via Softmax function with temperature. The temperature parameter `temperature` controls smoothness: high temperature makes distribution smoother (retaining more info on sub-optimal moves), low temperature makes it sharper (focusing only on best moves).
    - **Mapping & Saving:** Maps move strings to vocabulary indices and finally saves as JSONL format.

- **I/O Format:**
    - Input: FEN file (one position per line) + `move2idx.json` (vocabulary).
    - Output: JSONL file, each line contains `{"fen": "...", "label": [0.01, 0.95, ...]}`.

- **Main Parameters:** `pikafish_engine_path`, `input_fen_file`, `engine_depth` (search depth), `multipv_count` (candidate count), `temperature`.

---

## 8. **chinese_chess/play_with_ai.py**

- **Purpose:** This is a **Human-AI Battle Terminal Client** for testing and evaluating the trained policy network.

- **Logic:**
    - **Load Model:** Automatically loads the latest training checkpoint.
    - **Position Encoding:** Real-time encoding of current board state (FEN) into model input tensor.
    - **Policy Sampling:** Model outputs probability distribution for all next moves. Script adjusts probabilities based on `sampling_temperature` and selects move using Multinomial Sampling, achieving a diverse and high-level playing style.
    - **Interaction Loop:** Provides a simple command-line interface handling user input (UCI format, e.g., `h2e2`) and displaying AI moves.

- **Interaction Mode:** Command line input.
- **Main Configuration:** `model_dir`, `sampling_temperature`.