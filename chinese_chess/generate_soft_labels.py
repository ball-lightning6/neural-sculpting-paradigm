import os
import glob
import json
import math
import time
import threading
from multiprocessing import Pool, cpu_count, set_start_method, Manager
from tqdm import tqdm

# Import the worker function and engine from worker_logic
# Make sure chinese_chess/worker_logic.py is in PYTHONPATH or copied to the same directory
from chinese_chess.worker_logic import worker_label_generation, PikaFishEngineFinal

# ==============================================================================
# --- Configuration (Modify as needed) ---
# ==============================================================================

CONFIG_S2 = {
    "pikafish_engine_path": r"/root/pikafish/pikafish-avx2", # Modify this path
    "move_to_idx_file": "move2idx.json",
    "input_fen_file": "dataset_1048_5m.txt",  # Output from stage 1
    "final_output_file": "train_data_soft_labels.jsonl",
    "temp_dir": "./temp_labels", # Temporary directory for soft labels
    "engine_depth": 10,
    "multipv_count": 5,
    "temperature": 20.0,
    "num_processes": max(1, cpu_count() - 2) # Leave some cores for system
}

# ==============================================================================
# --- Main Execution Function ---
# ==============================================================================

def monitor_progress(pbar, counter, total):
    """Independent monitoring thread to update tqdm progress bar."""
    last_val = 0
    while last_val < total:
        time.sleep(0.5) # Update every half second
        current_val = counter.value
        pbar.update(current_val - last_val)
        last_val = current_val
    pbar.update(total - last_val) # Ensure progress reaches 100%

def run_stage2():
    print("--- 🚀 [Stage 2] Start: Multiprocess Soft Label Generation (Live Progress) ---")
    cfg = CONFIG_S2

    # 1. Load required files
    try:
        with open(cfg["move_to_idx_file"], 'r') as f: move_map = json.load(f)
        with open(cfg["input_fen_file"], 'r') as f: 
            # Read all lines, stripping whitespace
            fens_to_process = [line.strip() for line in f if line.strip()]
            # fens_to_process = fens_to_process[:10000] # Uncomment for testing small batch
    except FileNotFoundError as e:
        print(f"❌ Error: Missing required file: {e.filename}. Please ensure file exists and path is correct.")
        return

    # 2. Prepare temp directory and tasks
    os.makedirs(cfg["temp_dir"], exist_ok=True)
    # Clean up old temp files
    for f in glob.glob(os.path.join(cfg["temp_dir"], "*.jsonl")): os.remove(f)

    # 3. Create multiprocess manager and shared counter
    with Manager() as manager:
        progress_counter = manager.Value('i', 0)

        # Split FEN list into chunks, one for each process
        chunk_size = math.ceil(len(fens_to_process) / cfg["num_processes"])
        tasks = []
        for i in range(cfg["num_processes"]):
            chunk_start = i * chunk_size
            chunk_end = chunk_start + chunk_size
            fen_chunk = fens_to_process[chunk_start:chunk_end]
            if fen_chunk: # Only create task if chunk is not empty
                tasks.append((i, fen_chunk, move_map, cfg, progress_counter))
        
        print(f"Found {len(fens_to_process)} FENs, split into {len(tasks)} chunks, using {cfg['num_processes']} processes.")

        # 4. Start process pool and monitor thread
        with tqdm(total=len(fens_to_process), desc="Annotating FENs") as pbar:
            with Pool(processes=cfg["num_processes"]) as pool:
                monitor_thread = threading.Thread(target=monitor_progress, args=(pbar, progress_counter, len(fens_to_process)), daemon=True)
                monitor_thread.start()
                
                # imap_unordered provides best performance for progress tracking
                pool.map(worker_label_generation, tasks)
                
                # Wait for monitor thread to finish final update
                monitor_thread.join(timeout=1)

    # 5. Merge all temp files
    print("\n--- ✅ Workers finished, starting merge... ---")
    with open(cfg["final_output_file"], "w", encoding="utf-8") as f_out:
        for temp_file in tqdm(glob.glob(os.path.join(cfg["temp_dir"], "*.jsonl")), desc="Merging label files"):
            with open(temp_file, 'r', encoding='utf-8') as f_in:
                # Read and write line by line to avoid memory issues with large files
                for line in f_in:
                    f_out.write(line)
            os.remove(temp_file)
    # Cleanup temp directory
    os.rmdir(cfg["temp_dir"])

    print(f"\n🎉 [Stage 2] Completed! Soft label dataset generated.")
    print(f"   Output file: {cfg['final_output_file']}")

if __name__ == "__main__":
    # On Linux, 'fork' mode is usually defined. Since specific architecture is decoupled, 'fork' is safe here.
    if os.name != 'nt':
        try:
            set_start_method('fork', force=True)
        except RuntimeError:
            pass # Method might already be set
    run_stage2()
