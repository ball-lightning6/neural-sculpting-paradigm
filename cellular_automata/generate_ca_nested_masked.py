
import random
import json
from tqdm import tqdm

# ==============================================================================
# --- 1. Core Parameters Configuration ---
# ==============================================================================

class Config:
    NUM_SAMPLES = 500_000 # Large sample size for difficult task

    # --- Phase 1 CA Configuration ---
    CA1_WIDTH = 30
    CA1_LAYERS = 2

    # --- Masking Configuration ---
    MASK_VISIBLE_BITS = 15

    # --- Phase 2 CA Configuration ---
    CA2_WIDTH = MASK_VISIBLE_BITS
    CA2_LAYERS = 2

    # --- Auto-calculated Parameters ---
    INPUT_DIM_S0 = CA1_WIDTH
    INPUT_DIM_MASK = CA1_WIDTH
    INPUT_DIM_TOTAL = INPUT_DIM_S0 + INPUT_DIM_MASK
    OUTPUT_DIM = CA2_WIDTH

    # --- File Names ---
    OUTPUT_FILE = f"data_ca_nested_masked.jsonl"

# --- Rule 110 (Unchanged) ---
rule_110_map = {
    (1,1,1): 0, (1,1,0): 1, (1,0,1): 1, (1,0,0): 0,
    (0,1,1): 1, (0,1,0): 1, (0,0,1): 1, (0,0,0): 0
}

def evolve(state_list, layers):
    n = len(state_list)
    current_state = list(state_list)
    for _ in range(layers):
        next_state = [0] * n
        for i in range(n):
            left = current_state[(i - 1 + n) % n]
            center = current_state[i]
            right = current_state[(i + 1) % n]
            next_state[i] = rule_110_map.get((left, center, right), 0)
        current_state = next_state
    return current_state

# ==============================================================================
# --- 2. Core Logic and Sample Generation ---
# ==============================================================================

def generate_sample(cfg):
    """
    Generates a nested, partially observable CA computation sample.
    Flow: Evolve(Mask(Evolve(S0, L=CA1_LAYERS), M), L=CA2_LAYERS)
    """

    # --- Phase 1: Implicit Depth Computation ---
    # 1. Generate initial state for CA1 (S_0)
    initial_state_s0 = [random.randint(0, 1) for _ in range(cfg.CA1_WIDTH)]

    # 2. Compute full state of S_0 after evolving CA1_LAYERS steps (S_L1)
    final_state_s3 = evolve(initial_state_s0, cfg.CA1_LAYERS)

    # --- Phase 2: Dynamic Query ---
    # 3. Generate random Mask (M_1)
    mask_list = [1] * cfg.MASK_VISIBLE_BITS + [0] * (cfg.CA1_WIDTH - cfg.MASK_VISIBLE_BITS)
    random.shuffle(mask_list)

    # 4. Apply Mask, obtain intermediate result (Partial_S)
    partial_s3 = []
    for i in range(cfg.CA1_WIDTH):
        if mask_list[i] == 1:
            partial_s3.append(final_state_s3[i])

    # --- Phase 3: Computation on Query Result ---
    # 5. Use Partial_S as initial state for new CA, evolve CA2_LAYERS steps
    final_output = evolve(partial_s3, cfg.CA2_LAYERS)

    # --- Prepare Final Input/Output ---
    # 6. Combine into final input string
    input_s0_str = "".join(map(str, initial_state_s0))
    input_mask_str = "".join(map(str, mask_list))
    input_str = input_s0_str + input_mask_str

    # 7. Final output is final_output
    output_list = final_output

    assert len(input_str) == cfg.INPUT_DIM_TOTAL
    assert len(output_list) == cfg.OUTPUT_DIM

    return {
        "input": input_str,
        "output": output_list
    }

def main():
    cfg = Config()

    print("=" * 70)
    print(f"Nested Masked CA Dataset Generator")
    print("=" * 70)
    print(f"Flow: Evolve(Mask(Evolve(S0, L={cfg.CA1_LAYERS}), M), L={cfg.CA2_LAYERS})")
    print(f"Total Input Dimension: {cfg.INPUT_DIM_TOTAL}")
    print(f"Final Output Dimension: {cfg.OUTPUT_DIM}")
    print(f"Dataset Size: {cfg.NUM_SAMPLES:,}")
    print("=" * 70)

    with open(cfg.OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(cfg.NUM_SAMPLES), desc="Generating Samples"):
            sample = generate_sample(cfg)
            f.write(json.dumps(sample) + "\n")

    print(f"\n Dataset generation complete! Saved to '{cfg.OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
