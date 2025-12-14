# Neural Inverse Engineering (Neural Reverse Engineering)

## Project Overview

This project aims to explore a profound problem in scientific computing: **Can a neural network reverse engineer the underlying generation rules (source code) by observing input-output transformations?** This is often referred to as "Neural Inverse Engineering" or "Rule Induction".

The biggest challenge in this task is **Ambiguity**. For a finite set of observations, there may be multiple different combinations of rules that produce the exact same result. For example, with a limited bit width, "Left Shift 31 bits" and "Right Shift 1 bit" might be equivalent.

To tackle this problem, we explored two distinct technical paths:

1.  **Path 1: Neural Inducer**
    *   **Core Philosophy**: **Data Preprocessing**. Eliminate all ambiguity at the data generation stage using mathematical methods (fingerprint deduplication).
    *   **Features**: Constructs a perfect, one-to-one "unambiguous dataset", forcing the model to memorize the mapping between rules and behaviors.
    *   **Representative Script**: `generate_ca_meta_dataset_unique.py`

2.  **Path 2: Neural Scientist**
    *   **Core Philosophy**: **Online Active Inference**. Accepts ambiguity in data and solves it by introducing the concept of "Observation Sets" (Set-to-Sequence). The model acts like a scientist, reading large amounts of experimental data (Observational Data) and using the Transformer's attention mechanism to aggregate information, ultimately locking onto the single truth.
    *   **Features**: Dynamic data generation, naturally handles uncertainty, and mimics the learning style of AGI.
    *   **Representative Script**: `train_neural_scientist_transformer.py`

---

## Script Documentation

### 1. **train_neural_scientist_transformer.py** (Core)

- **Purpose:** **The culmination of Path 2**. Implements a complete "Neural Scientist" agent based on Transformer.
- **Logic:**
    1.  **Dynamic Environment**: Real-time random generation of rule combinations (e.g., "Rule 110 + Shift + NOT") during training.
    2.  **Observation**: Randomly generates $N$ (e.g., 24~48) input-output pairs as evidence for each rule group.
    3.  **Observer**: Uses an MLP to independently encode each observation sample.
    4.  **Aggregator**: Uses a Transformer Encoder to interact across all observation samples, extracting common "physical laws".
    5.  **Decider**: Decodes the final rule sequence.
- **I/O Format:**
    - Input: [Batch, N_Obs, 60] (30-bit Input + 30-bit Output).
    - Output: [Batch, Rule_Bits] (Predicted rule sequence encoding).
- **Key Parameters:** `MIN_OBSERVE`, `MAX_OBSERVE` (Number of observation samples), `RULE_LAYERS` (Rule depth).

### 2. **analyze_rule_equivalence.py** (General Tool)

- **Purpose:** **Theoretical tool for rule equivalence analysis**. Independent of any model, used to mathematically analyze the degree of ambiguity in rule systems.
- **Logic:**
    1.  Iterates through all possible rule combinations (e.g., $8^4 = 4096$).
    2.  For each combination, uses a fixed set of random probes to calculate its output.
    3.  Generates a "Behavior Fingerprint" for that rule.
    4.  Counts how many rules share the exact same fingerprint to calculate the theoretical upper limit caused by ambiguity.
- **Value:** It provides Ground Truth, telling us whether a model prediction error is due to the model being dumb or the rules being mathematically indistinguishable.

### 3. **analyze_ca_inverse_ambiguity.py** (CA Specific Tool)

- **Purpose:** **Ambiguity analyzer for Cellular Automata Inverse Engineering**.
- **Logic:**
    - Focuses specifically on 1D CA tasks.
    - Uses **Monte Carlo Simulation** instead of brute force enumeration.
    - Efficiently estimates the probability that a random initial state produces ambiguous results during inverse inference.
    - Highly optimized with bitwise operations (Speed: ~50k iter/s).
- **Output:** Provides an estimated probability of ambiguity (e.g., P ≈ 9.53e-7).

### 4. **generate_ca_meta_dataset_unique.py**

- **Purpose:** **Data Generator for Path 1**. Generates a rigorously deduplicated, unambiguous meta-learning dataset.
- **Logic:**
    1.  Pre-calculates fingerprints for all rule combinations (same as above).
    2.  When conflicts are found (Rule A and Rule B have the same fingerprint), **keeps only one** as the representative and forcibly deletes other equivalent rules.
    3.  The generated dataset guarantees that for any Input/Output behavior, there is only one unique correct Label.
- **I/O Format:** JSONL format, containing `input` (initial state), `output` (evolved state), `rule_and_layer_label` (rule label).

### 5. **generate_ca_meta_dataset_info_complete.py**

- **Purpose:** Early exploration of Path 1. Attempts to solve ambiguity at the **single sample** level.
- **Logic:** When generating samples, it forcibly checks if the sample contains all possible local patterns (e.g., covering all 000-111 substrings). Only "information complete" samples are retained.
- **Limitation:** For complex rules, it is very difficult for a single sample to be truly information complete.

### 6. **generate_rule_composition_dataset.py**

- **Purpose:** Static data version for Path 2. Generates training data for multi-layer rule compositions.
- **Logic:** Randomly selects $L$ layers of rules (controlled by `NUM_LAYERS`), executes them sequentially, and generates input-output pairs. unlike `Untitled55`, this generates files offline and is typically used for simple MLP training (viewing the composition of multiple rules as a black-box function).
- **Key Parameters:** `NUM_LAYERS`, `RULE_BITS`.
