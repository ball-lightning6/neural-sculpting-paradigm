# Neural Processor Scripts

This directory contains scripts for generating datasets and running models for Neural Processors.

## Project Status & Limitations

**This is a very preliminary attempt with limited time investment.**

The core motivation of this project stems from an observation: in this paradigm, training a direct "input-to-output" transformation becomes extremely difficult if there are too many intermediate steps. This relates to the Maze task, where predicting the "shortest path from start to end" directly is hard, but predicting the "optimal next step" is much easier. This inspired us: **Solving a difficult, potentially variable-length optimal solution problem can be transformed into a fixed-length "optimal single step" problem.**

Based on this, we had an idea: If we can simulate a mature CPU core, theoretically we can solve any computable problem through multiple single-step simulations. This is the origin of the "Neural Processor" idea.

However, this idea faces a core challenge: **What is the precision of the Neural CPU?**
- In actual training, using an RTX 4090 GPU, we could increase single-bit prediction accuracy to about **99.9999999% (9 nines)** and overall state accuracy to about **99.99999% (7 nines)** in about 1 day.
- Such precision is sufficient to support some simple, short-process computational problems.
- To further improve precision, we experimented with **Voting Methods**, but found the effect insignificant. The reason might be that different model instances make mistakes in similar patterns (non-independent errors), which may relate to data distribution characteristics.

**Future Vision:**
The ideal architecture should be **modular**: Not only the CPU core is simulated by a neural network, but the **Memory Access Module** could obviously also be simulated. The execution process would be: Neural CPU executes instruction -> Read/Write Memory -> Loop until CPU outputs Halt instruction.

Although we haven't invested more time deeply, and the specific application scenarios are unclear, as a **Proof of Concept**, it successfully proves that such a "Neural Instruction Set Computer" can run simple programs.

---

## Design Philosophy & Architecture

This project aims to explore the possibility of using neural networks as **Arithmetic Logic Units (ALU)** or even complete **CPUs** to execute classic computer programs.

1.  **Hybrid Architecture**:
    -   Early attempts to have the neural network manage all state and memory led to exponential complexity.
    -   Therefore, a **Hybrid Design Pattern** was established: a **Python Script** acts as the classic "Controller", responsible for instruction fetching, PC jumps, memory management, and I/O; while the **Neural Network** acts as a called "Black Box ALU", focusing on core logic operations and state transitions. This preserves precise control flow while leveraging neural learning.
    -   Future ideal form: Neural CPU computes -> Memory Module read/write -> Loop until Halt.

2.  **About "Neural Voter"**:
    -   Attempts were made to use Ensemble Learning (Voting) to improve Neural CPU precision.
    -   Experiments showed errors are non-i.i.d., limiting the gain from simple majority voting.
    -   Thus, reliability depends on improving training data and architecture, not post-processing voting. This explains the shift to refined single-model training.

3.  **Evolution Roadmap**:
    -   From basic Adder (v1) -> Microprocessor with memory/jumps (v2) -> Specialized Core for Algorithms (v2.1 GCD) -> Universal Turing-complete Core (v2.2).

---

## Training Data Generation

### 1. **generate_cpu_v1_basic.py**

- **Purpose:** A basic **Neural CPU Prototype** to verify if NNs can learn basic instruction sets.
- **Logic:** Simulates a minimal 4-register CPU. Randomly generates instructions (16-bit) and initial states (register values), calculates the next state. Supports basic arithmetic and data movement.
- **I/O Format:**
    - Input: Instruction (16b) + Current State (32b).
    - Output: Next State (32b).
- **Main Parameters:** `DATASET_SIZE`, `NUM_REGISTERS`.

### 2. **generate_cpu_v2_microprocessor.py**

- **Purpose:** A more complete **Neural Microprocessor**, introducing **Memory Access** and **Control Flow**, simulating von Neumann architecture features. Used to train a model capable of computation, data access, and program jumps.
- **Logic:** Extends state space to include 16-byte memory and PC pointer. Adds LOAD/STORE and conditional jump instructions. Simulator maintains full system state updates.
- **I/O Format:**
    - Input: Instruction (24b) + Full Machine State (Regs+Flags+Mem).
    - Output: Updated Full Machine State.
- **Main Parameters:** `DATASET_SIZE`, `MEM_SIZE`.

### 3. **generate_cpu_v2_1_gcd.py**

- **Purpose:** A **Specialized Neural ALU**, with an ISA tailored/optimized for **Greatest Common Divisor (GCD)** execution. Verifies that NNs learn specific algorithm primitives highly efficiently with minimal ISAs.
- **Logic:**
    - **ISA:** 4 registers (8-bit) + 2 flags (ZF, GF).
    - **Instructions:** MOV, MOVI, SUB, CMP.
    - **Simulation:** Randomly generates instructions and states, uses Python Perfect ALU to compute next state.
- **I/O Format:**
    - Input: 16-bit Instruction + 34-bit State Vector.
    - Output: 34-bit State Vector.
- **Main Parameters:** `DATASET_SIZE` (20,000,000).

### 4. **generate_cpu_v2_2_universal.py**

- **Purpose:** A **Universal Neural ALU** with almost all basic instructions for Turing completeness. The most powerful core in the series, designed to support arbitrary complex algorithms (e.g., Sorting).
- **Logic:**
    - **ISA:** Extended set including Arithmetic (ADD, SUB, INC, DEC), Logic (AND, OR, XOR, NOT), Shift (SHL), Control (CMP, HALT).
    - **Simulation:** Precisely simulates ALU behavior including flags for each instruction with random operands.
- **I/O Format:**
    - Input: 16-bit Instruction + 34-bit State Vector.
    - Output: 34-bit State Vector.
- **Main Parameters:** `DATASET_SIZE` (20,000,000).

### 5. **generate_cpu_v3_pi.py**

- **Purpose:** A **High-Precision Specialized ALU** introducing Carry/Borrow flags (`CF`), supporting multi-precision arithmetic, aiming to calculate Pi.
- **Logic:**
    - **ISA:** 16-bit register architecture. Core instructions include ADC (Add Moving Carry), SBC (Subtract Borrowing).
    - **Simulation:** Maintains ZF, GF, CF. Arithmetic updates Carry Flag correctly for software-based arbitrary precision math.
- **I/O Format:**
    - Input: 16-bit Instruction + 35-bit State Vector.
    - Output: 35-bit State Vector.
- **Main Parameters:** `DATASET_SIZE`, `BITS_PER_REGISTER` (16).

### 6. **generate_voter_dataset.py**

- **Purpose:** Generates datasets for "Neural Voter". Uses Ensemble Learning to correct single-point errors by voting among multiple Neural Processors.
- **Logic:** Simulates outputs from multiple predictors with error rates, uses majority voting as ground truth.
- **I/O Format:**
    - Input: Concatenated outputs from N models.
    - Output: Correct output.
- **Main Parameters:** `NUM_VOTERS`, `ERROR_RATE`.

## Verification & Execution

### 7. **run_gcd_program.py**

- **Purpose:** **End-to-End Verification Script**. Loads the trained `cpu_v2_1` GCD-specialized model and verifies its ability to execute the Euclidean algorithm.
- **Logic:**
    - **Program:** Defines GCD assembly (CMP, SUB, JZ, JG).
    - **Simulation:** Loads NN as ALU. Initializes registers.
    - **Execution:** Loops instructions. Python controller handles jumps; NN predicts next state for calc instructions.
    - **Verification:** Compares with Python built-in GCD result and checks per-step accuracy.
- **I/O Format:**
    - Input: None (Random internal test cases).
    - Output: Console logs of program/step accuracy.
- **Main Parameters:** `num_tests`.

### 8. **run_rainwater_program.py**

- **Purpose:** **End-to-End Verification Script**. Drives `cpu_v2_microprocessor` to execute the LeetCode Hard **Trapping Rain Water** (Two Pointers) assembly algorithm.
- **Logic:**
    - **Program:** Implements complex two-pointer logic with memory access (LOAD/STORE) and branches.
    - **Simulation:** NN handles arithmetic/compare; Python handles memory/PC.
    - **Verification:** Compares neural execution result with standard algorithm.
- **I/O Format:**
    - Input: 10 random heights.
    - Output: 10 water values.
- **Main Parameters:** `num_tests`.

### 9. **run_bubble_sort_program.py**

- **Purpose:** **End-to-End Verification Script**. Drives `cpu_v2_2` (Universal) model to execute **Bubble Sort**.
- **Logic:**
    - **Program:** Hardcoded assembly for sorting 3 integers (nested CMP/SWAP).
    - **Simulation:** NN acts as ALU for CMP/MOV.
    - **Verification:** Compares sorted registers with Python `sorted()`.
- **I/O Format:**
    - Input: 3 random 8-bit integers.
    - Output: Sorted integers.
- **Main Parameters:** `num_tests`.

### 10. **debug_gcd_program.py**

- **Purpose:** **Step-Debugging Version** of `run_gcd_program.py`. Used for deep analysis of specific instruction/state errors.
- **Logic:** Same execution logic, but adds detailed per-step logging (PC, Instruction, Pre-State, Prediction vs Truth).
- **I/O Format:** Detailed execution logs.
- **Main Parameters:** Same as `run_gcd_program.py`.
