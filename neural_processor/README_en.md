# Neural Procesor Scripts

This directory contains scripts for generating and running neural processor datasets and models.

## Project Status & Limitations

**This is a very preliminary attempt with limited time investment.**

The core motivation for this project stems from the observation that training direct "input-to-output" transformations in this paradigm becomes extremely difficult when the number of intermediate steps is large. This is reminiscent of earlier attempts to train maze tasks, where directly predicting the "shortest path from start to end" was very hard, but predicting the "optimal next step for the shortest path" was much more effective. This led to the idea: **Solving a difficult, potentially variable-length optimal solution problem can be transformed into a fixed-length "optimal single step" problem.**

Based on this, we conceived the idea: If we can simulate a mature CPU core, then theoretically we can solve any computable problem through multiple single-step simulations. This is the origin of the "Neural Processor" idea.

However, this idea faces a core challenge: **What is the precision of the Neural CPU?**
- In actual training, using an RTX 4090 GPU, we can improve single-bit prediction accuracy to about **99.9999999% (9 nines)** and overall output state accuracy to about **99.99999% (7 nines)** in about 1 day.
- Such precision is sufficient to support some simple, short-process problem calculations.
- To further improve precision, we tried **voting methods**, but found that the effect was not significant. The analysis suggests that the patterns of errors made by different model instances are similar (non-independent errors), which may be related to the distribution characteristics of the training data.

**Vision for the Future:**
The ideal architecture should be **modular**: not only is the CPU core simulated by a neural network, but the **memory access module** can obviously also be simulated by a neural network. The program execution process would be: Neural CPU executes instruction -> Read/Write Memory -> Loop repeatedly, until the CPU outputs a Halt instruction to interrupt execution.

Although we have not invested further time at present, and the specific subsequent application scenarios of this project are not yet clear, as a **Proof of Concept**, it successfully proves that this "Neural Instruction Set Computer" can run simple programs.

---

## Design Philosophy and Architecture

This project aims to explore the possibility of using neural networks as **Arithmetic Logic Units (ALUs)** or even complete **CPUs** to execute classical computer programs.

1.  **Hybrid Architecture**:
    -   After early attempts to manage all states and memory entirely with neural networks, it was found that training difficulty increased exponentially with complexity.
    -   Therefore, this project established a **hybrid design pattern**: **Python scripts** act as the classic "Controller", responsible for instruction fetching, Program Counter (PC) jumps, memory read/write management, and I/O; while the **Neural Network** acts as a called "Black Box ALU", focusing on executing core logic operations and state transitions. This design retains the precise control flow of traditional architectures while leveraging the learning capabilities of neural networks.
    -   The ideal future form might be: Neural CPU for computation -> Memory module for read/write -> Loop until a halt signal.

2.  **Regarding "Neural Voter"**:
    -   Attempts were made to use Ensemble Learning (Voting) to improve the precision of the Neural CPU.
    -   However, experiments revealed that the patterns of errors made by different model instances were often not independent (Non-i.i.d. errors), leading to limited improvement from simple majority voting.
    -   Therefore, the key to improving reliability lies in improving training data and the architecture itself, rather than relying solely on post-processing voting. This explains why the project ultimately shifted towards more refined single-model training.

3.  **Evolutionary Path**:
    -   From the most basic Adder (v1), evolving to a Microprocessor with memory and jumps (v2), then to a specialized core for specific algorithms (GCD) (v2.1), and finally to a Turing-complete universal core (v2.2).

---

## Training Data Generation

### 1. `generate_cpu_v1_basic.py`
- **Purpose:** Prototype for a basic **Neural CPU**, designed to verify if a neural network can learn to execute a minimal instruction set.
- **ISA Definition:**
    - **State:** 4 8-bit registers.
    - **Instructions:** `NOP`, `MOVI` (Move Immediate), `ADD`, `XOR`.
    - **Architecture:** 16-bit instruction length.
- **I/O Format:**
    - Input: Instruction (16b) + Current State (32b).
    - Output: Next State (32b).
- **Key Parameters:** `DATASET_SIZE`, `NUM_REGISTERS`.
- **Status:** Verified (Basic functionality validated).

### 2. `generate_cpu_v2_microprocessor.py`
- **Purpose:** A more fully-featured **Neural Microprocessor**, introducing **Memory Access** and **Control Flow**, simulating core features of the Von Neumann architecture. Used to train a model that handles not just computation but also data storage and branching.
- **ISA Definition:**
    - **State:** 8 8-bit registers + 1 flag bit + 16 bytes of memory.
    - **Instructions:** `LOAD`, `STORE`, `ADD`, `SUB`, `CMP`, `JMP`, `JLT` (Jump if Less Than), `PRINT`, etc.
    - **Architecture:** 24-bit instruction length.
- **I/O Format:**
    - Input: Instruction (24b) + Full Machine State (Registers + Flags + Memory).
    - Output: Updated Full Machine State.
- **Key Parameters:** `DATASET_SIZE`, `MEM_SIZE`.
- **Status:** **Verified**. Used in `run_rainwater_program.py` to successfully execute the Trapping Rain Water algorithm.

### 3. `generate_cpu_v2_1_gcd.py`
- **Purpose:** A **Specialized Neural ALU** with an instruction set tailored and optimized specifically to support the **Greatest Common Divisor (GCD)** algorithm (Euclidean algorithm).
- **ISA Definition:**
    - **State:** 4 8-bit registers + 2 flags (ZF, GF).
    - **Instructions:** `MOVI`, `MOV`, `SUB`, `CMP`.
- **Significance:** Demonstrates that by pruning the instruction set, a neural network can learn specific algorithmic primitives extremely efficiently. 
- **Status:** **Verified**. This model was verified to execute the Euclidean algorithm with 100% accuracy in `run_gcd_program.py`.

### 4. `generate_cpu_v2_2_universal.py`
- **Purpose:** A **Universal Neural ALU** containing almost all basic instructions required for Turing completeness. It is the most powerful general-purpose computation core in this series.
- **ISA Definition:**
    - **Instructions (Extended):** Arithmetic (`ADD`, `SUB`, `INC`, `DEC`), Logic (`AND`, `OR`, `XOR`, `NOT`), Shift (`SHL`), and Control (`CMP`).
- **Significance:** In `run_bubble_sort_program.py`, this model demonstrated strong generalization capabilities, successfully executing the complex Bubble Sort algorithm.
- **Status:** **Verified**. Achieved 100% accuracy in the Bubble Sort end-to-end test.

### 5. `generate_cpu_v3_pi.py`
- **Purpose:** A **High-Precision Computation Specialized ALU**, introducing Carry and Borrow flags to support multi-precision arithmetic, aiming to enable neural networks to compute high-precision values like Pi.
- **ISA Definition:**
    - **State:** 16-bit registers + 3 flags (ZF, GF, CF).
    - **Instructions:** `ADC` (Add with Carry), `SBC` (Subtract with Borrow), etc.
- **Status:** **Experimental**. Dataset generation script is available, but no end-to-end program execution has been verified with this specific architecture yet.

### 6. `generate_voter_dataset.py`
- **Purpose:** Generates datasets for training a "Neural Voter". This module aims to use Ensemble Learning to correct single-point errors by voting on the predictions of multiple neural processors.
- **Logic:** Simulates outputs from multiple predictors with certain error rates and uses majority voting as the ground truth label.
- **Note:** Subsequent research found that errors in neural processors often correlated (non-i.i.d.), so simple voting provided limited improvement. This module is kept primarily as an exploratory record.
- **Status:** **Experimental**. Concept implementation only; not integrated into the main execution pipeline.

## Verification & Execution

### 7. `run_gcd_program.py`
- **Purpose:** **End-to-End Verification Script**. Loads a trained `cpu_v2_1` binary model and drives it to execute a complete assembly program implementing the Euclidean algorithm (GCD).
- **Core Logic:** Implements a hybrid architecture simulator. **Control Flow** (Jump/Branch) is handled by Python code (the traditional Von Neumann controller), while **Arithmetic Logic & State Updates** are performed entirely by neural network (Neural ALU) inference.
- **Result:** Achieved **100%** program execution accuracy in tests.

### 8. `run_rainwater_program.py`
- **Purpose:** **End-to-End Verification Script**. Uses the neural processor (`cpu_v2_microprocessor`) to execute the classic LeetCode Hard problem — **Trapping Rain Water** using the Two Pointers algorithm.
- **Complexity:** Compared to GCD, this program involves more complex memory read/writes (`LOAD`/`STORE`) and pointer operations, verifying the neural microprocessor's ability to handle complex data structures.
- **Result:** Validated successfully.

### 9. `run_bubble_sort_program.py`
- **Purpose:** **End-to-End Verification Script**. Drives the `cpu_v2_2` (Universal) model to execute the **Bubble Sort** algorithm.
- **Result:** Logs show extremely high accuracy in sorting 3 random numbers, proving the model's perfect mastery of compare-and-swap logic.

### 10. `debug_gcd_program.py`
- **Purpose:** A **Single-Step Debugging Version** of `run_gcd_program.py`. It runs the program and prints detailed logs for every step, including PC pointer, instruction, pre-execution state, neural network prediction, and ground truth comparison. Used for deep analysis of exactly where or in what state the model might fail.
