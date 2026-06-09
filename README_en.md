<div align="center">

# 🧠 Neural Sculpting Paradigm

**Beyond Pattern Recognition: Awakening the Precise Reasoning Potential of Neural Networks**

[English](./README_en.md) | [中文](./README.md)

---

**[🎮 Interactive Demo](https://www.modelscope.cn/studios/raven316/neural-network-rule-learning-lab)** | **[📜 Chinese Paper](./paper_zh.pdf)** | **[📜 English Paper](./paper.pdf)** | **[🚀 Zenodo - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20446430.svg)](https://zenodo.org/records/20446430)**

</div>

## 🌟 Introduction

This repository is the official code implementation of the paper *Neural Networks Beyond Pattern Recognition: A New Paradigm Uniting Symbolism and Connectionism*.

We introduce a new paradigm called **Neural Sculpting**.

**What are we doing?**
Simply put, we attempt to **use general-purpose neural networks, such as Transformers, to directly fit precise symbolic rules and algorithmic logic**.
This challenges the common impression that neural networks can only perform statistical fitting or pattern recognition, and demonstrates a possible natural integration of connectionism and symbolism.

**Three Core Findings:**
1.  **Precise Rule Learning**: Under controlled conditions, standard neural networks can learn and execute rules, algorithms, and programmatic transformations with near-perfect precision.
2.  **The Possible Unity of Pattern Recognition and Rule Learning**: Cellular automata interpolation and MNIST + cellular automata experiments strongly suggest that pattern recognition and rule learning may both be endogenous capabilities of neural networks, and may even be the same capability.
3.  **A Compression-Is-Intelligence Explanation**: These experiments can be interpreted within the framework of "compression is intelligence" and Minimum Description Length (MDL). The training process may tend toward generative rules with lower Kolmogorov complexity.

**Key Features:**
1.  **Highly General**: The paradigm does not depend on a specific architecture. Standard Transformers, MLPs, and CNNs can all be used. It is also not limited to a particular domain, ranging from mathematical rules to physical simulations.
2.  **Extremely Low Input-Space Coverage**: Although large amounts of procedurally generated data are required, the training set typically covers only a tiny fraction of the enormous input space, often merely one-billionth of it.
3.  **Extremely Precise**: Rather than merely "guessing correctly with high probability," models achieve near **100% exact-match rates** on validation sets, realizing zero-error generalization.

**Core Idea:**
The Neural Sculpting Paradigm does not rely on complicated engineering tricks. With procedurally generated ideal data and suitable input-output formats, standard neural networks can gradually approximate precise rules through gradient descent. More importantly, these experiments provide a controlled experimental field for studying the nature of neural networks, and suggest a possible unifying relationship among rule learning, pattern recognition, and compression.

## ✨ Core Capabilities (What Can It Do?)

This repository contains numerous experimental scripts that demonstrate the ability of standard neural networks to learn rules, algorithms, and programmatic transformations from different perspectives.

*   **🧮 Symbolic Rule Learning**:
    *   Perfectly mastering multi-step evolution of one-dimensional cellular automata (Rule 110).
    *   Understanding abstract algebraic structures, such as base-N addition and Boolean logic, rather than merely memorizing symbols.
*   **🧠 Fusion of Pattern Recognition and Logical Reasoning**:
    *   Cellular automata interpolation and MNIST + cellular automata experiments show that pattern recognition and rule execution can be naturally coupled within the same model, the same task, and the same training process. This strongly suggests that they may both be endogenous capabilities of neural networks, and may even be different manifestations of the same underlying capability.
*   **🧩 Algorithm Fitting and Planning**:
    *   Solving LeetCode Hard-level algorithmic problems, such as Trapping Rain Water and Largest Rectangle.
    *   Performing search-free path planning in dense mazes.
*   **📐 Visual Reasoning and Geometric Construction**:
    *   Deriving precise geometric constructions from pixels, such as triangle incircles and **planar tessellations**.
    *   **ARC-AGI Exploration**: Successfully solving 16 randomly selected ARC-AGI-1/2 tasks by training on procedurally generated isomorphic samples. These are exploratory experiments rather than direct solutions under the official few-shot setting. Related scripts are located in the `arc_agi/` directory.
*   **🍎 Physical-Law Simulation**:
    *   Learning physical laws to precisely predict planetary orbits, catenary curves, and light-refraction paths.

## 🚀 Quick Start

### 1. Installation

Install the primary dependencies with `pip install -r requirements.txt`.

*   **Core Dependency:** `torch>=2.4.0`
*   **Chinese Chess Dependencies:** Scripts involving Chinese chess require the additional installation of the [Pikafish engine](https://www.pikafish.com/) and the [python-chinese-chess](https://github.com/windshadow233/python-chinese-chess) library.

### 2. Typical Workflow

The repository contains two types of scripts: `generate_*.py` for dataset generation and `train_*.py` for model training.

**Example: Training a model to learn cellular automata rules**

```bash
# 1. Generate training data
python cellular_automata/generate_cellular_automata_1d.py

# 2. Start training
python train_tiny_transformer.py
```

## 📂 Repository Structure

The repository contains the core experiments from the paper, as well as subsequent explorations of the mechanisms underlying rule learning in neural networks.

Root-level `generate_*.py` and `train_*.py` files are the main experimental entry points directly referenced by the paper. They are kept at the repository root to make reproduction easier for readers following the paper. Some dataset-generation scripts also have copies in their corresponding categorized directories.

### Core Experiments

*   `algorithms/`: Algorithm-learning tasks, including sorting, shortest paths, Tower of Hanoi, and Trapping Rain Water
*   `cellular_automata/`: Cellular automata tasks, including rule execution, interpolation, and inverse inference
*   `symbolic_math_logic/`: Symbolic mathematics and logic tasks, including addition, logical deduction, and SAT solving
*   `visual_reasoning/`: Visual reasoning and geometric construction, including geometric drawing, image transformations, and counting
*   `physics_simulation/`: Programmatic physical simulations, including orbital motion, catenary curves, and refraction
*   `arc_agi/`: Exploratory experiments on ARC-AGI-style tasks

### Subsequent Research

*   `research/`: Further experiments on neural-network learning mechanisms
    *   `research/ntk_batch_solver/`: NTK control experiments for studying the distinction between feature learning and lazy learning
    *   `research/rule_preference/`: Rule-preference phase-transition experiments exploring neural networks' preference for lower-complexity explanations
    *   `research/rule_ood_generalization/`: Rule OOD generalization experiments studying whether models can execute unseen rules
*   `neural_processor/`: Proof-of-concept experiments using neural networks as ALU or CPU components to execute simple programs
*   `neural_inverse_engineering/`: Explorations of inferring rules from input-output observations
*   `chinese_chess/`: Early explorations of Chinese-chess policy learning

### Tools and Documentation

*   `training_scripts/`: Supplementary training scripts for multi-task learning, curriculum learning, and linear probes
*   `docs/`: Interactive documentation pages and script index
*   `utils/`: General-purpose utility scripts

## 📚 Documentation

This project contains a large number of experimental scripts. For convenient reference, we provide complete interactive documentation:

*   **[🌐 Interactive Docs](https://ball-lightning6.github.io/neural-sculpting-paradigm/)**: **Highly Recommended**. Covers all public scripts, with quick indexing, bilingual support, and expandable details.
*   **[🎮 Interactive Demo (ModelScope)](https://www.modelscope.cn/studios/raven316/neural-network-rule-learning-lab)**: Generate random samples or enter inputs manually, then directly compare one-forward-pass neural network outputs with program-generated ground truth.

## 🛠️ Roadmap

Most experiments in this repository have been verified to varying degrees. Since the repository preserves some early scripts from the exploratory process, a small number of experiments still require further reproduction, documentation, and status labeling.

*   **Systematic Reproduction**: Gradually rerun the experiments and record key information, including training curves, convergence behavior, required dataset sizes, model configurations, training time, and random seeds.
*   **Experiment Status Labels**: Add status notes to a small number of early attempts or experiments that have not yet been fully verified, avoiding potential confusion.
*   **Documentation Improvement**: Continue refining the interactive documentation with clearer descriptions, usage instructions, parameter configurations, and experimental results.

---

<div align="center">

**If this looks like magic, it is! Just try it!**

If this project inspires you, please give it a ⭐ Star!

</div>
