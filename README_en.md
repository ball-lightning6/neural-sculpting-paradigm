<div align="center">

# 🧠 Neural Sculpting Paradigm

**Beyond Pattern Recognition: Awakening the Precise Reasoning Potential of Neural Networks**

[English](./README_en.md) | [中文](./README.md)

---

**[📜 Read Paper - paper.pdf]** | **[🚀 Zenodo - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20446430.svg)](https://zenodo.org/records/20446430)]**

</div>

## 🌟 Introduction

This repository is the official implementation of the paper "Beyond Pattern Recognition: A New Paradigm Uniting Symbolicism and Connectionism".

We introduce a new paradigm called **Neural Sculpting**.

**What are we doing?**
Simply put, we attempt to **use general-purpose neural networks (like Transformers) to directly fit precise symbolic rules and algorithmic logic**.
This directly challenges the industry consensus that "neural networks cannot truly learn symbolic reasoning" and achieves an elegant fusion of connectionism and symbolism.

**Key Features:**
1.  **Highly General**: Not dependent on specific architectures (standard Transformers/MLPs/CNNs all work); not limited to specific domains (from mathematical rules to physical simulations).
2.  **Data Efficient**: While procedurally generated data is required, the amount needed is minimal relative to the vast problem space (often only a billionth of the input space).
3.  **Extremely Precise**: The model is not just "guessing with high probability," but achieving near **100% exact match rates** on validation sets, realizing zero-error generalization.

**Core Philosophy:**
By using **procedurally generated ideal data** and a **non-autoregressive parallel solving framework**, we transform standard neural networks from probabilistic "imitators" into deterministic **precise rule executors**.

## ✨ Core Capabilities (What Can It Do?)

This repository contains numerous experimental scripts demonstrating the model's **zero-error generalization** capabilities in the following areas:

*   **🧮 Symbolic Rule Learning**:
    *   Perfect mastery of multi-step evolution of 1D Cellular Automata (Rule 110).
    *   Understanding abstract algebraic structures (e.g., N-base addition, Boolean logic) rather than simple symbol memorization.
*   **🧠 Fusion of Pattern Recognition & Logic**:
    *   The MNIST evolution experiments demonstrate that precise symbolic reasoning is **endogenous** to neural networks. It runs parallel to traditional pattern recognition and can coexist perfectly within the same network.
*   **🧩 Algorithm Fitting & Planning**:
    *   Solving LeetCode Hard-level algorithmic problems (e.g., Trapping Rain Water, Largest Rectangle).
    *   Zero-search path planning in dense mazes.
*   **📐 Visual Reasoning & Geometric Construction**:
    *   Deriving precise geometric constructions from pixels (e.g., triangle incircles, **planar tessellations**).
    *   **ARC-AGI Exploration**: Successfully solved 16 randomly selected ARC-AGI-1/2 tasks using an "indirect" strategy (procedurally generating isomorphic data). Scripts are located in `arc_agi/`.
*   **🍎 Physics Simulation**:
    *   Learning physical laws to precisely predict planetary orbits, catenary shapes, and light refraction paths.

## 🚀 Quick Start

### 1. Installation

Install primary dependencies using `pip install -r requirements.txt`.

*   **Core Dependency:** `torch>=2.4.0`
*   **Chinese Chess Dependency:** Scripts involving Chinese Chess require the additional installation of the [Pikafish engine](https://www.pikafish.com/) and the [python-chinese-chess](https://github.com/windshadow233/python-chinese-chess) library.

### 2. Typical Workflow

The repository contains two types of scripts: `generate_*.py` (dataset generation) and `train_*.py` (model training).

**Example: Training a model to learn Cellular Automata rules**

```bash
# 1. Generate training data
python cellular_automata/generate_cellular_automata_1d.py

# 2. Start training
python train_tiny_transformer.py
```

## 📂 Repository Structure

All experimental scripts are organized by function:

*   `algorithms/`: Algorithm learning tasks (Sorting, Shortest Path, Hanoi, Rain Water...)
*   `cellular_automata/`: Cellular Automata tasks (1D/2D, Inverse Inference...)
*   `symbolic_math_logic/`: Symbolic Math & Logic Reasoning (Addition, Deduction, SAT Solver...)
*   `visual_reasoning/`: Visual Reasoning & Geometric Construction (Geometry, Counting...)
*   `physics_simulation/`: Physics Simulation (Orbits, Catenary, Refraction...)
*   `arc_agi/`: ARC-AGI Challenge Exploration
*   `chinese_chess/`: Chinese Chess Experiments
*   `utils/`: Utility Scripts

## 📚 Documentation

This project contains a large number of scripts. For easy reference, we provide detailed documentation:

*   **[QUICK_INDEX_en.md](./QUICK_INDEX_en.md)**: **Highly Recommended**. A concise index of all scripts for quick lookup.
*   **[DOCS_GENERATE_en.md](./DOCS_GENERATE_en.md)**: Detailed parameter and logic descriptions for all `generate_` scripts.
*   **[DOCS_TRAIN_en.md](./DOCS_TRAIN_en.md)**: Detailed descriptions for all `train_` scripts.
*   **[🌐 Interactive Docs](https://ball-lightning6.github.io/neural-sculpting-paradigm/)**: New interactive documentation page with bilingual support and expandable details.

---

<div align="center">

**If this looks like magic, it is! Just try it!**

If this project inspires you, please give it a ⭐ Star!

</div>
