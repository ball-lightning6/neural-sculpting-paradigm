# Beyond Pattern Recognition: The Neural Sculpting Paradigm

This is the official implementation for the paper "Beyond Pattern Recognition: A New Paradigm Uniting Symbolicism and Connectionism".

The research introduces **"Neural Sculpting"** a new paradigm that transforms standard neural networks from probabilistic imitators into deterministic rule executors. By using procedurally generated ideal data and a parallel solving framework, we awaken the innate potential for precise reasoning in these models.

**[📜 Read the Paper - paper.pdf]** | **[🚀 arXiv - TBD]**

## Setup

Install the primary dependencies using `pip install -r requirements.txt`.

*   **Core Dependency:** `torch>=2.4.0`
*   **Chinese Chess Dependencies:** Scripts related to Chinese Chess additionally require the [Pikafish engine](https://www.pikafish.com/) and the [python-chinese-chess](https://github.com/windshadow233/python-chinese-chess) library.

## Usage

The repository contains two types of scripts: `generate_*.py` (dataset generation) and `train_*.py` (model training).

**Typical Workflow:**
1.  Run `python generate_cellular_automata_1d.py` to generate data.
2.  Run `python train_tiny_transformer.py` to start training.

All experimental scripts are located in the `to_be_organized/` directory. These scripts will be organized and moved out over time. While most should run correctly, they are not guaranteed to be 100% in sync with the final experimental versions.

## Model Dependencies

The training script uses the following model from the Hugging Face Hub.

*   **Qwen2-0.5B:** `Qwen/Qwen2-0.5B`
*   **Swin Transformer:** `microsoft/swin-base-patch4-window7-224-in22k`

### A Note on Documentation

The descriptions for each script in this repository were initially generated with the assistance of Gemini 2.5 Pro. While they provide a general overview of each script's purpose, they may not be perfectly accurate or up-to-date with the latest experimental details. I will be manually reviewing and refining this documentation over time. Please refer to the source code as the definitive source of truth.

- **QUICK_INDEX_en.md**: A **quick index** with brief descriptions of all scripts for easy lookup.
- **DOCS_GENERATE_en.md**: **Detailed documentation** for all generate_ scripts.
- **DOCS_TRAIN_en.md**: **Detailed documentation** for all train_ scripts.