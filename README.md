<div align="center">

**[中文](./README.md)** | **[English](./README_en.md)**

</div>

---
# 超越模式识别：神经雕刻范式

本仓库是论文《超越模式识别的神经网络——一种符号主义和联结主义结合的新范式》的官方代码实现。

研究提出了一种名为 **神经雕刻(Neural Sculpting)** 的新范式，通过使用程序化生成的理想数据和并行的求解框架，将标准神经网络从概率模仿者转变为确定性的规则执行器，从而唤醒其精确推理的潜能。

**[📜 阅读论文 - paper_zh.pdf]** | **[🚀 arXiv - 待补充]**

## 环境设置

使用 `pip install -r requirements.txt` 安装主要依赖。

*   **核心依赖:** `torch>=2.4.0`
*   **中国象棋实验依赖:** 涉及中国象棋的脚本需要额外安装 [Pikafish 引擎](https://www.pikafish.com/)，并安装 [python-chinese-chess](https://github.com/windshadow233/python-chinese-chess) 库：

## 使用说明

仓库包含两类脚本：`generate_*.py` (数据集生成) 和 `train_*.py` (模型训练)。

**典型流程:**
1.  运行 `python generate_cellular_automata_1d.py` 生成数据。
2.  运行 `python train_tiny_transformer.py` 进行训练。

所有实验脚本均存放于 `to_be_organized/` 文件夹。该目录下的脚本未来会逐步整理并移出，目前不保证所有脚本100%同步于最终实验版本，但绝大部分可直接运行。

## 模型依赖

训练脚本使用到了Hugging Face Hub里的如下模型。

*   **Qwen2-0.5B:** `Qwen/Qwen2-0.5B`
*   **Swin Transformer:** `microsoft/swin-base-patch4-window7-224-in22k`

### 脚本说明文档

本项目包含大量的用以生成数据集和训练模型的脚本。为了方便查阅，我们提供了结构化的详细文档，并同时支持中英双语。

### 关于文档的说明

本仓库中每个脚本的功能说明，初步由大型语言模型（如GPT-4, Gemini）辅助生成。虽然这些说明提供了脚本用途的概览，但可能并非完全精确，或与最新的实验细节保持同步。我会在后续逐步进行人工审阅和完善。在此期间，请以源代码为最准确的参考依据。

- **QUICK_INDEX_zh.md**: 所有脚本的**简明介绍**，用于快速查找。
- **DOCS_GENERATE_zh.md**: 所有 generate_ 脚本的**详细说明**。
- **DOCS_TRAIN_zh.md**: 所有 train_ 脚本的**详细说明**。