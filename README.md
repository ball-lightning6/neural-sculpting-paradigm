<div align="center">

# 🧠 神经雕刻范式 (Neural Sculpting Paradigm)

**超越模式识别：唤醒神经网络的精确推理潜能**

[English](./README_en.md) | [中文](./README.md)

---

**[📜 阅读论文 - paper_zh.pdf]** | **[🚀 Zenodo - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18728833.svg)](https://doi.org/10.5281/zenodo.18728833)]**

</div>

## 🌟 简介 (Introduction)

本仓库是论文《超越模式识别的神经网络——一种符号主义和联结主义结合的新范式》的官方代码实现。

我们提出了一种名为 **神经雕刻 (Neural Sculpting)** 的新范式。

**我们在做什么？**
简单来说，我们尝试**用通用的神经网络（如 Transformer）直接去拟合精确的符号规则与算法逻辑**。
这直接挑战了业界关于“神经网络无法真正学习符号推理”的共识，并以一种极其优雅的方式实现了联结主义与符号主义的融合。

**核心特性：**
1.  **高度通用**：不依赖特定架构，标准 Transformer/MLP/CNN 均可适用；不限于特定领域，从数学规则到物理模拟均可掌握。
2.  **数据高效**：虽然需要程序化生成数据，但相对于巨大的问题空间，所需训练数据量极少（往往仅占输入空间的亿万分之一）。
3.  **极致精确**：模型并非在“大概率猜对”，而是在验证集上达到了接近 **100% 的精确匹配率**，实现了零误差泛化。

**核心理念：**
通过使用**程序化生成的理想数据**和**非自回归的并行求解框架**，我们将标准神经网络从概率性的“模仿者”转变为确定性的**精确规则执行器**。

## ✨ 核心能力展示 (What Can It Do?)

本仓库包含大量实验脚本，证明了模型在以下领域的**零误差泛化**能力：

*   **🧮 符号规则学习**:
    *   完美掌握一维元胞自动机 (Rule 110) 的多步演化。
    *   理解抽象的代数结构（如 N 进制加法、布尔逻辑），而非简单的符号记忆。
*   **🧠 模式识别与逻辑推理的融合**:
    *   通过 MNIST 数字演化实验证明，精确的符号推理能力是**内生于神经网络**的。它与传统的模式识别能力并行不悖，可以在同一个网络中完美共存。
*   **🧩 算法拟合与规划**:
    *   解决 LeetCode Hard 级算法题（如接雨水问题、最大矩形）。
    *   在稠密迷宫中进行零搜索的路径规划。
*   **📐 视觉推理与几何构造**:
    *   从像素中推导精确的几何构造（如三角形内切圆、**平面镶嵌**）。
    *   **ARC-AGI 挑战探索**: 采用一种“迂回”策略（程序化生成同构数据），成功解决了随机抽取的 16 个 ARC-AGI-1/2 任务。相关脚本位于 `arc_agi/` 目录。
*   **🍎 物理规律模拟**:
    *   学习物理定律，精确预测行星轨道、悬链线形状、光线折射路径。

## 🚀 快速开始 (Quick Start)

### 1. 环境安装

使用 `pip install -r requirements.txt` 安装主要依赖。

*   **核心依赖:** `torch>=2.4.0`
*   **中国象棋实验依赖:** 涉及中国象棋的脚本需要额外安装 [Pikafish 引擎](https://www.pikafish.com/)，并安装 [python-chinese-chess](https://github.com/windshadow233/python-chinese-chess) 库。

### 2. 典型工作流

仓库包含两类脚本：`generate_*.py` (数据集生成) 和 `train_*.py` (模型训练)。

**示例：训练模型学习元胞自动机规则**

```bash
# 1. 生成训练数据
python cellular_automata/generate_cellular_automata_1d.py

# 2. 开始训练
python train_tiny_transformer.py
```

## 📂 仓库结构 (Repository Structure)

所有实验脚本已按功能分类整理：

*   `algorithms/`: 算法学习任务 (排序, 最短路, 汉诺塔, 接雨水...)
*   `cellular_automata/`: 元胞自动机相关任务 (1D/2D, 逆向推理...)
*   `symbolic_math_logic/`: 符号数学与逻辑推理 (加法, 逻辑演绎, SAT求解...)
*   `visual_reasoning/`: 视觉推理与几何构造 (几何作图, 计数...)
*   `physics_simulation/`: 物理规律模拟 (轨道, 悬链线, 折射...)
*   `arc_agi/`: ARC-AGI 挑战任务探索
*   `chinese_chess/`: 中国象棋相关实验
*   `utils/`: 通用工具脚本

## 📚 文档索引 (Documentation)

本项目包含大量脚本。为了方便查阅，我们提供了详细的文档：

*   **[QUICK_INDEX_zh.md](./QUICK_INDEX_zh.md)**: **强烈推荐**。所有脚本的简明索引，用于快速查找感兴趣的实验。
*   **[DOCS_GENERATE_zh.md](./DOCS_GENERATE_zh.md)**: 所有 `generate_` 脚本的详细参数和逻辑说明。
*   **[DOCS_TRAIN_zh.md](./DOCS_TRAIN_zh.md)**: 所有 `train_` 脚本的详细说明。
*   **[🌐 在线文档 (Interactive Docs)](https://ball-lightning6.github.io/neural-sculpting-paradigm/)**: 全新的交互式文档页面，支持中英切换与详情展开。

---

<div align="center">

**如果这看起来像是魔法，它就是！Just try it!**

如果这个项目对您有启发，请给它一个 ⭐ Star！

</div>