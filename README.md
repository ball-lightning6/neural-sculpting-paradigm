<div align="center">

# 🧠 神经雕刻范式 (Neural Sculpting Paradigm)

**超越模式识别：唤醒神经网络的精确推理潜能**

[English](./README_en.md) | [中文](./README.md)

---

**[🎮 在线交互 Demo](https://www.modelscope.cn/studios/raven316/neural-network-rule-learning-lab)** | **[🔬 Neural K 后续研究](./research/neural_k_framework/README.md)** | **[📜 中文论文](./paper_zh.pdf)** | **[📜 English Paper](./paper.pdf)** | **🚀 Zenodo** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20446430.svg)](https://zenodo.org/records/20446430)

</div>

> **最新理论核心（建议先读）：[Neural K 理论核心](./research/neural_k_framework/theory_core_zh.md)。** 该页由作者逐段审阅并深度修改；完整研究主文、证据总账与复现材料见 [Neural K 研究子目录](./research/neural_k_framework/README.md)。

## 🌟 简介 (Introduction)

本仓库是论文《超越模式识别的神经网络——一种符号主义和联结主义结合的新范式》的官方代码实现。

我们提出了一种名为 **神经雕刻 (Neural Sculpting)** 的新范式。

**我们在做什么？**
简单来说，我们尝试**用通用的神经网络（如 Transformer）直接去拟合精确的符号规则与算法逻辑**。
这挑战了神经网络只能进行统计拟合或模式识别的常见印象，并展示了一种联结主义与符号主义自然结合的可能方式。

**三项核心发现：**
1.  **精确规则学习**：标准神经网络可以在受控条件下学习并近乎精确地执行规则、算法和程序化变换。
2.  **模式识别与规则学习的统一性**：元胞自动机插值与 MNIST+元胞自动机实验强烈暗示，模式识别与规则学习可能都是神经网络的内生能力，甚至可能是同一种能力。
3.  **压缩即智能的 Neural K 解释**：固定网络、编码、损失和参数测度以后，函数先验高度不均匀；继续降低完整目标 loss 时，不同函数体积又会差异收缩，通常使更经济、可复用的实现获得相对优势。这与 MDL 和算法信息论自然相连，但复杂度是神经协议相对的 Neural K-profile，不直接等同机器无关 Kolmogorov complexity。

**核心特性：**
1.  **高度通用**：不依赖特定架构，标准 Transformer/MLP/CNN 均可适用；不限于特定领域，从数学规则到物理模拟均可掌握。
2.  **极低空间覆盖率**：虽然需要程序化生成大量数据，但训练集通常仅覆盖巨大输入空间的极小部分（往往仅占输入空间的亿万分之一）。
3.  **极致精确**：模型并非在“大概率猜对”，而是在验证集上达到了接近 **100% 的精确匹配率**，实现了零误差泛化。

**核心理念：**
神经雕刻范式本身并不依赖复杂的工程技巧：通过程序化生成的理想数据和合适的输入输出格式，标准神经网络可以在梯度下降过程中逐渐逼近精确规则。更重要的是，这些实验为研究神经网络的本质提供了一个受控实验场，并提示规则学习、模式识别与压缩之间可能存在统一的联系。

## ✨ 核心能力展示 (What Can It Do?)

本仓库包含大量实验脚本，从不同角度展示标准神经网络学习规则、算法与程序化变换的能力。

*   **🧮 符号规则学习**:
    *   完美掌握一维元胞自动机 (Rule 110) 的多步演化。
    *   理解抽象的代数结构（如 N 进制加法、布尔逻辑），而非简单的符号记忆。
*   **🧠 模式识别与逻辑推理的融合**:
    *   元胞自动机插值与 MNIST+元胞自动机实验表明，模式识别与规则执行可以在同一个模型、同一个任务和同一次训练过程中自然耦合。这强烈暗示二者可能都是神经网络的内生能力，甚至可能是同一种底层能力在不同数据形式下的表现。
*   **🧩 算法拟合与规划**:
    *   解决 LeetCode Hard 级算法题（如接雨水问题、最大矩形）。
    *   在稠密迷宫中进行零搜索的路径规划。
*   **📐 视觉推理与几何构造**:
    *   从像素中推导精确的几何构造（如三角形内切圆、**平面镶嵌**）。
    *   **ARC-AGI 挑战探索**: 通过程序化生成同构样本进行训练，成功解决了随机抽取的 16 个 ARC-AGI-1/2 任务。这是一组探索性实验，并非官方 few-shot 设置下的直接解法。相关脚本位于 `arc_agi/` 目录。
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

仓库包含论文中的核心实验，以及围绕神经网络规则学习机制展开的后续探索。

根目录中的 `generate_*.py` 和 `train_*.py` 是论文正文直接涉及的主要实验入口。为了方便读者按照论文复现实验，相关脚本集中保留在根目录；其中一部分数据生成脚本在对应分类目录中也有副本。

### 核心实验

*   `algorithms/`: 算法学习任务（排序、最短路、汉诺塔、接雨水等）
*   `cellular_automata/`: 元胞自动机相关任务（规则执行、插值、逆向推理等）
*   `symbolic_math_logic/`: 符号数学与逻辑任务（加法、逻辑演绎、SAT 求解等）
*   `visual_reasoning/`: 视觉推理与几何构造（几何作图、图像变换、计数等）
*   `physics_simulation/`: 程序化物理模拟（轨道、悬链线、折射等）
*   `arc_agi/`: ARC-AGI 风格任务的探索性实验

### 后续研究

*   `research/`: 围绕神经网络学习机制展开的进一步实验
    *   `research/neural_k_framework/`: 最新 Neural K 理论核心、研究主文、证据总账、E01--E25 复现实验和中英文静态网站
    *   `research/ntk_batch_solver/`: NTK 对照实验，用于研究特征学习与懒惰学习的区别
    *   `research/rule_preference/`: 规则偏好相变实验，探索神经网络对低复杂度解释的偏好
    *   `research/rule_ood_generalization/`: 规则 OOD 泛化实验，研究模型能否执行未见过的新规则
    *   `research/overfitting_related_research/`: 过拟合相关研究，通过 probe 一致性实验观察不同随机 seed 在训练集外形成的函数分布，探索“记忆”、泛化、随机标签与规则学习之间的关系
*   `neural_processor/`: 将神经网络作为 ALU 或 CPU 部件，执行简单程序的概念验证
*   `neural_inverse_engineering/`: 从输入输出观测反向推断规则的探索
*   `chinese_chess/`: 中国象棋策略学习的早期探索

### 工具与文档

*   `training_scripts/`: 多任务学习、课程学习、线性探针等补充训练脚本
*   `docs/`: 在线文档页面及脚本索引
*   `utils/`: 通用辅助工具

## 📚 文档索引 (Documentation)

本项目包含大量实验脚本。为了方便查阅，我们提供了完整的在线文档：

*   **[🧭 Neural K 理论与证据包](./research/neural_k_framework/README.md)**: 最新理论核心、完整判决链、E01--E25 实验说明和复现脚本。
*   **[🌐 在线文档 (Interactive Docs)](https://ball-lightning6.github.io/neural-sculpting-paradigm/)**: **强烈推荐**。覆盖全部公开脚本，支持快速索引、中英切换与详情展开。
*   **[🎮 在线交互 Demo (ModelScope)](https://www.modelscope.cn/studios/raven316/neural-network-rule-learning-lab)**: 可以随机生成样本或手动输入，直接对比训练后神经网络的一次前向传播输出与程序生成的 ground truth。
*   **[🔬 过拟合相关研究](./research/overfitting_related_research/index.html)**: 一个后续实验页面，展示不同数据量、不同规则复杂度与随机标签条件下，训练后模型在 probe 集上的跨 seed 一致性、熵、共同错误结构等指标。

## 🛠️ 待办事项与维护计划 (Roadmap)

本仓库中的大部分实验已经得到不同程度的验证。由于仓库保留了研究探索过程中的部分早期脚本，少量实验仍需进一步复现、整理和标注。

*   **系统复现实验**：逐步重新运行仓库中的实验，记录训练曲线、收敛程度、必要数据量、模型配置、训练耗时和随机种子等关键信息。
*   **标注实验状态**：为少量早期尝试或未充分验证的脚本补充状态说明，避免读者误解。
*   **完善实验文档**：持续整理在线文档，为各个脚本补充用途说明、运行方法、参数配置和实验结果。

---

<div align="center">

**如果这看起来像是魔法，它就是！Just try it!**

如果这个项目对您有启发，请给它一个 ⭐ Star！

</div>
