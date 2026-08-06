# Neural Processor / 神经处理器

## 中文

本目录包含两个阶段的神经处理器实验：

- [`neural_cpu_v3/`](./neural_cpu_v3/)：当前版本。它包含经过高精度训练的 Neural CPU v3、独立验证脚本，以及计算 Pi、解数独、搜索质数、推理 MNIST CNN 和图形渲染等演示。可直接打开[项目网页](./neural_cpu_v3/index.html)查看完整说明与结果。
- [`legacy/`](./legacy/README_zh.md)：早期概念验证与演进过程，包括基础 Neural CPU、专用 GCD 核心、通用 ALU 和早期程序执行实验。

当前版本与历史版本彼此独立。若要复现最新结果，请从 `neural_cpu_v3/` 开始。

## English

This directory contains two generations of neural-processor experiments:

- [`neural_cpu_v3/`](./neural_cpu_v3/): the current release, including the high-accuracy Neural CPU v3, an independent validator, and demonstrations covering Pi calculation, Sudoku solving, prime search, MNIST CNN inference, and graphics rendering. Open the [project site](./neural_cpu_v3/index.html) for the complete explanation and results.
- [`legacy/`](./legacy/README_en.md): early proofs of concept and the development path from basic Neural CPUs to specialized GCD cores, a general ALU, and early program-execution experiments.

The current and legacy projects are kept separately. Start with `neural_cpu_v3/` to reproduce the latest results.
