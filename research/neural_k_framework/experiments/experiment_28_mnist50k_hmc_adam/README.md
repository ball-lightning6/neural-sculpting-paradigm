# E28：MNIST 50k整网HMC与同协议Adam

## 目的

检验Gaussian参考测度下的完整有限宽静态HMC系综，能否在完整50,000样本
MNIST十分类任务上达到真实Adam训练的预测水平，并测量函数agreement与残余
链间差异。

## 入口

- experiment_mnist10_small_cnn_full_hmc_demo.py
- experiment_mnist10_small_cnn_adam_same_protocol.py
- MOTIVATION_AND_PREREGISTRATION.md
- RESULTS_AND_CONCLUSION.md

HMC脚本默认profile为50k。Adam脚本保留8k复现profile，但本单元只报告50k
HMC、plain Adam和MAP Adam。

## 冻结来源

| 文件 | 开发版来源 | SHA256 |
|---|---|---|
| experiment_mnist10_small_cnn_full_hmc_demo.py | research/function_information_conservation | 24d09f6ef8e5b227dcce2a8992f413d25a6ee6542cfc6252782de376c6377ae1 |
| experiment_mnist10_small_cnn_adam_same_protocol.py | research/function_information_conservation | af56b60af8ab59f3416a0a21047a8264dff48abae832b256001f0b5994538858 |

原始50k HMC与Adam包的SHA256分别为：

- fdb11e1ad74df89ae349aaef0cec6b795097ccaac409dcac4c71d7e335fa1c96
- 118559b4b5177c9ab593111923c653a1bb80f03bb9b45696940b56705db2a42d

