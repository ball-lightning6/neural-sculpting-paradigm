# E28：MNIST 50k likelihood加权后验HMC与同协议Adam

## 目的

检验Gaussian先验乘以训练集likelihood的整网后验HMC采样，能否在完整50,000样本
MNIST十分类任务上达到同协议Adam的预测水平，并测量函数agreement与残余链间差异。

**证据定位（2026-10-21澄清）：这是likelihood加权后验的预测对照，不是固定loss阈值下直接条件体积的验证。** 本实验的目标是

$$
Q_{\beta,D}(d\theta)=Z_{\beta,D}^{-1}e^{-\beta L_D(\theta)}\mu(d\theta),
\qquad \beta=N=50{,}000.
$$

其中$\mu$是脚本中归一化参数的标准Gaussian先验，$L_D$是平均十分类交叉熵。它不同于

$$
Q_{D,\epsilon}(d\theta)=
\frac{\mathbf 1_{\{L_D(\theta)\leq\epsilon\}}\mu(d\theta)}
{\mu\{L_D\leq\epsilon\}}.
$$

硬阈值条件化在允许区域内保留原测度的相对权重；指数加权仍按loss重新分配该区域
内部的权重，也不设置同一个有限loss截止。这不是可忽略的实现差异，不能假定函数
分布或预测接近。HMC使用梯度，本实现也没有估计阈值体积或Neural K；99.03%的
结果不能转述成硬阈值系综在完整MNIST上达到99.03%。本次澄清不改变冻结脚本与结果。

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

