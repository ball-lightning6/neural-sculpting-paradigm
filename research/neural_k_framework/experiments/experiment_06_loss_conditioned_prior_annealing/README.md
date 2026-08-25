# E06：连续 loss 条件下的静态函数质量

## 目的

该实验第一次直接测量：在网络、初始化分布和有限训练集都固定，并且所有参数已经满足相同 hard labels 后，继续收紧训练集 raw BCE 是否会改变完整函数的相对参数质量。

它是本文从“hard-conditioned 函数先验”转向“loss-resolved 函数体积”的第一项静态实验。它同时检验静态低-loss 质量能解释真实 SGD 方向的多少，以及两者是否可以等同。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`experiment_loss_conditioned_prior_annealing.py`](experiment_loss_conditioned_prior_annealing.py)

运行：

```bash
python experiment_loss_conditioned_prior_annealing.py
```

默认脚本为`PROFILE="pilot"`。复现正文低-loss 尾部时改成：

```python
PROFILE = "full"
```

## 冻结来源

| 文件 | 开发版来源 | SHA256 |
|---|---|---|
| `experiment_loss_conditioned_prior_annealing.py` | `research/function_information_conservation/experiment_loss_conditioned_prior_annealing.py` | `a66b973f118555812728b234ad5f02cb20f38d11b78b848722beff03c3a84310` |

本地原始结果：

| 结果包 | 作用 | SHA256 |
|---|---|---|
| `E:\Downloads\results_loss_conditioned_prior_annealing.zip` | 262,144网络pilot | `35573b995520ed868e1141bb408426395a04c91ec89e90b756c74b9c2a1c12fc` |
| `E:\Downloads\results_loss_conditioned_prior_annealing (1).zip` | 4,194,304网络full裁决 | `42c70ffe74a4f751c9be47f36ee94edf9fdf97b7089f017c48648dca71538381` |

两个 ZIP 均可快速重建，不进入最终发布包；正文数字以 full 结果为准。
