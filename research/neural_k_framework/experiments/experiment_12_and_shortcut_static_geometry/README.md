# E12：AND shortcut 的静态 loss 几何

## 目的

E12 沿完整 raw-BCE 范围追踪简单目标函数，并定向调查 AND、n=10中出现的真实反例：为什么结构上更复杂的兼容 shortcut 会在更深 loss 截面超过 AND，以及只改变训练样本覆盖能否因果消除该优势。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`experiment_4bit_simple_rule_loss_sweep.py`](experiment_4bit_simple_rule_loss_sweep.py)
- [`analyze_and_shortcut_loss_geometry.py`](analyze_and_shortcut_loss_geometry.py)

运行顺序：

```bash
python experiment_4bit_simple_rule_loss_sweep.py
python analyze_and_shortcut_loss_geometry.py
```

发布副本的第二步默认同时分析原始缺口、三种补洞、镜像缺口和最小平衡干预。

## 冻结来源

| 文件 | 来源 | SHA256 |
|---|---|---|
| `experiment_4bit_simple_rule_loss_sweep.py` | 同名开发版 | `8c8c65d52ec43726f1615a6119a70dcca9e4e85edfe554c60bd16523a8b8a416` |
| `analyze_and_shortcut_loss_geometry.py` | 开发版合并发布条件后的副本 | `76833fc466d79db20a5619190bedfbfc911f2914091e9b2a4fb05697272ad53c` |

本地结果：

| 结果包 | SHA256 |
|---|---|
| `results_4bit_simple_rule_loss_sweep_package.zip` | `0b825aca4f8f0350d900d3d118a834930b56215a4159eea48f396e40c287c006` |
| `results_and_shortcut_loss_geometry_package.zip` | `57467aeaa1fa66c237ed3d379994f615862311eb268ad94a467aba2a22d5e417` |
| `results_and_shortcut_balanced_prior_geometry_package.zip` | `7fb2adf631b98de6f16a0f72cc740d2564938a64a374887e09baae7661cd8876` |

原始 ZIP 位于`E:\Downloads`，可由脚本重建，不进入发布包。
