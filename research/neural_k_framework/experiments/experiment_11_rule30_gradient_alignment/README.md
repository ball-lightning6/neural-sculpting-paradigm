# E11：Rule30 数据量与训练/验证梯度对齐

## 目的

E11 检验：增加同一规则的训练样本，是否只改变最终泛化终点，还是会在相同 raw BCE 水平下系统性改变训练梯度与独立验证梯度的方向关系。

它为“有限训练集只支持规则对齐下降到有限 loss 深度；数据越多，该通道延伸得越深”提供局部动力学证据。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`experiment_rule30_train_val_gradient_alignment.py`](experiment_rule30_train_val_gradient_alignment.py)

运行：

```bash
python experiment_rule30_train_val_gradient_alignment.py
```

## 冻结来源

| 文件 | 开发版来源 | SHA256 |
|---|---|---|
| `experiment_rule30_train_val_gradient_alignment.py` | `research/ca_phase_transition/experiment_rule30_train_val_gradient_alignment.py` | `811ff2b3fb8b6bc16431b8226c02540633591be1b75c98b16948f6933bde00df` |

本地结果包：`E:\Downloads\results_rule30_train_val_gradient_alignment_package.zip`SHA256：`dd0f2b0dbf3d5b829a23199edda5507335885d79ff5197e3860f4d01f0ca702b`。

ZIP 可快速重建，不进入最终发布包。
