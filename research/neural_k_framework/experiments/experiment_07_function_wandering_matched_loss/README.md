# E07：函数 ID 迁移与 matched-loss 静态对照

## 目的

E07 把 E06 的静态 loss 几何与真实 optimizer 运输放进完全相同的微型协议中比较：

1. 直接记录训练集完全拟合以后，每条轨迹的完整 function ID 怎样变化；
2. 对同一网络、同一训练集大量采样未训练权重；
3. 在几乎相同的 normalized loss 下比较静态函数质量与 SGD 函数分布。

它回答的不是“函数是否会动”，而是“相同 loss 是否足以决定相同函数分布”。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`experiment_function_id_wandering.py`](experiment_function_id_wandering.py)
- [`experiment_wandering_matched_loss_prior.py`](experiment_wandering_matched_loss_prior.py)

运行顺序：

```bash
python experiment_function_id_wandering.py
python experiment_wandering_matched_loss_prior.py
```

第二个脚本读取第一个脚本保存的轨迹与 logits。

## 冻结来源

| 文件 | 开发版来源 | SHA256 |
|---|---|---|
| `experiment_function_id_wandering.py` | `research/function_information_conservation/experiment_function_id_wandering.py` | `86b1d1599ec3aba0230340500b41685fd69d6d97fc1dc8d3f610fe75f1a57b57` |
| `experiment_wandering_matched_loss_prior.py` | `research/function_information_conservation/experiment_wandering_matched_loss_prior.py` | `a6af0501654d7b100905dd705ca40ed19edccc0be02f0eec59f59ba1a11e2828` |

本地原始结果：

| 结果包 | SHA256 |
|---|---|
| `E:\Downloads\results_function_id_wandering.zip` | `a3be58b7d2c4cb5636abb86a05bfb3f221de4f504491df8429bd73b14066517d` |
| `E:\Downloads\results_wandering_matched_loss_prior.zip` | `93d8fb721452fa6d12d58afb456c338109dfd341d2aa72300b184219d0d57e89` |

结果可快速重建，ZIP 不进入最终发布包。
