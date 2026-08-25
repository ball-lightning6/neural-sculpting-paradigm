# E02：Rule-bit 反事实不变性与训练 loss 中心原则

## 目的

检验固定控制位是否会被网络自动识别为语义无关变量，以及加入多少反事实样本才能让两条控制分支严格执行同一 ECA 规则。该实验同时区分：

- 训练中恒0与恒1；
- binary `0/1`与 centered `-1/+1`编码；
- 已学规则后追加反事实样本（warm）；
- 从头训练最终数据（cold）；
- 继续训练但不加反事实样本（`k=0`）。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`experiment_rule_bit_invariance_completion.py`](experiment_rule_bit_invariance_completion.py)

运行：

```bash
python experiment_rule_bit_invariance_completion.py
```

默认输出目录：

```text
results_rule_bit_invariance_completion/
```

## 冻结来源与 SHA256

| 冻结文件 | 开发版来源 | SHA256 |
|---|---|---|
| `experiment_rule_bit_invariance_completion.py` | `research/overfitting_related_research/scripts/experiment_rule_bit_invariance_completion.py` | `6c0ee495797ed215774c37ae6b1b3f25b73b4145d8dc5ff1dda124cbe85f9210` |

本地原始结果缓存：`results/results_rule_bit_invariance_completion.zip`，SHA256`2932bc1ed92c4b7e0c67f251ca0ee595f6e5f49f0cd0c66162dcc07ae279ced4`。该 ZIP 不进入最终上传包，关键结果已完整抄入结果报告。
