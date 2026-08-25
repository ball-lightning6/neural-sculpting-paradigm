# E10：4-bit raw-BCE 简单/复杂规则竞争

## 目的

E10 先用多任务大规模静态扫描检验“lower loss 普遍提高简单函数质量”，再针对训练集歧义和外部复杂度代理问题，构造训练输出完全一致、但表示难度严格有序的 simple/complex 函数对进行直接赔率检验。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果、重审计与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`experiment_loss_conditioned_prior_scaling_4bit.py`](experiment_loss_conditioned_prior_scaling_4bit.py)
- [`analyze_loss_conditioned_rule_pair_competition.py`](analyze_loss_conditioned_rule_pair_competition.py)

运行顺序：

```bash
python experiment_loss_conditioned_prior_scaling_4bit.py
python analyze_loss_conditioned_rule_pair_competition.py
```

第二个冻结脚本已把默认来源恢复为第一步生成的`/root/results_loss_conditioned_prior_scaling_4bit`，并采用最终可靠性标准。

## 冻结来源

| 文件 | 来源 | SHA256 |
|---|---|---|
| `experiment_loss_conditioned_prior_scaling_4bit.py` | 同名开发版 | `dff27be60da28f4e537c5547f68d7d2a78b69727305b0ecf646ea0c5301cf5ab` |
| `analyze_loss_conditioned_rule_pair_competition.py` | 开发版恢复Aug-19数据路径后的发布副本 | `5cff544b8b96199af4dbdd6ca2270c701639745f511d1b27f9cef755294256a2` |

原始结果：

| 结果包 | SHA256 |
|---|---|
| `E:\Downloads\results_loss_conditioned_prior_scaling_4bit.zip` | `9436665cb0ce8587b664cb9ce8da664527805022029c88cef125415656d2aa20` |
| `E:\Downloads\results_loss_conditioned_rule_pair_competition.zip` | `6df85b66acedcd670866fbf3e0bdbbc9ade06c36d8f5ff52d8f261411049a388` |

旧 pair ZIP 中的自动`summary.json`使用了已废弃的尾部可靠性标准。正文数字来自`pair_odds_trajectories.csv`按最终零假设期望计数规则的重审计，冻结分析脚本可在重新生成 prior shards 后完整复现。ZIP 不进入最终发布包。
