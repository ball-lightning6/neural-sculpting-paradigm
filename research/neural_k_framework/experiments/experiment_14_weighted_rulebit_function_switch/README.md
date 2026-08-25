# E14：加权 Rule-bit 的 loss 阶段函数换序

## 目的

E14 人为构造多数 Rule110 与少数 Rule30 的层级约束，使高 loss 时忽略少数分支最经济，而极低 loss 时必须满足完整复合映射。它直接检验固定 hard function 的概率是否必须随 loss 单调。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`experiment_weighted_rulebit_sgd.py`](experiment_weighted_rulebit_sgd.py)

运行：

```bash
python experiment_weighted_rulebit_sgd.py
```

## 冻结来源

脚本 SHA256：`db944d90827da3ab3d4d68e2771ad86ee8639f27a67f166fbbf939923691bbcb`。开发版：`research/loss_level_function_switch/experiment_weighted_rulebit_sgd.py`。

本地结果：`E:\Downloads\results_weighted_rulebit_sgd_package.zip`SHA256：`c3ce46a4c99a8d5d1a9a6c19cbb1d359952bb2e50f9b8e1bf1541505d8604cbe`。

ZIP 含 checkpoint 且约177 MB，不进入发布包。
