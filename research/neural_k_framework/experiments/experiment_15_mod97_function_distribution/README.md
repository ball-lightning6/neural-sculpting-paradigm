# E15：Mod97 grokking 的完整函数与边缘凝聚

## 目的

E15 在原版风格 Mod97 除法任务中同时跟踪完整9,312点 hard function、逐输入边缘众数、agreement、target-function 质量和 matched train CE，区分“完整函数样本仍全部不同”与“逐点预测已经围绕同一规则凝聚”。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`experiment_mod97_matched_loss_function_distribution.py`](experiment_mod97_matched_loss_function_distribution.py)

运行：

```bash
python experiment_mod97_matched_loss_function_distribution.py
```

脚本 SHA256：`f87700c439929ce1e5dbc6cf0600ac88b71c453598646054b9758472ed110faf`。结果 ZIP SHA256：`4bf8e597268411c2c7dc483862ab871cfa22b4fe0fb1a1a0c899b4ea709381c8`。

原始 ZIP 位于`E:\Downloads`，可重建且不进入发布包。
