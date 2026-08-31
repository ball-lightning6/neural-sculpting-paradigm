# E27：Grokking前后的未见函数Agreement

## 目的

检验模型已经完全拟合训练集但尚未grokking时，不同seed是否共享同一个错误
函数，还是仍在未见输入上保持分散；并测量Agreement怎样随规则泛化变化。

## 入口

- experiment_mod97_matched_loss_function_distribution.py
- analyze_agreement_around_grokking.py
- MOTIVATION_AND_PREREGISTRATION.md
- RESULTS_AND_CONCLUSION.md

分析脚本读取原E15的trajectory.csv，不需要重新训练。主口径只在当前训练比例
的未见Mod97输入上计算pairwise seed Agreement。

## 冻结来源

| 文件 | 来源 | SHA256 |
|---|---|---|
| experiment_mod97_matched_loss_function_distribution.py | E15冻结脚本 | f87700c439929ce1e5dbc6cf0600ac88b71c453598646054b9758472ed110faf |
| analyze_agreement_around_grokking.py | E15补充分析 | 52e45aa9086cdc24887713694c2c7d6d36fd8e522bf41ff3d81770000aaa8d79 |

原始结果包results_mod97_matched_loss_function_distribution_package.zip的SHA256为
4bf8e597268411c2c7dc483862ab871cfa22b4fe0fb1a1a0c899b4ea709381c8。

