# Power 架构 mod97：原始 MSE 的 0/1 与 0/15 两轮

> 所属：E31 Loss、目标幅度与 grokking。本页补齐 raw MSE 在 Power 原版 Transformer 上的两轮结果；CE 仅作诊断，不参与梯度。

## 问题与配置

Power 原版两层、宽 128、4 头 Transformer（455,424 参数）、mod97 除法、70% 数据（6,518 训练/2,794 测试）、无 weight decay、Adam(lr=0.001, betas=(0.9,0.98))。训练目标为答案与 EOS 位置的原始 one-hot MSE；答案按样本、两个 RHS 位置和全部 239 类平均，不放大标签、不调整类别权重。每条件 3 个配对 seed（同一数据划分、同一初始化与 batch 顺序），首次测试答案 100% 即停。

## 结果

| 条件 | 恢复完整规则 | 说明 |
|---|---:|---|
| raw MSE 0/1 | **1/3** | seed 2 于 140,000 步满分；seed 0 最高 99.36% 未满分；seed 1 后期退化（训练 MSE 卡在 1.85e-5，瓶颈在隐藏表示而非输出层） |
| raw MSE 0/15（错误类 0，正确类 15） | **2/3** | 成功于 14,500 与 70,500 步（中位 42,500）；统一 100,000 步比较为 2/3 vs 0/1 的 0/3 |

参考标定点：完美 one-hot 解的"诊断 CE"为 **4.4836**——即对固定 one-hot 目标，交叉熵不趋于 0。这直接否定"grokking 必须把原始 CE 降到极低"，与 E31 中 CE+幅度对齐正则组在 CE≈1e-4–1e-3 即恢复的现象一致。

## 机制与边界（必须随结果保留）

- 两轮状态均为 `paused`（手动暂停，非跑完预算），机制状态 `mse_recovery_or_delay_not_yet_confirmed`；**首次满分即停，不保证继续训练仍稳定**；
- 3 seed、单数据划分；停止看语义答案，不把 EOS 作为容易样本抬高准确率；
- 失败 seed 未做归因（用户决定不追查），不能用"只统计成功 seed 的中位步数"代表整个条件；
- CE 仅作诊断不参与梯度；0/15 组实际训练不除以幅度平方，relative MSE 仅用于跨幅度比较；
- 不能据此声称"raw MSE 普遍能 grokking"或"0/15 普遍优于 0/1"；本页与 `POWER_MOD97_ALIGNMENT_REPORT.md`（CE+幅度对齐正则 3/3）和 CE 基线（0/3，训练 CE 已至 ~4e-11 仍未恢复）一起阅读。

## 文件

- `experiment_power_mod97_raw_mse01.py`、`experiment_power_mod97_raw_mse15.py`：冻结实验脚本；
- `analyze_power_mod97_raw_mse01.py`、`analyze_power_mod97_raw_mse15.py`：分析脚本；
- 结果包：`E:/Downloads/results_power_mod97_raw_mse01_n70_package.zip`、`results_power_mod97_raw_mse15_n70_package.zip`；
- 分析包：`tmp/power_mod97_mse01_analysis_20260911/`、`tmp/power_mod97_mse15_analysis_20260911/`（analysis.json、逐 seed 表、权重核验）。
