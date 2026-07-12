# 过拟合相关研究 / Overfitting-Related Research

> **先看这里 / Start here:**  
> **[完整说明页 / Main explanation page](index.html)**  
> **[交互式结果页 / Interactive dashboard](results/probe_consistency_dashboard.html)**

这个目录是一组围绕“过拟合到底是不是单纯记忆”的后续实验。核心做法是：对同一训练集训练多个随机 seed，然后在未见 probe 集上比较它们的输出一致性、预测熵和共同错误结构。这样可以直接观察训练数据如何约束模型在训练集之外形成的函数分布。

This directory contains follow-up experiments on whether overfitting is merely memorization. The core method is to train multiple random seeds on the same training set, then compare their agreement, prediction entropy, and shared-error structure on unseen probe inputs. This gives a direct behavioral view of how training data constrains the function distribution outside the training set.

## 当前实验 / Current Experiment

当前版本比较 `random_bits` 和 Rule 30 的 1/2/3 层任务，并扫描多个训练样本数 `n`。主要观察是：随机标签任务中，probe accuracy 始终接近 0.5，但跨 seed agreement 随数据量增加下降；规则任务中，随着数据量增加，probe accuracy 和 agreement 最终共同上升并趋向真实规则。

The current version compares `random_bits` with Rule 30 layer 1/2/3 tasks across multiple training-set sizes `n`. The main observation is that in the random-label task, probe accuracy stays near 0.5 while cross-seed agreement decreases with more data; in rule-based tasks, probe accuracy and agreement eventually rise together toward the true rule.

## 文件 / Files

- [index.html](index.html): 主要说明页，包含研究动机、实验设计、指标解释、图表和理论讨论。
- [results/probe_consistency_dashboard.html](results/probe_consistency_dashboard.html): 可交互结果页。
- [results/probe_consistency_dashboard_aggregated.csv](results/probe_consistency_dashboard_aggregated.csv): 聚合结果表。
- `scripts/generate_ca_rule_dataset.py`: 生成 Rule 30 数据集。
- `scripts/sweep_task_difficulty_plateau.py`: 运行训练 sweep。
- `scripts/build_probe_consistency_dashboard.py`: 重新聚合结果并生成 dashboard。

- [index.html](index.html): Main explanation page with motivation, experimental design, metric definitions, charts, and theoretical discussion.
- [results/probe_consistency_dashboard.html](results/probe_consistency_dashboard.html): Interactive result dashboard.
- [results/probe_consistency_dashboard_aggregated.csv](results/probe_consistency_dashboard_aggregated.csv): Aggregated result table.
- `scripts/generate_ca_rule_dataset.py`: Generates Rule 30 datasets.
- `scripts/sweep_task_difficulty_plateau.py`: Runs the training sweep.
- `scripts/build_probe_consistency_dashboard.py`: Re-aggregates results and rebuilds the dashboard.

## 复现 / Reproduction

```bash
python research/overfitting_related_research/scripts/generate_ca_rule_dataset.py
python research/overfitting_related_research/scripts/sweep_task_difficulty_plateau.py
python research/overfitting_related_research/scripts/build_probe_consistency_dashboard.py
```

训练 sweep 需要 `torch`、`numpy` 和 `tqdm`。如果只想看结果，直接打开上面的两个 HTML 页面即可。

The training sweep requires `torch`, `numpy`, and `tqdm`. If you only want to inspect the results, open the two HTML pages linked above.
