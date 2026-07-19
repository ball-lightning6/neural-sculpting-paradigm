# 过拟合相关研究

这个目录整理一组围绕“过拟合之后，神经网络在训练集外到底实现了什么函数”的实验。

建议阅读顺序：

1. [函数后验采样补充实验](function_posterior_sampling_experiments.html)
2. [主说明页](index.html)
3. [交互式结果页](results/probe_consistency_dashboard.html)

## 当前上传范围

这次保留的是相对干净、容易复现、解释比较明确的实验：

- 多 seed 训练同一个过拟合训练集，在 probe 集上比较跨 seed agreement。
- 未训练 MLP 的先验 baseline，以及极小训练集下 agreement 从接近 0.5 快速跃升的现象。
- 主动采样实验：用 ensemble 不确定性选样，比较 `uncertain / random / certain` 三种策略。
- 单 seed 在稳定训练后按时间间隔采样，检验它是否能近似替代多 seed 的函数后验采样。
- grokking 时间轴上的 agreement 曲线，用来和数据量轴上的 agreement 变化作对照。

后续一些更复杂、混杂因素更多的探索暂时不作为本次主线上传内容。

## 目录结构

- [function_posterior_sampling_experiments.html](function_posterior_sampling_experiments.html)：函数后验采样补充实验说明和图表。
- [index.html](index.html)：过拟合相关研究的主说明页。
- [results/probe_consistency_dashboard.html](results/probe_consistency_dashboard.html)：任务难度与数据量 sweep 的交互式 dashboard。
- [results_function_posterior_sampling/](results_function_posterior_sampling/)：函数后验采样补充页面使用的轻量 CSV/JSON 数据。
- [scripts/function_posterior_sampling_experiments/](scripts/function_posterior_sampling_experiments/)：函数后验采样补充实验的 Python 复现脚本。
- [scripts/build_function_posterior_sampling_dashboard.py](scripts/build_function_posterior_sampling_dashboard.py)：重建函数后验采样补充页面和轻量数据的脚本。

## 复现与重建网页

如果只想查看结果，直接打开 HTML 文件即可。

## 脚本说明

函数后验采样补充实验的复现脚本位于 [scripts/function_posterior_sampling_experiments/](scripts/function_posterior_sampling_experiments/)：

| 脚本 | 用途 |
|---|---|
| `train_ca_overfit_ensemble.py` | 对一个任务和一个训练样本数训练多个 seed，并保存 probe 预测。 |
| `analyze_ca_overfit_ensemble.py` | 分析一个 ensemble 结果目录，计算 agreement、entropy、共同错误等统计。 |
| `sweep_overfit_ensemble_grid.py` | 批量扫描多个训练样本数或任务，用于 tiny-n 和数据量轴实验。 |
| `untrained_mlp_prior_probe.py` | 不训练 MLP，只采样随机初始化函数，用作 n=0 先验 baseline。 |
| `train_single_seed_time_sampling.py` | 单 seed 平台期后按时间间隔采样，和多 seed 系综作对照。 |
| `active_soft_uncertainty_sampling_manual_ca.py` | 多 seed 主动采样实验，比较 `uncertain / random / certain`。 |
| `active_time_committee_sampling_ca.py` | 单 seed 时间委员会版本的主动采样实验。 |
| `train_grokking_agreement_time_axis.py` | 固定训练集，沿训练 step 记录跨 seed agreement。 |
| `scripts/build_function_posterior_sampling_dashboard.py` | 从轻量数据或原始 zip 包重建补充实验网页。 |

如果只想根据仓库内已经提交的轻量数据重建函数后验采样补充页面：

```bash
python research/overfitting_related_research/scripts/build_function_posterior_sampling_dashboard.py
```

如果要从原始 zip 包重新抽取轻量数据，把 zip 放到：

```text
research/overfitting_related_research/source_packages_function_posterior_sampling/
```

然后运行同一个 build 脚本。也可以显式指定 zip 所在目录：

```bash
python research/overfitting_related_research/scripts/build_function_posterior_sampling_dashboard.py --source-dir path/to/source_zips
```

脚本不依赖本机硬编码路径。默认输出会落在：

```text
research/overfitting_related_research/results_function_posterior_sampling/
research/overfitting_related_research/function_posterior_sampling_experiments.html
```

训练脚本需要 `torch`、`numpy`、`tqdm`，可视化重建只需要 Python 标准库。
