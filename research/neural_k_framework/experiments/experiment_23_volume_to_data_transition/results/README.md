# E23 补充实验精简结果

这里保留 2026-08-26 三项补充实验中足以审计论文数字、曲线与数值边界的精简结果。完整 checkpoint、逐步 trajectory、全部 posterior prediction 和原始 ZIP 不进入仓库；三个脚本都可以重新生成完整结果。

## 1. `deep_tail/`

对应 `experiment_8bit_mismatch_joint_deep_bridge.py`：

- `deep_mismatch_bridge.png`：五目标 Gaussian 深尾体积与预注册比值；
- `pairwise_volume_ratios.csv`：共同阈值上的 replica 体积比；
- `pairwise_ratio_growth_rates.csv`：深窗相对收缩速度；
- `stopping_diagnostics.json`：交叉与停止判据；
- `summary.json`：运行状态。

完整结果包 SHA256：

```text
56FA81613CFC3DFF4DA8AED91C261619B10781972C6DD248334E12ABE7285F2E
results_8bit_mismatch_gaussian_joint_deep_bridge_package.zip
```

## 2. `coverage_intervention/`

对应 `experiment_parity2_mux3_coverage_shortcut_intervention.py`：

- `coverage_shortcut_intervention_curves.png`：uniform、cell、conflict 三协议的恢复率、agreement、目标准确率和目标函数质量；
- `transition_estimates.csv`：`n50/n90` 与 target-aligned agreement 亚网格位置；
- `transition_bootstrap_contrasts.csv`：2,000次配对 dataset-bootstrap 干预差；
- `same_n_summary.csv`：每个样本数的完整聚合统计；
- `summary.json`：资格检查与自动裁决。

完整结果包 SHA256：

```text
A59FE2E0D225E9FA94924937810A3DEE0FBD8F8CA5DE6C4FC4C027C70340EB2F
results_parity2_mux3_coverage_shortcut_intervention_package.zip
```

## 3. `fixed_d_smc/`

对应 `experiment_parity2_mux3_fixed_d_static_smc.py`：

- `fixed_d_static_smc_curves.png`：三协议、两目标的静态 unseen accuracy、exact target mass 与 agreement；
- `protocol_target_summary.csv`：10个 loss 阈值的协议级主结果；
- `condition_threshold_summary.csv`：8份数据集逐条件结果；
- `replica_threshold_summary.csv`：32个 replica 的 target mass、lineage 与 log-volume；
- `smc_levels.csv`：1,154层 survival、acceptance 与阈值轨迹；
- `top_functions_eps002.csv`：最深阈值的 top hard functions；
- `posterior_cell_summary_eps002.csv`：最深阈值按语义格聚合的后验目标准确率；
- `summary.json`：运行状态。

完整结果包 SHA256：

```text
C6D81CF65A4BDC5C956E95EB04185667DC208704BA502684992FFD67082EFAF6
results_parity2_mux3_fixed_d_static_smc_package.zip
```

## 数值边界

Gaussian 深尾与 fixed-D SMC 在最深处均出现 lineage 收缩，因此绝对 log-volume 或 target-mass 小数不应当成高精度积分值。主裁决依赖预注册方向、跨 loss 窗口稳定性、独立数据集、replica 同号、定向干预和真实 optimizer 外部对照。

