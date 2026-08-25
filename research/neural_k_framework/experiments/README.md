# 短论文实验代码

本目录按短论文证据出现顺序冻结可上传脚本。每个 `experiment_XX_*` 文件夹对应正文中的一个明确实验单元，并包含：

- 原样冻结的自包含脚本；
- 运行顺序和输出说明；
- 源文件位置与 SHA256；
- 该实验能支持和不能支持的主张。

开发版脚本仍位于 `research/`。论文目录副本用于复现与归档，不应在没有同步源码哈希、正文方法和结果说明的情况下单独修改。

当前进度：

| ID | 实验 | 入口 |
|---|---|---|
| E01a | 静态条件先验 vs Adam逐函数分布 | [`experiment_01_static_conditioning_failure`](experiment_01_static_conditioning_failure/README.md) |
| E01b | 1024×3单样本函数质量运输 | [`experiment_01_static_conditioning_failure`](experiment_01_static_conditioning_failure/README.md) |
| E02 | Rule-bit反事实不变性与训练loss中心原则 | [`experiment_02_rule_bit_invariance`](experiment_02_rule_bit_invariance/README.md) |
| E03a | Prior-consistent函数系综动力学 | [`experiment_03_function_posterior_dynamics`](experiment_03_function_posterior_dynamics/README.md) |
| E03b | tanh宽度扫描与网络参考语言 | [`experiment_03_function_posterior_dynamics`](experiment_03_function_posterior_dynamics/README.md) |
| E04 | [Mingard 等（2025）](https://www.nature.com/articles/s41467-024-54813-x)协议的 post-fit 函数漂移 | [`experiment_04_mingard_postfit_drift`](experiment_04_mingard_postfit_drift/README.md) |
| E05 | 固定NTK足够时的特征学习 | [`experiment_05_ntk_sufficient_feature_learning`](experiment_05_ntk_sufficient_feature_learning/README.md) |
| E06 | 连续loss条件下的静态函数质量 | [`experiment_06_loss_conditioned_prior_annealing`](experiment_06_loss_conditioned_prior_annealing/README.md) |
| E07 | Function-ID迁移与matched-loss静态对照 | [`experiment_07_function_wandering_matched_loss`](experiment_07_function_wandering_matched_loss/README.md) |
| E08 | 共享中间表示与有限容量压力 | [`experiment_08_shared_intermediate_pressure`](experiment_08_shared_intermediate_pressure/README.md) |
| E09 | MNIST 80%隐藏标签噪声长训 | [`experiment_09_mnist_hidden_noise_long_training`](experiment_09_mnist_hidden_noise_long_training/README.md) |
| E10 | 4-bit raw-BCE简单/复杂规则竞争 | [`experiment_10_raw_bce_rule_pair_competition`](experiment_10_raw_bce_rule_pair_competition/README.md) |
| E11 | Rule30数据量与训练/验证梯度对齐 | [`experiment_11_rule30_gradient_alignment`](experiment_11_rule30_gradient_alignment/README.md) |
| E12 | AND shortcut静态loss几何 | [`experiment_12_and_shortcut_static_geometry`](experiment_12_and_shortcut_static_geometry/README.md) |
| E13 | AND shortcut的SGD运输与单样本干预 | [`experiment_13_and_shortcut_sgd_intervention`](experiment_13_and_shortcut_sgd_intervention/README.md) |
| E14 | 加权Rule-bit的loss阶段函数换序 | [`experiment_14_weighted_rulebit_function_switch`](experiment_14_weighted_rulebit_function_switch/README.md) |
| E15 | Mod97 grokking的完整函数与边缘凝聚 | [`experiment_15_mod97_function_distribution`](experiment_15_mod97_function_distribution/README.md) |
| E16 | Parity终点偏好、全局入口与局部恢复 | [`experiment_16_parity_reachability`](experiment_16_parity_reachability/README.md) |
| E17 | 静态低-loss质量与optimizer运输 | [`experiment_17_static_vs_optimizer`](experiment_17_static_vs_optimizer/README.md) |
| E18 | 高函数共识与符号可读性 | [`experiment_18_consensus_symbolicity`](experiment_18_consensus_symbolicity/README.md) |
| E19 | 完整真值表规则的loss-resolved体积 | [`experiment_19_full_truth_rule_volume`](experiment_19_full_truth_rule_volume/README.md) |
| E20 | Full-rule、固定D与hard-margin测度闭环 | [`experiment_20_measure_closure`](experiment_20_measure_closure/README.md) |
| E21 | Agreement控制与共识复杂度前沿 | [`experiment_21_agreement_complexity_frontier`](experiment_21_agreement_complexity_frontier/README.md) |
| E22 | 逐样本自由能与信息不变量 | [`experiment_22_static_free_energy_information`](experiment_22_static_free_energy_information/README.md) |
| E23 | 完整目标体积前瞻预测数据相变 | [`experiment_23_volume_to_data_transition`](experiment_23_volume_to_data_transition/README.md) |
| E24 | 深尾Neural K-profile与排序交叉 | [`experiment_24_deep_neural_k_crossing`](experiment_24_deep_neural_k_crossing/README.md) |
| E25 | MNIST样本复杂度与静态体积分支预测 | [`experiment_25_mnist_static_prediction`](experiment_25_mnist_static_prediction/README.md) |

当前时间线主实验已整理至2026年8月25日。后续实验仍按同一规则逐项加入，不提前收纳尚未完成论文定位的脚本。
