# 2026-09-21 当前上传文件清单

基线：Git `4596b2b`（2026-09-02）。本文件只描述应进入公开仓库/论文配套材料的范围，不执行 Git 提交、Zenodo 发布或 GitHub 推送。

## A. 主包根目录材料

上传以下文件：

```text
README.md
WEB_README.md
build_site.py
validate_site.py
short_paper_zh.md
short_paper_en.md
theory_core_zh.md
theory_core_en.md
evidence_ledger_zh.md
evidence_ledger_en.md
references_zh.md
references_en.md
minimal_strict_statistical_language_tutorial_zh.md
papers/README.md
papers/compression_is_intelligence_zh.md
papers/compression_is_intelligence_en.md
papers/compression_is_intelligence_v1.1_en.pdf
```

`site/`整目录随主包上传，当前为 110 个中英文 HTML 页面；`source_docs/`提供网页中引用的完整理论源稿；`assets/`中被网站实际引用的 PNG/图像和 `web_assets/`网站源资源一起保留。网页不复制 JSON/CSV/分析脚本；这些结果由实验目录中的代码重跑，报告中的对应条目只保留说明文字。

## B. E01--E30 基础实验

上传：

```text
experiments/experiment_01_* 到 experiments/experiment_30_*
```

每个目录保留 README、预注册/动机、结果结论、复现脚本和必要图表。不要包含 `experiment_31_static_agreement_paths/`。

## C. 提交后新增实验的公开编号映射

物理目录名和公开编号不同，公开说明以右列为准：

| 公开编号 | 物理目录 | 内容 |
|---|---|---|
| E31 | `experiments/experiment_31_loss_amplitude_grokking/` | Power 原版 mod97 的 CE+幅度对齐 grokking、raw MSE 0/1 与 0/15 两轮，以及 4-bit/Rule30 幅度对照 |
| E32 | `experiments/experiment_32_random_memory/` | 随机记忆近线性收缩，含外部复杂度校准说明 |
| E33 | `experiments/experiment_33_cross_architecture/` | 跨架构联合输出体积 |
| E34 | `experiments/experiment_34_conditional_rule_duel/` | 75 次条件规则决斗 |
| E35 | `experiments/experiment_35_mod97_frozen_function_duels/` | grokking 前后冻结函数决斗 |
| E36 | `experiments/experiment_36_peleg_g_and_c_validation/` | Peleg/G&C 独立验证 |
| E37 | `experiments/experiment_37_measure_robustness/` | Gaussian、固定范数球欧式测度、端点响应与真实训练对应 |

E15/E23 的提交后补充分别放在其对应的发布树目录中；基础实验页仍使用旧 E15/E23 页面，不能重复改号。

## D. 主包理论与核验

上传：

```text
release_20260920/theory/minimal_strict_statistical_language_zh.md
release_20260920/theory/minimal_statistical_language_audit_zh.md
release_20260920/theory/check_minimal_statistical_language.py
release_20260920/theory/minimal_statistical_language_check_results.json
release_20260920/theory/MDL_FRONTIER_AND_GROKKING.md
release_20260920/theory/mdl_identifiability_grokking_derivation_zh.md
release_20260920/theory/WEBSITE_PROBABILITY_LANGUAGE_ZH.md
release_20260920/theory/WEEKEND_THEORY_UPDATE.md
release_20260920/theory/MEASURE_ROBUSTNESS_AND_GAUSSIAN_ALIGNMENT_ZH.md
supplementary/architecture_estimator/WEBSITE_ESTIMATOR_FULL_ZH.md
```

网站对应的英文源：

```text
translations_en/theory/probability_language.md
translations_en/theory/mdl_identifiability_grokking_derivation.md
translations_en/theory/weekend_theory_update.md
translations_en/theory/measure_robustness_and_gaussian_alignment.md
translations_en/architecture_bias/website_estimator_full.md
```

`translations_en/`整目录上传：包括 E01--E30 全文翻译和提交后 E31--E37 全文翻译。

## E. 补充包

建议上传：

```text
supplementary/architecture_estimator/
```

测度鲁棒性材料的公开入口已统一为 `experiments/experiment_37_measure_robustness/`（见 C 节），本地 `supplementary/measure_robustness/` 是其重复副本，不再进入公开包。原始大粒子记录不进入网页包。

保留 architecture_estimator 中的 README、方法说明、`current_method/`脚本与协议、NEURAL_K_BACKTEST_REPORT、NEURAL_K_LARGE_BENCHMARK_REPORT、PROVENANCE_CORRECTION、CSV/JSON汇总和必要 PNG。不要上传 checkpoint、粒子池、zip、缓存或临时目录。

同时可上传：

```text
experiments/experiment_36_peleg_g_and_c_validation/
```

以及本文件 D 节列出的严格概率核验脚本和 JSON。

## F. E19/E23/E24 四 bit 证据

保留父目录中 E19、E23、E24 的协议、脚本、结果和必要图表。这些是 4-bit 完整规则体积、深尾收缩和规则对决的基础证据；不要把被裁决不上传的旧规则决斗面板混入其中。

## G. 发布树管理文件

上传：

```text
release_20260920/README.md
release_20260920/MANIFEST.md
release_20260920/UPLOAD_DECISION.md
release_20260920/ARCHIVE_MAP.md
release_20260920/POST_UPLOAD_EXPERIMENTS_ZH.md
```

## H. 明确排除

以下保留本地，不进入公开包：

```text
release_20260920/evidence/rule_ood_program_generalization/
release_20260920/evidence/reachability_scaffold/
release_20260920/evidence/early_rule_learning_replay/
release_20260920/evidence/addition8_prediction/
release_20260920/experiments/E31_static_agreement_paths/
release_20260920/experiments/E36_rule_duel_panel/
release_20260920/experiments/E35_input_output_compression/
release_20260920/experiments/E32_loss_amplitude_grokking/
release_20260920/experiments/E33_random_memory/
release_20260920/experiments/E34_cross_architecture/
release_20260920/experiments/E37_conditional_rule_duel/
release_20260920/experiments/E38_mod97_frozen_function_duels/
release_20260920/evidence/peleg_recheck/
release_20260920/evidence/architecture_bias/
release_20260920/supplementary/architecture_estimator/
release_20260920/theory/ait_omega_mixture_agreement_audit_zh.md
release_20260920/theory/joint_output_rate_derivation_zh.md
release_20260920/theory/volume_contraction_conditional_theorem_zh.md
release_20260920/literature/
kimi_experiments/
```

另外排除：`.tmp_*`、`tmp/`、checkpoint、SMC 粒子池、`.pt`、`.pth`、`.npz`、`.zip`、`.jsonl`、聊天缓存、GPU 环境目录和与本研究无关的 RSA/text2image 文件。

## I. 上传前最终检查

```powershell
cd research/neural_k_framework
python -X utf8 build_site.py
python -X utf8 validate_site.py
git diff --check
```

当前结果：`Built 110 HTML pages`，`Validated 110 HTML pages`。这些命令只验证本地树，不执行发布。
