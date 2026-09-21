# 上传裁决表

基准：2026-09-02短论文提交，Git `4596b2b`。

这里的“上传”分成三种：

- **主包**：应进入公开仓库/论文配套材料，读者需要它理解核心主张；
- **补充包**：值得公开，但不应挤进短论文主线；
- **归档**：保留本地或单独备份，不进入公开上传包。

## 一、主包：应上传

| 材料 | 路径 | 原因 |
|---|---|---|
| 修订后的正式短论文 | `papers/compression_is_intelligence_{zh,en}.md`及本地v1.1 PDF | 当前主叙事、联合输出和静态/动态边界已经修正 |
| 理论核心 | `theory_core_zh.md`、对应英文核心 | 给出最终主张、Neural K、MDL、Agreement和优化可达性边界 |
| 严格概率语言 | `minimal_strict_statistical_language_tutorial_zh.md`及审计结果 | 修复 full-target/fixed-D、KL方向、信息增益和loss事件混淆 |
| MDL—grokking桥 | `theory/mdl_identifiability_grokking_derivation_zh.md` | 形式化复杂度、竞争函数分歧和数据相变的关系 |
| 测度鲁棒性补充 | `theory/MEASURE_ROBUSTNESS_AND_GAUSSIAN_ALIGNMENT_ZH.md`、`../experiments/experiment_37_measure_robustness/` | 纠正 Gaussian/欧式体积混淆，复核多种参考测度下的排序与真实训练对应边界 |
| 4-bit规则曲线/规则对决 | 父目录 E19/E23/E24及对应结果摘要 | 核心静态体积和收缩证据 |
| E15/E27 Agreement | `experiments/E15_grokking_agreement` | 证明 hard fit 后 Agreement不等于已泛化，并与grokking共同变化 |
| E23 Parity2/MUX3 | `experiments/E23_volume_to_data_transition` | 证明完整目标 Neural K 不能单独决定数据相变，竞争延拓和样本覆盖很重要 |
| E31 raw MSE与幅度对齐 | `../experiments/experiment_31_loss_amplitude_grokking` | 包含 Power 原版 CE+幅度对齐 grokking 主实验，以及 raw MSE 0/1、0/15 两轮；核心结论是 grokking 不要求唯一的极低 CE，联合输出形态很关键 |
| E32 随机记忆近线性 | `../experiments/experiment_32_random_memory` | 为“复杂度随记忆样本数增长”提供独立容量证据 |
| E33 跨架构联合输出 | `../experiments/experiment_33_cross_architecture` | 证明收缩现象不是单一MLP协议的孤例，同时保留排序/系数边界 |
| E34 条件规则决斗 | `../experiments/experiment_34_conditional_rule_duel` | 独立冻结4-bit复杂度全序预测75次8-bit决斗 |
| E35 Mod97 冻结函数决斗 | `../experiments/experiment_35_mod97_frozen_function_duels` | 用真实grokking前后函数检验完成规则的动态偏好 |
| E37 测度鲁棒性 | `../experiments/experiment_37_measure_robustness` | 把 Gaussian 密度效应与欧式体积分开，并复核多种参考测度下的排序和真实训练对应边界 |
| 关键证据总账和参考文献 | `evidence_ledger_zh.md`、`references_zh.md`及英文版 | 让每个数字、主张和外部理论接口可追溯 |

主包应包含脚本、协议、结果摘要和必要图；不必包含 checkpoint、粒子池或原始 zip。

## 二、补充包：建议上传

| 材料 | 路径 | 上传定位 |
|---|---|---|
| 训练前架构估计 | `../supplementary/architecture_estimator` | 成功浅区和失败深尾都值得公开，不能包装成通用预测器 |
| E36 Peleg/G&C独立验证 | `../experiments/experiment_36_peleg_g_and_c_validation` | 独立验证实验：外部比较和体积假说边界，适合作为补充材料 |
| 严格定义的有限概率核验 | `theory/check_minimal_statistical_language.py`及JSON | 低成本、可复算，建议随理论附录上传 |

补充包可以保留轻量 JSON、CSV和PNG；大型原始结果另建带 SHA256 的下载附件。

## 三、暂不上传

| 材料 | 处理 |
|---|---|
| `kimi_experiments` 输入级/concat输出级实验 | 已有稳定结果（64/64 完成），但经裁决不进公开增量包；原始脚本与结果保留在工作区 |
| 旧版规则决斗面板（物理目录 E31/E36_rule_duel_panel） | 与公开 E34 重复或未结案，保留在工作区；不代表公开 E31/E36 编号 |
| 规则空间 OOD、解耦可达性、早期脚本审核 | 保留在工作区，不进公开增量包 |
| addition8 静态预测 | 保留在工作区，不进公开增量包 |
| AIT/Omega、体积假说独立文献审计 | 保留在工作区，不进公开增量包 |
| `.tmp_*`、`tmp/`、`_analysis_*` | 作为工作区溯源缓存；其中 `_analysis_*`大多早于本次提交，正式结果已被报告吸收 |
| `.venv-gpu`、pip缓存、GPU临时目录 | 环境和缓存，不是研究材料 |
| checkpoint、SMC粒子池、`.pt/.npz/.zip/.jsonl` | 只在有人需要完全复核时按专题单独打包，不进主包 |
| `chatbox_*`、聊天恢复数据库、截图缓存 | 隐私/工程缓存，不上传 |
| MSE失败seed的未解释工程日志 | 只保留结果摘要和失败状态，不上传大量重复日志 |
| 未经独立结果支持的经验公式草稿 | 保留为研究笔记，不写成论文结论 |
| 与本研究无关的RSA和text2image修改 | 留在原工作区，不纳入Neural K发布 |

## 四、推荐上传顺序

1. 先上传主包：短论文、理论核心、E01-E30基础实验和本次主包增量；
2. 再上传补充包：E23、E31--E37以及架构估计报告；公开包只使用 canonical E31--E37 目录，旧物理 E37/E38 目录不上传；
3. 最后根据平台附件限制，单独上传早期脚本审核和大型原始结果；
4. E35和其他未裁决实验等有正式结果后再追加版本，不提前占用主叙事。

## 五、公开时必须保留的限定

- Neural K是协议相对的实现质量码长，不等于机器无关Kolmogorov complexity；
- “复杂函数收缩更快、复杂度分布收窄”是有限协议中的经验趋势和条件机制，不是跨架构定理；
- 静态函数分布、loss截面和SGD最终分布必须分开；
- grokking数据相变、训练时间相变和Neural K不是同一个标量；
- 所有失败、局部换序、采样退化和可达性边界都应与成功结果一起保留。
