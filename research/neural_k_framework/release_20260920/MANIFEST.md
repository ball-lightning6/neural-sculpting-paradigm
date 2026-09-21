# 增量发布清单

版本：`2026-09-20-post-upload-v1`。基线：Git `4596b2b`，2026-09-02。

## 1. 理论层

`theory/`包含：最小严格统计语言及逐项审计、网站用严格概率语言入口、MDL最低可行复杂度/竞争质量/分歧谱/grokking数据相变、周末理论增量，以及 Gaussian/欧式测度鲁棒性与真实训练对应的复核。架构估计器的完整网站用方法稿和 520 节点回测在 `supplementary/architecture_estimator/` 中；测度复核脚本与轻量摘要在公开 E37 目录中。

最重要的新增主线是：约束淘汰负责删除不再可行的捷径；联合输出的连续 loss 截面负责在剩余候选中重排质量。复杂函数收缩更快时，复杂度分布才会在有限深尾窗口内收窄。

## 2. 实验层

| 目录 | 内容 | 证据等级 |
|---|---|---|
| `E15_grokking_agreement` | 关联既有 E15/E27：Mod97 hard-fit、Agreement与未见准确率 | 已完成分析增量 |
| `E23_volume_to_data_transition` | 关联既有 E23：Parity2/MUX3自然覆盖、selector和fixed-D竞争补充 | 预注册分析闭环 |
| `../experiments/experiment_31_loss_amplitude_grokking` | **公开 E31**：Power 原版 CE+幅度对齐 grokking、raw-MSE 0/1 与 0/15、Rule30/4-bit 幅度目标 | 直接训练证据，保留 seed 边界 |
| `../experiments/experiment_32_random_memory` | **公开 E32**：8-bit随机标签记忆与近线性复杂度响应；报告还明确把它作为 Neural K 的外部构造效度校准 | 多组结果，非唯一容量定理 |
| `../experiments/experiment_33_cross_architecture` | **公开 E33**：跨架构联合输出体积与3-bit排序 | 有限协议证据，非通用公式 |
| `../experiments/experiment_34_conditional_rule_duel` | **公开 E34**：8-bit条件规则决斗，冻结复杂度全序预测全部75次判决 | 事前预测，公开目录与网站编号一致 |
| `../experiments/experiment_35_mod97_frozen_function_duels` | **公开 E35**：mod97真实grokking候选的冻结函数决斗 | 动态偏好证据，公开目录与网站编号一致 |
| `../experiments/experiment_36_peleg_g_and_c_validation` | **公开 E36**：Peleg/G&C独立验证 | 独立外部验证实验 |
| `../experiments/experiment_37_measure_robustness` | **公开 E37**：Gaussian、固定范数球欧式测度、端点响应与真实训练对应 | 测度鲁棒性与参考测度边界实验 |

## 3. 内部保留材料

发布树中仍可能保留用户明确不上传的研究目录，作为本地溯源和后续追加的工作区材料；Git/Zenodo公开包不应纳入它们。包括规则空间 OOD、reachability scaffold、early replay、addition8、旧物理目录 E31/E36 和独立文献审计。这里的物理目录名不代表网站公开编号。

## 4. 证据层

- `../supplementary/architecture_estimator`：架构核/作用量估计、深尾失败和多层失败边界。
- `../experiments/experiment_36_peleg_g_and_c_validation`：**公开 E36**，Guess & Check、低loss静态样本和SGD对照的独立验证。
- `evidence/engineering`：本地 GPU 检查和运行辅助脚本，不是科学结果。

## 5. 文献层

`literature/`属于内部审计材料，不进入本次公开增量包。公开包只保留主文、理论核心和参考文献入口。

## 6. 文件政策

- 允许：`.md`、`.py`、`.json`、`.csv`、必要的 `.png` 和有限核验输出；
- 排除：`.pt`、`.pth`、`.npz`、`.zip`、`.jsonl`、缓存目录、临时 smoke 结果和本地聊天数据库；
- 结果摘要必须同时报告协议、seed、硬件、loss口径、采样/混合边界和失败状态；
- 不把“脚本存在”当作“实验已经复现”。
