# 2026-09-20 提交后增量发布树

这是 `4596b2b`（2026-09-02 首次短论文提交）之后的可上传整理版。它是现有 `research/neural_k_framework/` 证据包的增量层，不替代父目录中的 E01-E30、正式短论文、理论核心和网站。

本目录只保留可以让读者复核的材料：理论定义、条件推导、新增实验脚本与结果摘要、外部文献定位、训练前预测边界、规则 OOD 和优化可达性边界。原始 checkpoint、粒子池、SMC 中间快照、临时 smoke 目录、下载压缩包和聊天缓存不放入本发布树，位置和处理方式见 [ARCHIVE_MAP.md](ARCHIVE_MAP.md)。

## 入口

1. [提交后聊天全量审计](../../notes/20260920_post_upload_chat_audit_zh.md)：理论、实验、失败边界、文献和工程工作的完整回顾。
2. [严格统计语言](theory/minimal_strict_statistical_language_zh.md)：当前概率对象、loss截面、函数cell和optimizer分布的定义。
3. [MDL、辨识样本与grokking](theory/mdl_identifiability_grokking_derivation_zh.md)：复杂度、竞争函数分歧谱和数据轴相变的条件桥。
4. [发布清单](MANIFEST.md)：本目录实际包含的材料及其证据等级。
5. [上传裁决表](UPLOAD_DECISION.md)：哪些进入主包、补充包或本地归档。

网站还提供两个面向读者的双语理论入口：严格概率语言、CGF Neural K 估计器及其 520 节点回测。严格概率语言的中文源在 `theory/WEBSITE_PROBABILITY_LANGUAGE_ZH.md`；估计器的 canonical 源在上级 `supplementary/architecture_estimator/WEBSITE_ESTIMATOR_FULL_ZH.md`；英文全文源在上级 `translations_en/` 目录。

`evidence/rule_ood_program_generalization/`保留提交后形成的规则空间 OOD/组合泛化线，`evidence/reachability_scaffold/`保留解耦与优化可达性边界。这两条线不能被静态 Neural K 结果吞并。

## 核心新增主线

### 约束淘汰与可行候选内重排

增加样本会按竞争函数与目标的分歧位置淘汰错误延拓；在仍然可行的函数和连续实现之间，继续降低 loss 又会重新分配质量。若复杂实现的收缩率更高，复杂度分布可以在深尾收窄。这个说法是条件机制和有限协议经验，不是所有架构/loss的普遍定理。

### 联合输出而不是单一 BCE

参数先诱导完整联合输出分布，BCE、raw MSE、目标幅度和对齐正则只是从该分布中选择不同区域。这个修正解释了为什么 0/1 MSE 可以出现 grokking，也解释了为什么幅度对齐能在更高平均 CE 处改变泛化，同时不把不同 loss 的数值直接比较。

### 训练前预测的边界

单层小网络的架构核/作用量估计在部分 3-bit 节点有效；多层深尾和真实加法任务仍有明显失真或只得到部分预测。所有失败和未校准结果都保留，不能把本目录称为通用训练前预测器。

## 发布边界

本目录不声称 Neural K 等于机器无关 Kolmogorov complexity；不声称复杂度与收缩系数跨任意架构、测度和 loss 普遍单调；不声称静态质量分布等于 SGD 最终分布；不声称 grokking 时间点、数据相变点和 Neural K 是同一个量；也不声称 MSE、BCE 和幅度对齐在所有任务上等价。

## 使用方式

上传整个 `research/neural_k_framework/` 时，将本目录作为增量材料一起保留。若只上传增量，至少同时提供父目录的正式短论文、`theory_core_zh.md`、`evidence_ledger_zh.md` 和 `experiments/` 基础目录，因为本目录中的 E15/E23 文件引用了它们的原始实验协议。
