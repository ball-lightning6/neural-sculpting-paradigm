# E31：Loss、目标幅度与 grokking

公开实验 E31 的核心不是某一个孤立的 loss 对照，而是一条在同一 Power 架构 mod97 任务上逐步展开的证据链：**原版 grokking + 幅度对齐正则 → 原始 MSE 的目标幅度对照 → 小型网络上的幅度系列与静态复核**。它检验的是 loss 所选择的输出联合分布如何改变规则恢复，而不是把“训练 CE 越低”当成唯一解释。

## 2026-11-06 修订：幅度正则的机制解释

**旧机制解释已修订，实际观察保留。** 幅度正则首先约束跨样本的 margin 分散；多分类形式还包括实际训练标签相对各非目标类的 margin 差异。它既可能提高低 margin，也可能降低高 margin 或缩小整体输出尺度，不指定共同的高幅度。因此不能把它自动等同于普遍增强简单性偏置，也不能从绝对方差下降单独推出相对拟合更均衡或更多噪声记忆。

原 Power 规则恢复、MSE 幅度对照和固定训练集下的静态候选质量变化均保留，不把机制尚未分离改写成效果不存在。历史报告中的早期机制解释以本修订为准；下列原有数值、成功与失败条件及后续历史报告保持不变。

## 核心实验链

1. **Power 原版 mod97 的 CE + 幅度对齐正则**：无正则 CE 基线在训练 CE 约 `4e-11` 时仍未恢复规则；两个幅度对齐条件均为 3/3 完整泛化，并且仍保留拟合后的 grokking 延迟。这是本实验最重要的原版 Transformer 结果，见 `POWER_MOD97_ALIGNMENT_REPORT.md`。
2. **同一 Power 架构的原始 MSE 两轮**：分别使用 0/1 与 0/15 目标幅度，结果为 1/3 与 2/3 完整恢复；它说明固定 one-hot 目标的诊断 CE、实际优化目标和是否发生 grokking 不能混为一谈，见 `POWER_MOD97_MSE_ROUNDS_REPORT.md`。
3. **4-bit copy、Rule30 与静态幅度复核**：用更小的网络拆开目标中心、类间隔、幅度对齐和有限样本效应，见同目录的其他报告。

## 公开报告入口

- `POWER_MOD97_ALIGNMENT_REPORT.md`：原版 Power Transformer 的 CE/grokking 与幅度对齐正则主实验；
- `POWER_MOD97_MSE_ROUNDS_REPORT.md`：Power 架构 raw MSE 的 0/1、0/15 两轮；
- `COPY_4BIT_MSE_AMPLITUDE_SERIES.md`：4-bit copy 的 MSE 幅度系列与幅度对齐正则起点；
- `RULE30_MSE_PM01.md`、`RULE30_CENTERED_MSE_AMPLITUDES.md`：Rule30 的目标中心和幅度对照；
- `MARGIN_ALIGNMENT_STATIC_VERIFICATION.md`：固定训练集下的独立静态分布复核。

失败 seed、暂停状态、单一数据划分和目标幅度边界必须与成功结果一起阅读；这些结果不支持“某种 loss 普遍更好”，而支持更窄的结论：**loss 及其幅度/对齐方向会改变联合输出分布，从而改变规则恢复的可达性与选择压力。**
