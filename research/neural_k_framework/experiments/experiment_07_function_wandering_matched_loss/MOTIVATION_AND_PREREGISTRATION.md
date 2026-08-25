# E07：实验动机与预注册判据

## 1. 问题

E06 已经证明静态低-loss 质量会重排完整函数，但四样本条件也证明静态退火不能替代 SGD。E07 因此把函数分布粗粒化为可精确追踪的 function ID，直接测量：

- 单个模型在 hard fit 后是否继续跨函数 cell；
- 迁移是随机跳跃还是集中到少数通道；
- 多 seed 分布是否趋于稳定；
- 静态先验在同等 loss 下是否给出相同分布。

## 2. 函数迁移实验

- 任务：`4-bit -> 1-bit`，完整16点可穷举；
- 训练集：6个固定平衡样本，留下10个未见输入和1,024个 hard-compatible 函数；
- 网络：`4 -> 16 x 3 -> 1 tanh`；
- 模型：1,024个独立 seed；
- 优化：full-batch Adam，学习率0.003，weight decay 为0；
- 对齐：按每条轨迹首次达到训练集100% hard accuracy 对齐；
- 观察：post-fit 20,000步，密集保存 function ID 与部分 logits。

训练标签事后发现恰好满足`y=x1`。这不是预注册目标，因此相关规则解释只作为结果审计，不能伪装成实验设计时已知。

## 3. Matched-loss 静态对照

第二个脚本在完全相同的网络和训练集下采样4,194,304个未训练网络，先筛选 hard-exact 权重，再用两种方式与 SGD 年龄对照：

1. raw BCE 子水平集或`exp(-beta L)`重加权；
2. RMS-normalized BCE 退火，在静态和 SGD 平均 normalized loss 近似相同时比较 function ID 分布。

## 4. 判据

- function-ID 转换次数、Hamming 距离、转移边与停留时间；
- 多 seed 函数熵、top-state 质量与 agreement；
- 静态/SGD 分布的 JS divergence；
- ID 666与 ID 722的静态和动态赔率；
- 相同 normalized loss 是否产生相同函数分布。

## 5. 边界

- 宽度16是高密度轨迹 pilot，不承担跨宽度普遍性；
- 极低 loss 接近 Adam `eps=1e-8`和 float32 数值尺度，晚期再扩张谨慎解释；
- `wandering_overview.png`右下累计转换子图错误使用最终总数，禁止引用；
- 原结果包自动生成的`normalized_loss_match_*`误拿 raw SGD loss 作匹配目标；正文 normalized 对照必须使用保存 logits 后的独立复算；
- hard function ID 相同不表示连续 logits、margin 或内部实现相同。
