# E14：实验动机与预注册判据

## 1. 构造

输入含一个 rule bit。rule bit 0对应 Rule110，rule bit 1对应 Rule30。训练目标按：

```text
Rule110 : Rule30 = 1, 10, 100, 1000, 10000
```

加权。少数 Rule30 样本绝对数量仍足以表达规则，但对高 weighted loss 的贡献可被多数分支暂时忽略。

## 2. 竞争预测

- 若固定 hard 函数概率对 loss 全程单调，则`Rule110-both`不应先增后减；
- 若不同 loss 深度激活不同约束，训练应先优化多数分支，随后逐位修正三个冲突 neighborhood，最终形成完整复合映射；
- 权重比若只平移有效 loss 尺度，则用少数权重`epsilon`归一化后，各比例迁移应塌缩。

## 3. 配置

- `4 -> 64 x 3 -> 1` GELU+LayerNorm；
- plain full-batch SGD，学习率0.05，无 momentum 和 weight decay；
- 每个比例512个 paired seeds，共2,560个模型；
- matched weighted raw-BCE first crossing；
- 最长运行125,204步。

## 4. 边界

该实验有意通过样本权重制造约束接管，证明“可以非单调”，不主张所有自然数据集都出现同一路线。完整复合函数在足够低 loss 处还受到 hard 约束强制，不能单独解释成 MDL 终点证据。
