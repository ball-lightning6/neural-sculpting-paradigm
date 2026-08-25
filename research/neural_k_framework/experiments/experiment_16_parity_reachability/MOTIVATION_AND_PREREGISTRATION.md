# E16：实验动机与预注册判据

## 1. 初始歧义

高维 parity 从随机初始化常停在`ln2`平台。这可能意味着：

- 网络无法表示精确 parity；
- 低-loss 区域不偏好 parity；
- 数据缺口使另一延拓更受偏好；
- 目标存在且局部稳定，但普通梯度看不到全局入口。

Leave-one-out 又会刻意打破输入空间对称性，不能把最后一个点预测错误直接解释成目标体积不足。

## 2. 四层判决

1. **维数扫描**：`n=4...16`，训练完整真值表减一个分层选择的 holdout；
2. **随机半空间**：14-bit 随机、标签平衡的50%训练/50%测试 mask；
3. **错误揭示**：从相同父权重复制 continue、随机正确点、当前错误点、错误 replay 和完整揭示五分支；
4. **Scaffold 恢复**：16-bit 网络先预测所有 prefix parity，迁移最终 head 后撤去全部中间监督，再从精确 anchor 施加0--0.2相对 L2 扰动并仅用 endpoint BCE 恢复。

## 3. 共同参考机

GELU+LayerNorm 三隐藏层、AdamW `1e-3`、weight decay 为0。leave-one-out 与 half-space 主网络宽64；最终 scaffold 判决宽256。完整输入空间均可穷举。

## 4. 边界

- parity 在人类逻辑语言中短，不等于在该网络梯度语言中易达；
- holdout 位置和 mask 频谱会改变优化可见性；
- 高测试 accuracy 不等于 exact full function；
- 辅助监督改变路径，成功后必须撤除并以 endpoint-only 稳定/恢复作最终判据。
