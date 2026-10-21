# E11：实验动机与预注册判据

## 1. 问题

在数据充分的合成任务中，train 与 validation loss 常从训练初期同步下降。一个过强解释是“高 loss 区域已几乎全是正确函数”；更局部、可测的解释是：数据越多，训练梯度越接近总体规则梯度，训练集特有方向在样本平均中越充分抵消。

沿负原始 full-train 梯度的一步小学习率 SGD 更新满足一阶近似：

$$
\Delta L_{val}
\approx
-\eta\nabla L_{val}\cdot\nabla L_{train}.
$$

因此 train/validation 梯度余弦给出沿负原始训练梯度小步更新时 validation loss 变化的一阶预测。实际 Adam 的动量和自适应预条件会改变更新方向，不能把该余弦直接解释为实际 Adam 步的 validation loss 变化；后者需另测。

## 2. 正式配置

- 任务：30-bit 循环边界 Rule30 单步更新，输出30 bit；
- 网络：`30 -> 1024 x 3 -> 30` GELU+LayerNorm；
- 训练样本：256、512、768、1024、1280、1536、2048、4096，严格嵌套；
- 独立 validation：4,096；
- 5个 paired model seeds，共40个模型；
- Adam，学习率`1e-3`，batch 512；
- dropout 与 weight decay 均为0；
- 最大20,000步。

## 3. 三个梯度指标

- `train_validation_gradient_cosine`：训练与独立 validation 完整梯度余弦；
- `train_half_gradient_cosine`：训练集两半的梯度相干性；
- `validation_half_gradient_cosine`：validation 两半的一致性参考线。

第三项只是有限 validation 采样噪声的参考，不是严格数学上限。

## 4. 关键控制：matched raw BCE

不同样本量到达同一 step 时 loss 不同，不能直接比较。冻结实现对每个样本量和 seed，在已保存的诊断 checkpoint 中按`log(raw BCE)`距离选择最接近下列目标 loss 的点，而不做插值：

```text
0.68, 0.6, 0.5, 0.4, 0.3, 0.2,
0.1, 0.05, 0.02, 0.01, 0.003
```

仅保留`max(actual_loss / target_loss, target_loss / actual_loss) <= 1.35`的匹配点，再逐层比较梯度余弦和 validation-train BCE gap。这是带容差的近邻比较，不是精确同 loss 比较；归档`fixed_raw_bce_comparison.csv`保留每个点的实际 loss 和实际/目标比值。

## 5. 预注册预测与边界

若数据量重塑的不只是终点，则在相同 raw BCE 下，训练样本越多：

1. train/validation 梯度余弦越高；
2. 泛化 gap 越小；
3. 该顺序应跨 seed 保留；
4. 低数据条件可在深 loss 转为负余弦，高数据仍保持正向。

该实验只使用一个嵌套数据排列和一个任务，不建立跨任务普适定律；梯度范数接近数值零时余弦失去意义，主分析采用不低于0.003的目标 BCE 层级，实际 checkpoint loss 受上述容差约束。
