# E15 补充分析：Grokking 前后的 Agreement

## 1. 问题

检验以下直觉：在模型已经记住训练集、但尚未 Grokking 时，不同 seed 在未见输入上的预测是否仍然分散；Grokking 后，Agreement 是否才随目标规则恢复而接近1。

本分析不重新训练模型，只读取既有 E15 `trajectory.csv`。E15 使用 Mod97 除法任务、多档嵌套训练比例和32个 paired seeds，并在训练过程中保存逐 seed 预测及跨 seed Agreement。

## 2. 预先固定的口径

- **主 Agreement**：只在当前训练比例的未见 validation 输入上计算 pairwise seed Agreement。
- **Hard fit**：全部 seed 的训练准确率首次不低于0.999。
- **Grokking位置**：平均未见准确率首次达到0.90，仅作描述性里程碑。
- **准确率基线**：若每个 seed 以概率 $a$ 给出同一个正确标签，错误时在其余96类上独立均匀选择，则

$$
A_{\mathrm{base}}(a)=a^2+\frac{(1-a)^2}{96}.
$$

该基线用于判断 Agreement 的增长是否超出“各 seed 逐渐答对同一个目标”本身。

不能使用包含训练输入的 full-domain Agreement 作主判决。Hard fit 后，所有 seed 在训练输入上已经一致；若训练集覆盖90%的输入，全域 Agreement 即使在未见输入上近乎随机，也会被机械抬到接近0.9。

## 3. 结果

| 训练比例 | Hard fit 时未见 Agreement | Hard fit 时全域 Agreement | 最终未见准确率 | 最终未见 Agreement |
|---:|---:|---:|---:|---:|
| 60% | 0.033 | 0.613 | 0.028 | 0.027 |
| 70% | 0.029 | 0.709 | 0.164 | 0.046 |
| 80% | 0.027 | 0.805 | 0.520 | 0.273 |
| 90% | 0.027 | 0.903 | 0.916 | 0.839 |

97分类的随机 pairwise Agreement 为约0.010。四档数据在 hard fit 时的未见 Agreement 都远离1，只略高于随机基线；全域 Agreement 的高值主要来自已经记住的训练输入。

90%训练条件实际跨过了预设的0.90未见准确率阈值。在越过前后的相邻 checkpoint：

- 未见准确率约从0.900变为0.901；
- 未见 Agreement 约从0.809变为0.811；
- 最终未见准确率约0.916，Agreement约0.839。

Agreement 没有在某个独立时刻突然跳到1，而是随未见准确率共同增长。进入中后期后，观测 Agreement 几乎完全贴合 $A_{\mathrm{base}}(a)$；也就是说，不同 seed 主要共享正确的目标预测，而剩余错误大多是 seed 特异的。

早期训练存在一次短暂 Agreement 峰值，但此时准确率仍接近随机，反映多个 seed 共享某种简单而错误的输出偏置。该峰值随后消退，不能被解释为规则收束。

## 4. 裁决

现有 E15 结果支持用户提出的核心直觉：

> 在训练集已经完全拟合、但尚未发生规则泛化时，未见输入 Agreement 不接近1；随着 Grokking 恢复共同目标规则，Agreement 才向1上升。

需要增加两个限定：

1. Agreement 的上升不是独立于泛化准确率的第二次突变；在当前实验中，它主要由不同 seed 同时恢复正确标签解释。
2. “Grokking 后接近1”要求各 seed 的未见准确率本身接近1。E15 的90%条件最终未完全泛化，因此 Agreement 也停在低于1的位置。

另一个重要结果是，ensemble modal function 可以在单 seed 高准确率和高 Agreement 之前就成为目标函数：不同 seed 已共享目标骨架，但各自仍带有不同残余错误。因此完整函数质量、modal function、未见准确率和 Agreement 仍需分开报告。

## 5. 范围

E15复现的是经典 Mod97 除法任务，但使用 MLP、无 weight decay 和更高训练比例，不是 Power 等人原始两层 Transformer 协议的逐项复刻。它已经直接回答函数系综中的 Agreement 直觉；原版 Transformer 复现可作为确认性外部对照，而不是建立该现象的必要前提。

分析脚本：`analyze_agreement_around_grokking.py`。
