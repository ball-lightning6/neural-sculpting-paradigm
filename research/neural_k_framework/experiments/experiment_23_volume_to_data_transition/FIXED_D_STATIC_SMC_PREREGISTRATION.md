# Parity2 / MUX3 fixed-D 静态条件分布 SMC：预注册说明

## 1. 缺失的逻辑桥

完整目标体积与 grokking 数据相变不是同一个量：

- full-target volume 给出完整256标签全部固定时，目标函数在某个 loss 深度拥有多少参数质量；
- 固定部分训练集的目标赔率还取决于所有兼容延拓组成的分母；
- `n50/n90` 又进一步依赖训练集抽样分布、恢复阈值和 optimizer 可达性。

E23 对 parity 家族得到严格前瞻排序，并在全部规则上得到很强相关，但这仍是经验桥，不是从 full-target Neural K-profile 到 grokking 相变点的定理。MUX3/parity2 的跨族例外表明，中间的 fixed-D 分母必须直接测量。

## 2. 触发实验的观察

同一 `8 -> 16 x 2 -> 1 tanh` 网络下，Gaussian full-target 深尾体积强烈偏向 MUX3；均匀随机部分训练集却更早恢复 parity2。数据干预进一步发现，在 `n=32--48` 时富集 selector 冲突样本会使 MUX3 的真实目标准确率、目标函数质量和恢复率大幅上升，而不只是提高 raw agreement。

本实验检验这一巨大差异是否已经存在于静态 loss-conditioned 参数分布中。

## 3. 固定协议

- 网络与参数化：E23 的 `8 -> 16 x 2 -> 1 tanh`，433个归一化坐标；
- 参考测度：各坐标独立 `N(0,1/3)`，与 Gaussian full-target 深尾实验一致；
- 不使用梯度，不运行 SGD/AdamW；
- constrained SMC：4个 replica，每个条件每副本2,048粒子；
- 每种抽样协议8份独立数据集；
- 两个目标、三个协议，共48个 fixed-D 条件；
- 所有条件共享同一批初始粒子，随后独立重采样和 pCN 变异；
- raw BCE 阈值：`0.68,0.60,0.50,0.40,0.30,0.20,0.10,0.05,0.03,0.02`；
- 训练样本数固定为 `n=32`。

## 4. 模式权重怎样实现

权重通过不同语义格点中实际抽到的、nuisance bits 不同的原始输入数量实现，而不是重复同一个原型点。

### `uniform_random`

从全部256个8-bit状态均匀无放回抽32个。八个 `(x0,x1,x2)` 格点的计数服从对应的多元超几何波动，期望各为4。

### `cell_balanced`

八格严格各抽4个。它消除格点计数波动，但 MUX3 的 `copy x1/x2` 仍各能解释75%的模式。

### `conflict_enriched`

`x1!=x2` 的四个 selector 冲突格点严格各抽6个，`x1=x2` 的四个普通格点严格各抽2个。冲突总权重为75%，使 `copy x1/x2` 的经验准确率降至62.5%。同时 parity2 的四个 `(x0,x1)` 格点仍严格各8个。

每个格点内部从其余5 bit形成的32个不同原始状态中无放回抽样。

## 5. 冻结判决

在匹配 raw BCE 阈值下比较：

- 未见输入目标准确率；
- 未见输入 agreement；
- 完整目标 hard function 的条件质量；
- 完整函数 collision、modal function 和 top functions；
- 对 `copy x1`、`copy x2`、`x0 XNOR x2` 的函数相似度；
- 每个未见输入的后验标签概率；
- 条件事件的 log-volume 与 replica/lineage 诊断。

主要预测是：若 fixed-D 静态竞争分母是第一阶原因，则 MUX3 应在足够低 loss 下呈现 `conflict_enriched > cell_balanced > uniform_random` 的目标倾向，且这一选择性提升不应在 parity2 上同向复现。

若静态 SMC 中三种协议接近，而 AdamW 中仍强烈分离，则 optimizer 运输是不可缺少的主要解释。

## 6. 理论边界

无论本实验结果如何，`n50/n90` 都不能直接定义为 full-target Neural K 的同义词。更准确的层次是：

1. full-target Neural K-profile：完整函数的协议相对精度成本；
2. fixed-D 条件函数分布：目标与兼容延拓的相对赔率；
3. 数据辨识复杂度：随机训练集使目标赔率跨过阈值所需的样本量；
4. optimizer 恢复相变：真实训练运输使目标函数质量跨过阈值所需的样本量。

E23 已建立这些层次之间的强经验联系，但普遍的定量映射仍是开放问题。

