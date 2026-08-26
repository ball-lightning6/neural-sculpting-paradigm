# Parity2 / MUX3 语义覆盖与捷径竞争干预：预注册说明

## 1. 为什么做这个实验

E23 的均匀随机部分训练集显示，parity2 比 MUX3 更早恢复完整目标；后验 agreement 亚网格位置约为 59.98 与 69.47。随后，无界 Gaussian 完整目标 SMC 却显示，在深低 loss 区域，MUX3 的 full-target 参数体积远大于 parity2。

这不必意味着两种测量互相矛盾。完整目标体积问的是“给出全部256个标签后，哪个完整映射拥有更多低-loss实现”；随机训练集相变还包含“有限样本能否排除其他兼容延拓”。两条规则在后一个问题上并不拥有相同的有效数据覆盖：

- parity2 只依赖4个 `(x0,x1)` 格点，同样的样本数下每格平均得到 `n/4` 次观测；
- MUX3 依赖8个 `(x0,x1,x2)` 格点，每格平均只有 `n/8` 次观测；
- `copy x1` 和 `copy x2` 对完整 MUX3 都有75%准确率。只有 selector 冲突样本才足以否定这些捷径。

因此，均匀随机抽取原始8-bit状态并没有把两个目标的语义覆盖和竞争分母归一化。

## 2. 固定项

- 网络：`8 -> 16 x 2 -> 1` tanh MLP，433参数；
- 优化器：AdamW，`lr=0.001`，`weight_decay=0`；
- full-batch raw BCE；
- 每个条件32个配对初始化；
- 每个 `n` 128份配对数据集；
- `n=32,40,...,112`；
- 最大40,000步；
- 两个目标、三个抽样协议共享 dataset id 和初始化 seed；
- 完整函数恢复判据与 E23 相同：训练拟合率不低于0.90、目标函数质量不低于0.90、目标是 modal function、函数 collision 不低于0.80；
- 先进行完整256状态资格检查。

## 3. 三个抽样协议

### 3.1 `uniform_random`

在256个原始输入状态上均匀无放回抽样。每份数据集先生成一个完整随机排列，不同 `n` 使用同一排列的嵌套前缀。

### 3.2 `cell_balanced`

对8个 `(x0,x1,x2)` 格点轮转抽样，每8个样本恰好覆盖各格一次。由于每个 `(x0,x1)` 格点正好包含两个 `x2`，这个协议也自动严格均衡 parity2 的4个相关格点。

该干预删除随机格点计数波动，但保留 parity2 有4格、MUX3有8格的二倍语义单元数差异。

### 3.3 `conflict_enriched`

将 `x1 != x2` 的四个 selector 冲突格点赋予3倍抽样权重，其余四格权重为1。每个完整16样本周期中，冲突样本占75%。该设计仍保持 `x0`、`x1` 及 parity2 四个 `(x0,x1)` 格点的边缘平衡。

两个目标都在该协议下训练。因而 parity2 是负对照：如果变化只是一般性输入分布效应，而不是 MUX 捷径被否定，parity2 也会发生相近移动。

## 4. 冻结预测

### H1：随机覆盖效应

相对于 `uniform_random`，`cell_balanced` 应降低结果方差和格点漏见概率。由于 MUX3 格点更多，预期它从均衡化中获得的收益不小于 parity2。

### H2：捷径竞争效应

相对于 `uniform_random`，`conflict_enriched` 应使 MUX3 的 target accuracy、target function mass 和恢复率更早上升，并使其 `n50/n90` 或 target-aligned agreement 重新凝聚位置左移。这个左移应明显大于 parity2 的对应变化。

raw agreement 在干预初期可以下降：如果原先多个 seed 一致选择同一个错误捷径，打破捷径会先增加函数分叉。只有通过目标准确率守门后的重新凝聚才算支持目标恢复。

### H3：按语义重复次数归一化

除原始样本数 `n` 外，同时报告：

- parity2：`n/4`；
- MUX3：`n/8`。

若 MUX3 需要更大的原始 `n`，但在相变时所需的每格重复次数反而更少，这说明 full-target 体积与原始样本相变的错序主要来自有效格点数量，而不是 MUX3 的完整程序本身更难。

## 5. 主要输出

- 每个协议与目标的 `n50/n90`；
- accuracy>=0.90 守门后的 agreement=0.95/0.99/0.995 重新凝聚位置；
- recovery、target mass、unseen target accuracy、raw agreement 曲线；
- 每份训练集的相关格点最小/最大计数；
- selector 冲突样本比例；
- `copy x1` 与 `copy x2` 在训练集上的经验准确率；
- 训练后 cohort 与 modal hard function 在完整256状态上和 `copy x1/x2` 的相似度，用于捕捉“copy + 少量例外”而非只统计纯 copy 函数；
- dataset bootstrap 区间；
- `conflict_enriched - uniform_random` 与 `cell_balanced - uniform_random` 的配对相变差；
- 以 `n/相关格点数` 表示的相变位置。

## 6. 判决边界

- 若 conflict enrichment 选择性地帮助 MUX3，支持“有效覆盖与捷径竞争改变 fixed-D 分母”的解释；
- 若只有 cell balancing 有效，主要问题是随机格点计数波动，而非 copy 捷径；
- 若两种干预都不选择性帮助 MUX3，则应继续检查其他兼容延拓、网络对无关位的归纳偏置或 optimizer 运输；
- 无论结果如何，该实验都不改变两个完整目标的 full-target 体积。它只检验为什么该静态量不能未经部分数据分母校正，直接等同于原始样本数相变。
