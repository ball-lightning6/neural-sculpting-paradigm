# E23 补充：Uniform 自然冲突覆盖分析预注册

## 1. 问题

既有 intervention 已证明：人为提高 MUX3 selector 冲突格 `x1!=x2` 的采样比例，会显著提前 MUX3 的完整函数恢复相变；fixed-D SMC 在没有 optimizer 时复现同方向。

本分析不重新训练，而使用既有 `uniform_random` 数据集的自然随机波动，检验普通随机采样中“是否同时排除两个 copy shortcut”能否预测 MUX3 恢复。

## 2. 数据与独立单位

- 仅使用 `adamw / uniform_random / step=40000` 的既有结果；
- `n=32,40,...,112`；
- 每个 `n` 有128份数据集，每份32个模型 seed；
- 同一 `dataset_index` 在不同 `n` 上是同一随机排列的 nested prefix；
- 统计独立单位是128条 dataset path，bootstrap、置换和交叉验证均按 `dataset_index` 分块。

不得把32个模型 seed 或11个 nested cutoff 当作独立样本扩大显著性。

## 3. 预注册机制变量

目标为：

\[
f_{MUX}(x)=\begin{cases}
x_1,&x_0=1,\\
x_2,&x_0=0.
\end{cases}
\]

定义：

\[
d_1(D)=\#\{x\in D:f_{MUX}(x)\ne x_1\},
\]

\[
d_2(D)=\#\{x\in D:f_{MUX}(x)\ne x_2\}.
\]

其中 `d1` 反驳 `copy x1`，`d2` 反驳 `copy x2`。主变量冻结为：

\[
d_{min}(D)=\min(d_1,d_2).
\]

因为只排除一个 shortcut 仍会留下另一个。次级变量包括：

- `d1+d2`：selector 冲突样本总数；
- `|d1-d2|`：两类反证不平衡；
- 四个 selector-conflict 语义格覆盖数；
- 四格最小计数；
- 八个 `(x0,x1,x2)` 语义格计数标准差与最小计数；
- MUX3 和 parity2 各自的标签不平衡、相关语义格计数标准差。

所有训练集索引必须由原脚本、原 `DATASET_SEED=2026082901` 和 dataset index 精确重建，并核对 CSV 中的 dataset signature。

## 4. 主结果与阴性对照

### 主结果

MUX3 的 `target_function_mass`。它比二元 `dataset_recovered` 保留更多接近相变区的信息。

主统计量是：对每个 `n` 分别计算 `d_min` 与 MUX3 target mass 的 Spearman 相关，再对有效 `n` 等权平均。置信区间按 dataset path 做 block bootstrap；零假设通过对完整 outcome path 做 block permutation 检验。

### 次级结果

- `dataset_recovered`；
- `modal_is_target`；
- `unseen_target_bit_accuracy_mean`；
- `function_collision`；
- `unseen_bit_agreement`。

### 阴性对照

在完全相同的 uniform 输入集上，对 parity2 计算同一个 `d_min` 与各结果的关系。若 `d_min` 对 parity2 同样强，则它可能只是一般覆盖质量代理，而不是 MUX shortcut 判别量。

## 5. 预测模型比较

按 dataset index 分组交叉验证三类模型：

1. `n-only`：只含 `n` 固定效应；
2. `generic`：`n` 加标签不平衡和一般语义格覆盖；
3. `shortcut`：在 generic 上加入 `d_min`、反证不平衡和冲突格覆盖。

对 `target_function_mass` 比较 cross-validated MSE/R²；对 `dataset_recovered` 比较 Brier score 与 log loss。模型只用于测增量解释力，不把回归系数解释成普适因果常数。

## 6. 预注册裁决

### 强支持

同时满足：

1. MUX3 主相关的95% block-bootstrap CI完全大于0；
2. block-permutation `p<0.01`；
3. MUX3 相关显著强于 parity2 阴性对照，差值95% CI大于0；
4. shortcut 模型对 MUX3 的 grouped-CV 指标优于 generic 模型。

### 部分支持

MUX3 主相关稳定为正且CI大于0，但阴性对照差值或模型增量不稳定。这表示自然冲突覆盖有信息，但尚不能证明它特异地通过两个 copy shortcut 起作用。

### 不支持

MUX3 主相关不稳定、CI覆盖0，或加入通用覆盖控制后完全消失。

## 7. 边界

- 这是 uniform 随机采样产生的观察性自然实验；人工 conflict intervention 才承担外生因果判决；
- `d_min` 只显式表示两个已知 copy shortcut，不代表 aggregate competitor denominator；
- 若残差仍大，可能存在其他竞争函数或 optimizer 可达性差异；
- 本分析只补 E23 的局部竞争机制，不检验两种简单性原则或普适 Neural K 定律。
