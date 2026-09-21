# E23 补充：Uniform 冗余位假 Selector 分析预注册

## 1. 新假说

MUX3 的目标为：

\[
f(x)=\operatorname{MUX}(x_0,x_1,x_2).
\]

在 `x1=x2` 的一半输入上，selector 完全不可辨识。在 `x1!=x2` 的另一半输入上，任何与 `x0` 独立的 selector 都有1/2概率选对。因此，用任意冗余 bit `xj` 替代真实 selector 可构造：

\[
g_j(x)=\operatorname{MUX}(x_j,x_1,x_2),
\]

其总体期望准确率为：

\[
\Pr[g_j=f]=\frac12+\frac12\cdot\frac12=\frac34.
\]

同理，`NOT xj` 也形成75%总体准确率的假 selector。MUX3 在8-bit输入下有5个冗余位，因此除 `copy x1/x2` 外，还存在10个极简单的 nuisance-literal selector。假说是：同样 `n` 和 copy-shortcut 反证量下，训练集中偶然拟合更好的 nuisance selector 会继续压低 MUX3 的完整目标质量。

Parity2 的冗余位替换只有50%总体准确率，因此同一 nuisance-selector 统计量不应强烈预测 parity2 恢复。

## 2. 数据

- 使用既有 `adamw / uniform_random / step=40000` 结果；
- 11个 nested `n=32...112` cutoff；
- 128条独立 dataset path；
- 每份数据集32个模型 seed；
- 不重新训练，不运行 SMC。

训练集索引由原 `DATASET_SEED=2026082901` 精确重建，并核对原 dataset signature。

## 3. 预注册 Selector 族

### 主族：单冗余 Literal

\[
\mathcal S_{lit}=\{x_3,\neg x_3,\ldots,x_7,\neg x_7\}.
\]

只在 `x1!=x2` 的 conflict 样本上比较 `s(x)` 与真实 `x0`。冻结以下主变量：

- `literal_best_conflict_errors`：10个 literal 中最少错误数；
- `literal_advantage_over_copy`：`copy x1/x2` 最少错误数减去 literal/copy 联合最少错误数；
- `literal_outperform_copy_count`：严格优于最佳 copy shortcut 的 literal 数量。

`literal_advantage_over_copy` 越大，表示至少一个冗余位 selector 比两个 copy shortcut 更符合当前训练集。

### 次级族：冗余位 GF(2) Affine Selector

枚举所有非恒定：

\[
s(z)=b\oplus\bigoplus_{j\in J}x_j,
\qquad J\subseteq\{3,\ldots,7\},\ J\ne\varnothing.
\]

该族用于诊断更宽的 nuisance competition，但 XOR 对 tanh MLP 未必便宜，因此不承担主判决。

## 4. 主判据

对 MUX3，将 `literal_advantage_over_copy` 与 `target_function_mass` 做 partial Spearman：

- 控制 `n` 固定效应；
- 控制标签不平衡；
- 控制八格计数标准差/最小计数；
- 控制既有 `copy_refutation_min`、双侧不平衡和 conflict 四格覆盖。

独立单位仍为 dataset path。95% CI 使用 path-level block bootstrap；单侧零假设使用 outcome-path block permutation。

预期方向：

\[
\rho_{MUX}<0.
\]

Parity2 使用完全相同输入集和同一 nuisance 统计量作阴性对照，预期其 partial correlation 接近0。

## 5. Grouped cross-validation

按 dataset index 分5折比较：

1. `copy_baseline`：`n`、一般覆盖和原 copy shortcut 变量；
2. `literal_nuisance`：在 baseline 上加入 nuisance literal 指标；
3. `affine_nuisance`：进一步加入 affine selector 指标。

对 target mass 比较 CV MSE/R²，对 dataset recovery 比较 Brier/log loss。主增量是 `literal_nuisance` 相对 `copy_baseline`；affine 仅作次级诊断。

## 6. 裁决

### 强支持

1. MUX3 partial correlation 的95% CI完全小于0且单侧 `p<0.01`；
2. MUX3−parity2 的相关差值95% CI完全小于0；
3. literal 模型同时改善 MUX3 target-mass MSE 与 recovery Brier；
4. 对 parity2 不出现同方向增益。

### 部分支持

MUX3 partial correlation 或 grouped-CV 增量只有一项稳定，或 parity2 阴性对照不能完全排除一般覆盖解释。

### 不支持

控制 copy shortcut 和一般覆盖后，nuisance literal 不再携带独立信息。

## 7. 边界

- 分析只检验8-bit下是否已有 nuisance-selector 痕迹，不直接证明总输入维数增加时劣势必然扩大；
- 最优候选统计量存在 multiple-candidate selection，但候选族和判据已在看结果前冻结；
- nuisance selector 仍只是已知竞争族的一部分；
- 若结果支持，输入维数扫描才值得作为后续独立实验。
