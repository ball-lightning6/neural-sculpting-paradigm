# E25：实验动机与冻结协议

## 1. 从规则表到真实输入

对未见样本`x`，框架的直接预测不是先训练一个额外分类器，而是在同一个`L_D<=epsilon`静态 parent 中比较：

$$
P_{hard}(y\mid x,D,\epsilon)
=\frac{\mu\{L_D\le\epsilon,h_\theta(x)=y\}}
{\mu\{L_D\le\epsilon\}}.
$$

soft 版本在 parent 内加入 Bernoulli likelihood；`-log P_soft(y)`是单样本自由能增量。两者都选择剩余参数质量更大的标签。

## 2. Stage 0校准

- 任务：MNIST `0 vs 1`和`3 vs 8`；
- 输入：28x28 平均池化到7x7 并映射到[-1,1]；
- 网络：`49 -> 32 -> 1 tanh`，1,633参数；
- n=4到512，四份数据集 x 八 seed，15,000步；
- 测量 train/validation BCE、hard accuracy、agreement 与 U 形转折。

Stage 0允许使用 validation 选择两个对照任务、最小 n 和要覆盖的 loss 网格，因此后续是“校准后冻结”的确认，不是完全盲的新任务预测。

## 3. Stage 1冻结

冻结`0/1,n=4,dataset0`与`3/8,n=4,dataset0`，以及九个 train raw-BCE 截面`0.6...4e-5`。每个条件使用6副本 x4,096粒子的 Gaussian-pCN SMC。每个截面的 validation/test 预测先写入 unscored 文件并计算 SHA256，再读取标签评分。

主要判决：

1. 静态分支是否显著高于随机预测真实 MNIST；
2. 静态 soft NLL 是否自行形成 U 形；
3. 其最低区间是否与 matched-loss SGD validation 转折对齐；
4. 两任务的低-loss 体积差是否与最小充分样本量同向。
