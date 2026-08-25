# E01a/E01b：实验动机与预注册判据

## 1. 为什么做

早期百余合成任务中，训练数据只覆盖巨大输入空间的极小部分，网络却常跨 seed 稳定收敛到生成规则。[Dingle 等（2018）](https://www.nature.com/articles/s41467-018-03101-6)、[Valle-Pérez 等（2019）](https://arxiv.org/abs/1805.08522)与 [Mingard 等（2021）](https://www.jmlr.org/papers/v22/20-676.html)提供了最接近的解释：网络架构在初始化时已经诱导非均匀函数先验，训练主要删除不符合标签的函数，因此

$$
P_{\mathrm{trained}}(f\mid D)
\approx
\frac{P_0(f)\mathbf1[f\models D]}
{\sum_gP_0(g)\mathbf1[g\models D]}.
$$

严重欠约束却稳定泛化迫使这一图景承担一个很强的定量预言：所有偏离生成规则、但仍满足有限训练集的候选函数，其**总条件先验质量**必须远小于目标规则。这个要求虽然反直觉，却不能仅凭直觉排除，因为单个函数的指数小概率可能被指数多候选数量抵消。必须在可枚举空间直接测量。

## 2. 竞争假说

### H-static：硬条件化

1. 训练分布应在有限采样误差内接近 hard-conditioned initialization prior；
2. 训练集不变后，post-fit 年龄不应系统改变 hard-function 赔率；
3. 静态后验只依赖最终训练集，样本加入顺序不应留下持久差异；
4. 已经满足训练标签的初始化函数不需要再发生系统迁移。

### H-transport：优化诱导运输

1. 训练函数分布可以显著超出条件先验抽样噪声；
2. post-fit 继续优化仍可改变具体函数赔率；
3. 同一最终训练集的不同训练历史可以产生不同函数；
4. 即使初始化函数已满足全部训练标签，优化仍可把它运输到另一兼容函数。

H-transport 不预言迁移一定朝人类定义的最低复杂度函数，也不否定初始化先验提供零阶偏置。

## 3. E01a 判决设计

- 完整函数空间：`3-bit -> 1-bit`，共256个 hard function；
- 网络：`3 -> 64 x 10 -> 1 tanh`，`sigma_w=1`、`sigma_b=0.2`；
- prior：1,048,576个初始化网络；
- 训练：每条件8,192个初始化，full-batch Adam，学习率`1e-3`；
- 条件：`000->0/1`及 Rule30 的`k=2/3/4`嵌套样本；
- 时间：首次拟合、post-fit 100/1,000步、margin 2/4/8；
- 路径：direct、forward、reverse，各4,096个配对初始化；
- 零分布：从同一静态后验重复采样并 bootstrap 1,000次。

主判据：逐函数 TV/JS 必须显著超过 prior 与训练样本量产生的抽样波动。复杂度直方图、平均准确率或 LZ 不作为核心判据。

## 4. E01b 判决设计

- 网络：`3 -> 1024 x 3 -> 1`，GELU+LayerNorm、Adam、无 weight decay；
- prior：4,096个初始化网络；
- 条件：`000/111 -> 0/1`四种单样本；
- 每条件128个配对初始化；
- 快照：初始化、首次拟合、post-fit 1/2步；
- 公平对照：单独统计初始化时已经满足该训练标签、但完整函数非常数的 seed。

主判据：若目标常数在 hard-conditioned prior 中并不占压倒性质量，而训练仍使全部 seed 坍缩到目标常数，且 prior-compatible 非常数 seed 也全部迁移，则硬条件化抽样模型被直接拒绝。

## 5. 结果前明确保留的边界

- E01a 实际使用 Adam，不能把结果自动推广到所有 SGD；
- hard function 相同不等于 logits、margin 和表示相同；
- 有限时间路径依赖不等于无限时间永不汇合；
- 结果只能反驳 hard-function prior 的被动筛选强版本，不反驳所有连续 Bayesian 或含 optimizer 动力学的概率描述。
