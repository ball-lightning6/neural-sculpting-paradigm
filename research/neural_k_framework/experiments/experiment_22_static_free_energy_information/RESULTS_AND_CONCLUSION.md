# E22：完整结果与阶段裁决

## 1. 数值闭合

2,097,152个 prior 样本、6,561个部分状态、256条规则和每条40,320个顺序全部完成。

```text
max path-invariance error       = 2.84e-14 bit
max stage-decomposition error   = 1.42e-14 bit
max Shapley-efficiency error    = 5.68e-14 bit
hard predictive normalization  = 1.89e-15
beta=1 normalization error      = 6.10e-08
```

因此，逐样本广义惊讶度确实是同一静态势函数的差；不同顺序重新分配每一步成本，但总成本严格等于完整规则端点自由能。

![代表规则的静态自由能成本](../../assets/figures/e22_free_energy.png)

## 2. Rule150 仍是困难端点

Rule150 即3-bit parity。在 hard-conditioned 口径下，其完整规则成本约29.0000007 bit，在256条规则中排名最难；在`beta=1` Gibbs 口径下成本约8.0625 bit，排名第二。这复现了此前逐样本启发式中 parity 困难的方向，但现在总量不再依赖加入顺序。

## 3. 理论意义

E22 建立了三者的严格关系：

1. 当前训练集上的预测分支质量；
2. 新样本的广义惊讶度；
3. 完整规则的端点自由能或码长。

它为 [Blier 与 Ollivier（2018）的 prequential coding](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3b712de48137572f3849aabd5666a4e3-Abstract.html)、Bayesian evidence 和 [Levin、Tishby、Solla（1989）的统计力学配分函数](https://mlanthology.org/colt/1989/levin1989colt-statistical/)提供了共同语言，也说明 agreement 与 surprise 可以被理解为函数系综凝聚和自由能增量的两个侧面。

## 4. 边界

- 该不变量属于静态参考测度，不意味着 SGD 是 Gibbs 采样器；
- 每一步信息量仍依赖加入顺序，只有端点总和与 Shapley 平均具有相应不变性；
- 端点难度依赖网络、参数化、输入编码、loss 和参考测度，不是机器无关 K 复杂度；
- 3-bit 穷举确认数学闭环，但不能单独证明大网络上的定量外推。
