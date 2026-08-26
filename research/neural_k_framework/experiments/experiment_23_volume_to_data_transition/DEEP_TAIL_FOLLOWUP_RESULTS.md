# E23 Gaussian 深尾后续：结果与裁决

## 1. 运行状态

- 协议：8-bit、`8 -> 16 x 2 -> 1 tanh`、433参数；
- 参考测度：归一化参数独立 Gaussian，`sigma=1/sqrt(3)`；
- 8个replica，每副本8,192粒子；
- 五目标共享parent后锁步推进；
- 状态：人工中断；70个请求阈值中至少62个共同索引完成；
- 最深可比较共同阈值：`epsilon=0.00046351354785`；
- 全部分支在该深度的 hard-exact fraction 均为1；
- proposal acceptance稳定在约0.30。

该实验是与原Uniform cube同方差的Gaussian深尾扩展，不是原E23 bounded profile的数值续段。

## 2. Random 的两组浅层错序得到强支持

在最深共同阈值：

| 比值 | 中位log比 | replica范围 | 相对速度中位数 | 判决 |
|---|---:|---:|---:|---|
| `log V(parity3)/V(random)` | 5513.72 | [5186.25,5987.52] | +444.35 | 8/8支持 |
| `log V(parity4)/V(random)` | 3203.36 | [2940.75,3705.96] | +161.40 | 8/8支持 |

两对均满足预注册的连续深窗体积比与速度方向，`pair_robust=true`。Random-balanced 在Gaussian深尾中的体积远小于 parity3/4，与其 `n50/n90>240` 的右删失操作性复杂度一致。

## 3. Parity2/MUX3 没有在深尾交叉

预注册期望为 `V(parity2)>V(MUX3)`，因为操作性测量中 parity2 的 `n50/n90=64/80`，MUX3为`80/112`。实际体积比轨迹为：

1. `epsilon=0.700`时比值仅略正，median `+0.235`；
2. 到0.621迅速转负为`-56.37`；
3. 在0.488达到`-97.53`；
4. 随后回追，在约0.05达到局部最高`-69.17`，仍未接近零；
5. 0.05以下相对速度再次转负，体积差重新扩大；
6. 最深共同阈值0.0004635时，median为`-96.918`，8/8 replica范围`[-109.600,-94.677]`全部为负；
7. 最后窗口比值增长率median为`-3.446`，8/8 replica均为负。

因此当前深尾下：

$$
\frac{V(\mathrm{parity2})}{V(\mathrm{MUX3})}
\approx e^{-96.9}
\approx 10^{-42.1}.
$$

MUX3 的 full-target Gaussian 参数体积仍显著大于 parity2，而且最深处没有继续追向交叉。该对`pair_robust=false`，不是因为接近零，而是因为符号稳定地与预注册操作性顺序相反。

## 4. 数值边界

- 运行在计划结束前人工中断，不能证明无限低loss极限永不交叉；
- parity2/MUX3 的最深8副本全部同号，离零约95个自然对数单位以上，继续在剩余8个请求阈值内反超缺乏轨迹支持；
- parity4与random的独立replica绝对log-volume范围很宽，反映极深尾归一化不确定性；但两者比值在全部副本中远离零，符号裁决不受影响；
- 深尾lineage祖先数很低，MCMC rejuvenation acceptance仍稳定约0.30。结果适合做比值符号和数量级判决，不应把单个绝对volume小数位当成高精度真值。

## 5. 理论裁决

该实验支持较弱而更准确的结论：

1. full-target体积对操作性样本相变具有强一阶预测力；
2. random与parity3/4的浅层错序可由深尾profile换序解释；
3. 但 parity2/MUX3 给出一个稳定的跨族例外：Gaussian full-target静态体积排序与AdamW随机训练集相变/连续agreement排序相反；
4. 因此不能把 full-target Neural K-profile 与 `n50/n90` 定义成普遍等价的同一个复杂度标量；
5. fixed-D候选竞争、样本覆盖与optimizer运输具有独立作用。

下一判决应对 parity2/MUX3 使用密集n网格、更多随机训练集，并跨 AdamW、plain SGD、momentum SGD 比较恢复率和agreement。若操作性顺序跨optimizer稳定，则该例外主要属于静态full-target体积与partial-data训练之间的差距；若顺序随optimizer变化，则动力学贡献可以被直接定位。
