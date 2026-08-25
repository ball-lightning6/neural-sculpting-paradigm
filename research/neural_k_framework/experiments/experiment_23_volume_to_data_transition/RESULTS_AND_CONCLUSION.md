# E23：完整结果与阶段裁决

## 1. 确认性 parity 预测严格命中

完整数据上八条规则均可100%拟合并恢复目标 hard function。冻结的 parity 顺序与独立测得的样本相变为：

| 规则 | volume score | `n50` | `n90` |
|---|---:|---:|---:|
| parity1 | 125.21 | 24 | 48 |
| parity2 | 3266.21 | 64 | 80 |
| parity3 | 4618.60 | 96 | 112 |
| parity4 | 6719.58 | 160 | 160 |

四个`n50`和`n90`网格区间均互不重叠；Spearman 为1.0/1.0。该顺序从5,000步起已经稳定，不是40,000步截止点偶然产生。由此获得从完整目标静态体积到随机训练集辨识样本量的直接前瞻证据。

![完整目标体积与数据相变](../../assets/figures/e23_volume_to_transition.png)

## 2. 跨族关系很强，但单标量定律失败

全部八规则上，volume score 对`n50/n90`的 Spearman 为0.898/0.868；去掉随机规则后七条结构规则为0.955/0.927。但出现三组明确反转：MUX3/parity2、random/parity3、random/parity4。random 到`n=240`仍未恢复完整映射。

因此浅层单区间斜率是强一阶预测量，却不是跨任意函数族的充分统计量。固定训练集中的竞争分母、样本覆盖结构和 optimizer 可达性仍可贡献差异。

## 3. 复杂度排序本身随 loss 改变

random-balanced 的局部收缩率随 loss 深入急剧加速，并依次超过 parity2、parity3 和 parity4。绝对体积排名也发生交叉。用更深局部窗口预测`n50/n90`时，Spearman 从约0.73/0.71上升到0.97/0.95；用单截面`-log V`预测时也从约0.83/0.77升到0.95/0.93。

这迫使复杂度对象从单个数修正为完整 profile：

$$
K_f^N(\epsilon)=-\log V_f(\epsilon),\quad
s=-\log\epsilon,\quad
\kappa_f(s)=\frac{dK_f^N}{ds}.
$$

局部斜率、曲率和交叉都携带信息；任一浅层切片都可能遗漏深尾成本。

## 4. Agreement 复现分叉再凝聚

所有规则在`n=1`时 agreement 都接近1而目标函数质量为0。parity 随 n 增加先分叉，再按`parity1 -> parity4`的次序重新凝聚到目标函数；random 的 agreement 则持续降到约0.656且目标质量始终为0。高 agreement 只代表当前函数分布窄，不代表外部正确。

## 5. 边界

- parity 是预注册确认性家族，跨族曲线诊断包含事后分析；
- `n50/n90`是协议相对的操作阈值，不是机器无关样本复杂度；
- full-rule profile 强烈预测数据相变，但并未单独决定所有跨族排序；
- profile 为何具有特定形状仍是本文明确不求解的微观数学问题。
