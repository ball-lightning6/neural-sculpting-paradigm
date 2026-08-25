# E06：完整结果与阶段裁决

## 1. 完整性

- full 采样4,194,304个`1024x3`未训练网络；
- 观察到255/256个 hard function；
- GPU 先验前向耗时65.9秒；
- 单/双/四样本 hard-exact 数量分别为2,096,325、867,458、365,248；
- 四样本最低0.1%尾部仍有366个样本。

## 2. 单样本：raw 效应主要受 logit scale 驱动

从 hard-exact 总体到 raw BCE 最低0.1%：

- 函数熵：`5.021 -> 1.098 bit`；
- 常量0质量：`28.1% -> 86.4%`；
- agreement：`0.610 -> 0.946`。

但在 normalized 和 fixed-scale 控制中，常量0质量下降或反转。因此该条件证明 raw loss 会强烈重排函数质量，却不能把这次重排全部解释成纯 hard-function 几何。

## 3. 双样本：排除尺度后的稳健函数选择

真实 SGD 主要吸引子 ID 113可写成：

```text
majority(not x0, not x1, x2)
```

其参数质量为：

- hard-exact 静态质量：1.57%；
- raw BCE 最低0.1%：23.0%；
- normalized BCE 最低0.1%：50.7%；
- fixed-scale 最低0.5%：44.1%。

因此，在相同 hard constraints 下，连续 loss 确实会对完整函数进行强烈、非公共因子的重排，而且方向与真实 SGD 后来选择的函数一致。ID 113在普通 AND/OR 公式语言中并非最短，却是 MLP 自然表达的对称线性阈值，这同时暴露了架构相对复杂度问题。

真实 SGD 一步后曾把 ID 113推到77.3%，高于静态可靠区间。静态几何给出方向，但没有完整给出 optimizer 运输强度。

## 4. 四样本：静态退火与 SGD 不可等同

SGD 吸引子 ID 48/243的总质量：

- hard-exact：21.4%；
- raw 中间低-loss 区最高约29.0%；
- raw 最低0.1%又降至23.0%；
- normalized 深尾消失并向其他函数偏移。

函数熵也没有持续下降：hard-exact 为3.715 bit，raw 最低0.1%为3.724 bit，normalized 最低0.1%为3.959 bit。真实 SGD 一步后却把 ID 48/243推至约77.2%。

因此真实 SGD 函数分布不能写成初始化测度在相同 loss 截面的静态条件分布，也不能写成简单的平衡退火系综。

## 5. 阶段裁决及后来的修正

E06 直接建立了本文后续理论的第一个静态事实：

> 在网络、参数测度、有限训练集和 hard constraints 固定后，继续收紧连续训练 loss 会以函数特异的方式重排完整函数质量；函数常数乘公共 loss 因子的可分离图景不成立。

它还建立了第二个分层事实：

> 静态低-loss 质量与真实多 seed 训练分布可以方向相近，但真实优化器是初始化分布的路径依赖 pushforward，不是该静态质量的无偏 Bayesian 样本。

在8月18日当时，“更低 loss 通常提高简单函数质量”被保留为高可信倾向；E06 本身尚不足以把它提升为定律。后来的 full-truth SMC、固定 D 测度闭环、体积到样本相变和深尾交叉实验，才把“简单”替换为可测的 Neural K-profile，并进一步证明排序依赖 loss 尺度、可以发生交叉。

不能声称：

- 每个数据集上 function entropy 都随 loss 单调下降；
- 任意预先指定的“真实函数”质量都随 loss 单调增加；
- raw BCE 效应都独立于 logit scale；
- SGD 等于静态 loss-conditioned posterior；
- E06 已经测量了动态逃逸率、稳定性或 Kolmogorov complexity。
