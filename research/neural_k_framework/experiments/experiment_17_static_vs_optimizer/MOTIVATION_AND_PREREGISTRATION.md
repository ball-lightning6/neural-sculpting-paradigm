# E17：实验动机与预注册判据

## 1. 两层体积假说

当时区分：

1. 静态层：lower loss 是否改变函数质量并偏向经济实现；
2. 训练层：真实泛化是否主要由该静态质量决定，使 optimizer 只剩二阶影响。

E17 专门压力测试第二层，同时校准第一层的 rare-event 测量。

## 2. 实验链

### A. 暴力 prior 与 optimizer crossing

4-bit 平衡 AND、`4 -> 16 x 2 -> 1 tanh`；33,554,432个未训练网络与4,096个 paired 初始权重，比较 AdamW、full-batch SGD 和 momentum SGD 首次跨越相同 raw BCE 时的函数分布。

### B. Constrained SMC

32,768粒子、8副本，通过 prior-preserving MCMC 逐层压低 BCE；在0.68/0.65/0.60与暴力 prior 交叉校准，再推进到0.07以下 hard-exact 深尾。

### C. 深尾真实训练

- 同网络、同训练集的32,768-seed AdamW 真实训练；
- 从 SMC 深尾 D440 权重抽取4,096个起点，配对正常 AdamW 与每步投影回初始化盒的 AdamW，检验静态深尾是否流回 AND。

### D. 时间混合

8-bit AND 与随机标签，2,048 seeds；比较单条轨迹64个晚期 checkpoint、IID-64 seed 样本和32条轨迹 pool，判断时间采样能否替代多 seed 分布。

## 3. 预先边界

- 静态质量与 optimizer crossing 必须在相同 loss 并有足够 prior/SMC 支持时比较；
- top 函数一致不等于完整分布相同；
- SMC 测量指定参数测度，不是坐标无关体积；
- hard ID 相同仍可能处于不同连续流管；
- 单个 agreement 相似不能证明完整分布混合。
