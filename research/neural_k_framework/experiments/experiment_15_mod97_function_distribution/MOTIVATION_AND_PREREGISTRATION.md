# E15：实验动机与预注册判据

## 1. 问题

Grokking 前的多 seed 分布究竟是共同 shortcut，还是共享规则骨架加 seed 特异残差？只看 validation accuracy 或 agreement 无法区分。完整函数 exact 又可能过严：每个 seed 只错不同少数点时，完整 hash 仍全部不同。

## 2. 设置

- 任务：`x / y mod 97`，完整输入空间9,312；
- 嵌套训练比例60%、70%、80%、90%；
- 每比例32个 paired seeds，共128个模型；
- MLP：`194 -> 512 x 3 -> 97` GELU+LayerNorm；
- Adam `1e-3`，无 weight decay；
- 运行121,101步；
- 在相同 train CE first crossing 处比较函数分布。

## 3. 同时报告的对象

- 完整目标函数质量；
- 完整函数 hash entropy 和 top 重复质量；
- coordinate-wise modal function；
- seed agreement；
- 所有条件共用的最后10% holdout；
- train CE 与未见准确率的关系。

## 4. 边界

32 seeds 只能解析不低于3.125%的单函数质量；modal function 是逐坐标边缘组合，不一定是任一单 seed 真实实现的完整函数；本实验测量 Adam 轨迹，不是静态体积。
