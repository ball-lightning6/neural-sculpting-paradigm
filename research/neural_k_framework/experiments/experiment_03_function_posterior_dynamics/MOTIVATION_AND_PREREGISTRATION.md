# E03：动机与预注册判据

## 1. 循环论证问题

原计划曾试图给256个 Boolean 函数构造手写 DSL 最短长度，再检验训练是否向最低复杂度函数迁移。该方案被主动放弃，因为：

1. DSL 最短长度只代表人为参考语言，不等于神经网络实际程序复杂度；
2. 用训练后函数惊讶度定义“神经复杂度”，再用它解释同一后验，会形成循环；
3. ANF、BDD、DNF 等指标可以作独立结构坐标，但没有单项足以冒充真实 K。

因此 E03 不测 K，也不把迁移方向预先命名为“压缩”。它只回答更基础的事实问题：固定训练约束后，优化是否继续系统改变完整函数？

## 2. E03a 设计

- `3-bit -> 1-bit`，完整256个 hard function；
- 网络：`3 -> 1024 x 3 -> 1`，GELU+LayerNorm，Adam，`weight_decay=0`；
- prior：65,536个初始化网络；
- 三个部分训练集：1、2、4样本，分别留下128、64、16个 hard 兼容函数；
- ordinary cohort：每条件128个普通初始化；
- prior-consistent cohort：每条件512个初始化时已满足全部训练标签的网络；
- post-fit 年龄：0到5,000步，共13个截面；
- 同时保存 hard function ID 与完整 logits；
- soft likelihood 温度族仅作基线，不作为事后自由调参的解释器。

主判据：prior-consistent 模型在没有新增 hard 约束时是否改变完整函数，以及 ordinary 模型在首次拟合后是否继续迁移。

## 3. E03b 设计

- 与 Oxford 同族：`3 -> width x 2 -> 1 tanh`，无归一化；
- `sigma_w=1`、`sigma_b=0.2`、Adam、`weight_decay=0`；
- width 16/32/64/128；
- 复用 E03a 完全相同的三组训练约束、prior/ordinary/consistent 样本量和年龄；
- width 128作为同族锚点，1024×3仅作跨架构行为参照，不伪装成纯宽度消融。

## 4. 竞争预测

- 若 hard-conditioned prior 足够，prior-consistent cohort 应保持初始函数分布；
- 若优化主动运输函数质量，一次更新即可产生超过抽样噪声的函数迁移；
- 若迁移只是大网络/LayerNorm 特例，tanh 小网络中应显著减弱或消失；
- 若网络协议定义有效参考语言，具体吸引函数和概率应随架构改变，即使“post-fit 仍迁移”的定性现象保留。

## 5. 结果前边界

- 不测量 Kolmogorov complexity；
- 不用后验本身给函数贴复杂度标签；
- 不预言人类看来最短公式一定胜出；
- 所有主体优化仍是 Adam，不能自动推广到任意优化器；
- aggregate 分布近似稳定不代表配对 seed 没有大量相反方向迁移。
