# E30：动机与分阶段冻结协议

## 1. 问题

[Power等人的原版grokking实验](https://arxiv.org/abs/2201.02177)显示weight
decay显著促进延迟泛化，但grokking和规则恢复并不以weight decay为必要条件。
[Zhang等（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/file/92f67b9047fa7a43d7506054b5f0ec6a-Paper-Conference.pdf)
也用无显式正则的WanD找到高范数泛化解。一个常见解释仍把weight decay视作
optimizer动力学中的特殊干预；
但对显式L2，optimizer实际只接收完整标量目标

$$
J_\lambda(\theta)
=
L_D(\theta)+\lambda\lVert\theta\rVert^2/2
$$

的梯度。原框架最基本的两个对象本来就是loss地形与optimizer；显式L2只是在
同一标量loss中加入一项，并不是新的理论机制。首选预测应当是：无L2的raw-loss
地形本身已经可以支持规则接管，而加入L2会通过改变静态地形增强这种偏好。

## 2. 竞争解释

1. **完整静态地形解释。** 在相近raw BCE下，提高$\lambda$会增加正确AND延拓
   的静态质量，并压缩错误延拓的函数多样性。
2. **仅拟合深度解释。** L2条件只因raw BCE更低而拥有更多AND；matched-BCE后
   差异应消失。
3. **optimizer-specific解释。** 静态地形近乎不变，真实训练差异主要来自路径。

E30只直接裁决前两项。只有可靠静态测量留下明显残差后，第三项才值得检验。

## 3. 固定任务与测度

- 输入：全部8-bit字符串；目标为`x0 AND x1`，其余6 bit是nuisance。
- 训练集：四个主语义格各10个样本，共40个冻结样本。
- 网络：`8->16->16->1` tanh，433个标准化Gaussian坐标。
- 参考测度：坐标独立$N(0,1)$，网络前向使用fan-in缩放。
- 静态采样：direct constrained SMC；16个replica，每个8192粒子。
- 函数审计：每个replica保存2048个参数，并在完整256点上判定hard function。

## 4. 分阶段阈值冻结

本实验不是一次完全盲的单步预注册，而是只依据BCE、范数和采样诊断进行的分阶段
校准。阈值不得依据待报告的AND质量调整。

最终三个主条件是：

| $\lambda$ | 静态约束 | 实际raw BCE均值 |
|---:|---|---:|
| 0 | $L_D\le0.00268$ | 0.00258150 |
| $5\times10^{-5}$ | $J_\lambda\le0.0160$ | 0.00251046 |
| $10^{-4}$ | $J_\lambda\le0.0211$ | 0.00258978 |

中间点的BCE比两端低约3%，因此只能称为近似matched-BCE。所有图表必须同时展示
实际BCE，不能把三点写成精确相等。

## 5. 质量门槛

- replica log-volume标准差不高于1 nat；
- 最大probe跨replica标准差不高于0.03；
- 完整AND质量的replica标准差不高于0.05；
- 全部保存样本满足约束并100% hard-fit训练集；
- 以replica而非粒子作为配对统计单位。

浮点边界检查使用相对容差加16个float32 ULP；深loss最小容差仍为$10^{-12}$。

## 6. 外部关联

[Zhang等（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/file/92f67b9047fa7a43d7506054b5f0ec6a-Paper-Conference.pdf)
用Wang--Landau方法在模运算Transformer上观测到低loss泛化态的高熵优势。
E30不复现其完整Transformer熵图，而是在可严格审计的433参数系统中直接比较
显式L2完整目标的函数级静态质量。
