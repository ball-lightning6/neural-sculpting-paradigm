# MDL最低可行复杂度、剩余候选重排与grokking

这是 2026-09-19 至 2026-09-20 理论讨论的可上传版本。完整推导见[MDL可辨识样本与grokking条件推导](mdl_identifiability_grokking_derivation_zh.md)，严格概率对象见[最小统计语言](minimal_strict_statistical_language_zh.md)。

## 最低可行复杂度

固定训练集 `D`、网络协议和参考复杂度 `K_ref(f)`。对完整函数 `f`，令 `Omega_D(f; epsilon)` 为函数cell中满足训练 loss 不超过 `epsilon` 的参数质量。则：

\[
\mathcal F_D(\epsilon)=\{f:\Omega_D(f;\epsilon)>0\},
\qquad
K_D^*(\epsilon)=\inf_{f\in\mathcal F_D(\epsilon)}K_{\rm ref}(f).
\]

它是“允许最大误差时最简单的可行函数”这一问题的严格定义。对同一个训练集和嵌套子水平集，`epsilon`降低时 `F_D(epsilon)`只能缩小，因此 `K_D^*(epsilon)`不下降。

这条严格性质不等于网络一定选择该函数，也不等于该函数的参数体积最大。

## 剩余候选中的简单性偏置

静态参考系综中的候选质量分布是：

\[
Q_{D,\epsilon}(f)=
\frac{\Omega_D(f;\epsilon)}{\sum_g\Omega_D(g;\epsilon)}.
\]

它不直接等于实际训练选择。若 $T_{D,\tau}$ 输出训练时刻 $\tau$ 的参数、$H_U$ 读取固定输入域上的 hard 函数，则参数分布为 $\nu_{D,\tau}=(T_{D,\tau})_\#\mu_{\rm init}$，函数分布为 $Q^{\rm opt}_{D,\tau}=(H_U)_\#\nu_{D,\tau}$；训练随机性一并纳入初始联合分布和协议。

降低 loss 有两个效果：

1. 淘汰不再满足训练约束的捷径；
2. 在仍然可行的函数之间重新分配质量。

若 `t` 是loss深度坐标（不是训练时间），`s_t(f)=-partial_t log Omega_t(f)`是函数的局部收缩率，则在正质量分支可微、相关矩有限且可交换微分与求和的区间，对固定复杂度标尺 `kappa(f)` 有：

\[
\partial_t E_{Q_t}[\kappa]
=-\operatorname{Cov}_{Q_t}(\kappa,s_t).
\]

复杂函数收缩更快、使该协方差为正时，平均复杂度下降。但这不保证方差下降；即使没有新模态，原有分支之间的质量重排也可以让方差增加。在同样的正则条件下：

\[
\partial_t\operatorname{Var}_{Q_t}(\kappa)
=-\operatorname{Cov}_{Q_t}
\left((\kappa-E_{Q_t}[\kappa])^2,s_t\right).
\]

因此收窄须单独检查这个协方差的符号或实测方差。即使复杂度方差为零，同成本的不同函数也未必凝聚到唯一共识；还须检查完整函数 collision、modal mass 或指定 Hamming 半径质量。这是联合输出分布、参数测度和loss共同决定的条件机制，而不是所有loss的普遍定理。

## MDL解释

\[
L_{\rm MDL}(f;D)=K_{\rm ref}(f)+L_{\rm residual}(D\mid f).
\]

`K_D^*(epsilon)`对应最低模型码长前沿，`Q_D,epsilon`对应静态参考系综中剩余可行候选的质量分布；实际训练偏好仍须由 `Qopt` 检查。网络没有显式优化第二个“程序长度”目标；这里的静态 MDL式偏好来自网络参考质量与loss几何的共同作用。

## 与grokking的关系

数据量轴、loss轴和训练时间轴必须分开：数据样本按竞争函数分歧位置淘汰函数，loss收紧重排剩余实现质量，训练时间还包含optimizer transport。因而复杂度可以和 `n50/n90`有条件的近似关系，但不等于数据相变点或grokking时间点。
