# 2026-09-19 至 2026-09-20 理论增量整理

这份文件专门整理周六到周日形成的理论内容，不与更早的85项增量混在一起。

## 1. 统计语言从类比变成严格对象

本轮把参数测度、联合输出、hard function cell、fixed-D loss事件、full-target事件、shell、信息增益、参数KL/函数KL和optimizer分布重新分开。核心修正是：

\[
\theta\longrightarrow Z_U(\theta)\longrightarrow \text{联合输出分布}
\longrightarrow \text{loss/幅度截面}
\longrightarrow \text{函数质量分布}.
\]

loss不是“能量”的先验类比，而是联合输出分布上的一个可测事件。full-target Neural K与fixed-D函数cell profile也不再共用一个未限定的符号。

对应材料：

- `release_20260920/theory/minimal_strict_statistical_language_zh.md`
- `release_20260920/theory/minimal_statistical_language_audit_zh.md`
- `release_20260920/theory/joint_output_rate_derivation_zh.md`
- `release_20260920/theory/volume_contraction_conditional_theorem_zh.md`

## 2. 允许最大误差时，最简单的可行函数

这是本周最重要的 MDL 增量。固定输入域、训练集 `D`、网络协议和参考复杂度 `K_ref(f)`。允许平均 loss 不超过 `epsilon` 时，定义可行函数集合：

\[
\mathcal F_D(\epsilon)
=
\{f:\Omega_{D}(f;\epsilon)>0\}.
\]

允许误差下的最低可行复杂度是：

\[
K^*_D(\epsilon)
=
\inf_{f\in\mathcal F_D(\epsilon)}K_{\rm ref}(f).
\]

这严格表达了：

> 允许最大训练误差是多少时，最简单的函数仍然能够完成任务？

对于同一个 `D` 和嵌套的 loss 子水平集，`epsilon` 降低时可行集合只会缩小，所以 `K^*_D(epsilon)` 不会下降。这是集合定义带来的严格性质。但它只说明“最简单可行函数”的前沿，不说明该函数占据最大体积，也不说明 SGD 一定会找到它。

## 3. 剩余候选中的简单性偏置

真正描述神经网络选择的是完整质量分布：

\[
Q_{D,\epsilon}(f)
=
\frac{\Omega_D(f;\epsilon)}
{\sum_g\Omega_D(g;\epsilon)}.
\]

因此需要区分两个过程：

1. **可行性淘汰**：某些捷径或错误延拓不再满足更低的 loss，离开 `F_D(epsilon)`；
2. **可行候选内重排**：仍然可行的函数，其实现质量以不同速度下降，`Q`发生变化。

若用单调的 loss 深度坐标 `t` 表示收紧过程，并记

\[
s_t(f)=-\partial_t\log\Omega_t(f),
\]

则对固定的复杂度标尺 `kappa(f)` 有：

\[
\partial_tE_{Q_t}[\kappa]
=-\operatorname{Cov}_{Q_t}(\kappa,s_t).
\]

当复杂函数的收缩率更大时，协方差为正，平均复杂度下降；若同时没有新的远离主模态的质量出现，复杂度分布会收窄。这正是“随 BCE/loss 下降，简单性偏置还会增强”的精确定义版本。

它仍是条件机制，不是任意 loss 的定理。`s_t`由联合输出分布、参数测度、架构和 loss 共同决定；BCE只是已测协议中的一种方向。

## 4. 与 MDL 的关系

两段式 MDL 写成：

\[
L_{\rm MDL}(f;D)
=K_{\rm ref}(f)+L_{\rm residual}(D\mid f).
\]

`K^*_D(epsilon)`描述允许残差预算下的最低模型码长；`Q_{D,epsilon}`描述在该残差预算下不同模型实现的质量分布。前者是“最简单可行解”的前沿，后者才对应网络在剩余候选中实际偏向谁。

因此不能把“MDL选择最简单函数”写成网络训练代码中额外存在的第二个目标。网络显式降低的仍然是训练 loss；MDL式偏好来自架构参考质量和 loss 截面共同产生的质量重排。

## 5. 与 grokking 数据相变的关系

增加数据和降低 loss不是同一个轴：

- 增加数据主要按照竞争函数与目标的分歧位置淘汰候选；
- 降低 loss主要在给定训练集的联合输出截面上继续重排实现质量；
- grokking时间轴还要加入optimizer transport和可达性。

在竞争函数分歧谱模型中，数据相变由目标质量、竞争总质量以及每个竞争函数被样本击中的概率共同决定；因此 `K^*_D(epsilon)`和 Neural K 是重要的一阶对象，但都不单独等于 `n50/n90`或grokking时间点。

## 6. 本周其余理论增量

- 联合输出的局部收缩率、端点交互项和有限窗口近似可加条件；
- Gaussian、固定范数球和自然测度下复杂度排序的鲁棒性边界；
- AIT/Omega与Neural K的实现质量结构对应，但不作机器无关等同；
- Peleg/G&C复核对“静态低loss质量”和“SGD分布”关系的外部核对。

## 7. 联合输出、幅度对齐和收缩定律的修正

周末讨论明确修正了一个容易出错的推导：不能给单个样本定义独立的“函数复杂度”，也不能把多个样本的体积简单相乘，因为共享网络参数会产生联合相关性。

幅度正则是否偏向简单函数，正确对象是同一个联合训练事件内不同函数cell的类内保留率；BCE方向的收缩率差异不能自动推出幅度对齐后的函数赔率差异。平移对称性也只能约化某些坐标，不能单独推出指数体积收缩。

有限窗口近似收缩需要额外条件：

- 输出平移造成的整体测度变化在研究窗口内足够小；
- 正负输出组新增的交互成本变化足够小；
- 深尾局部误差可控，且不能只用两个端点反推整段直线。

因此当前能上传的是“条件推导 + 可证伪实验协议”，不是跨架构、无限深尾的收缩定律。

## 8. Peleg/G&C周末复核的完整边界

文献审计发现，Peleg的G&C并非完全不考虑loss，但其归一化loss分桶不等于本项目的raw-BCE深尾事件；公开代码中G&C与SGD的bias活动空间、初始化尺度和学习率还存在需要区分的条件。

本地复核得到：G&C hard-start约82.13%，低loss静态样本约96.37%，SGD关键组约96.36%。这支持静态低loss质量具有预测力，但不证明静态分布等于SGD，也不允许把原论文的差异简单归因于“loss没有降够”。

## 9. AIT/Omega的最小保留内容

Omega、不完备性和算法概率只能提供实现质量求和、编码长度和不可计算性边界。它们不能推出：

- Neural K等于机器无关Kolmogorov complexity；
- 复杂度方差降低就等于函数混合修正降低；
- 有限神经网络的Agreement继承停机问题不可判定性。

这部分不作为独立文献审计上传，只把已经进入严格统计语言的边界保留在理论附录中。

## 10. 算法互信息是后续接口，不是本周实验结果

rule bit或concat输出可以用联合事件的条件信息成本近似测量输出块之间的冗余和协同，但它不是已经完成的独立实验，也不自动等于关于真实规则的可辨识信息。小任务和网络的输入/输出头开销可能使理论上近似等价的协议出现明显数值差异，因此本次不上传该实验草稿。

这些内容已分别放入 `release_20260920/theory/` 和 Peleg补充材料；本页作为周末理论增量的总入口。

## 11. 完整文档对应关系

本页是周末理论的公开正文入口；发布包同时保留以下完整源稿：

| 文档 | 作用 | 网站入口 |
|---|---|---|
| `minimal_strict_statistical_language_zh.md` | 最小严格定义、对象和条件恒等式 | [严格概率语言](../../site/post-upload-language.html) |
| `minimal_strict_statistical_language_tutorial_zh.md` | 面向阅读的逐步教学版 | [理论增量全文](../../site/post-upload-theory-update.html) |
| `minimal_statistical_language_audit_zh.md` | 前稿漏洞、有限反例和逐项勘误 | [理论增量全文](../../site/post-upload-theory-update.html) |
| `MDL_FRONTIER_AND_GROKKING.md` | 最低可行复杂度与剩余候选重排 | [MDL—grokking](../../site/post-upload-mdl.html) |
| `mdl_identifiability_grokking_derivation_zh.md` | 复杂度—分歧谱与样本相变条件推导 | [MDL—grokking](../../site/post-upload-mdl.html) |
| `MEASURE_ROBUSTNESS_AND_GAUSSIAN_ALIGNMENT_ZH.md` | Gaussian、欧式球和真实训练对应边界 | [E37 测度鲁棒性](../../site/experiments/e37.html) |

网页正文采用经过链接和双语结构校验的公开版，源稿保留完整推导、审计记录和复算入口；两者不是互相替代的重复副本。

完整源稿可直接打开：
<a href="source_docs/zh/minimal_strict_statistical_language_tutorial_zh.md">教学版</a> ·
<a href="source_docs/zh/minimal_strict_statistical_language_zh.md">严格定义版</a> ·
<a href="source_docs/zh/minimal_statistical_language_audit_zh.md">审计与勘误</a> ·
<a href="source_docs/zh/WEEKEND_THEORY_UPDATE.md">本周增量源稿</a>。
