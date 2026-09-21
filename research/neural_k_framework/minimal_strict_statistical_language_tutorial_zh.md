# Neural K研究的最小严格统计语言：教学版

日期：2026-09-20

这份文档是教学版。它解释之前统计物理、算法信息论和信息增益语言中出现的对象怎样重新定义，以及为什么要这样改。

形式定义见[最小严格理论](../notes/20260920_minimal_strict_statistical_language_zh.md)，逐项问题见[审计与勘误](../notes/20260920_minimal_statistical_language_audit_zh.md)。

这份文档只保留和当前研究直接有关的内容。统计物理中的temperature、free energy、phase transition等词，如果没有对应的明确概率对象，都不作为基本机制使用。

## 1. 先固定研究协议

研究对象不是“一个抽象的神经网络”，而是一整套协议：

- 使用什么架构和参数化；
- 输入和输出怎样编码；
- 参数按照什么参考分布或测度抽取；
- 使用什么loss；
- 连续输出怎样读成hard标签；
- 讨论哪些输入。

参数记为 \(\theta\)，参考测度记为 \(\mu_\Pi\)。例如，\(\mu_\Pi\) 可以是某种高斯初始化分布。

网络有连续输出 \(F_\theta(x)\)。二分类时，它可能是logit。然后通过固定的读取规则得到hard输出 \(h_\theta(x)\)。

这两个对象必须分开：

\[
\text{连续输出和margin}
\qquad\neq\qquad
\text{hard标签}.
\]

如果一个logit是 \(0.1\)，另一个是 \(20\)，它们都可能读成标签1，但BCE完全不同。后面的loss和体积分析主要发生在连续输出空间中，hard function只是把参数分桶的方式。

## 2. 参数怎样变成函数质量

固定有限输入表：

\[
U=(x_1,\ldots,x_m).
\]

网络在这些输入上的联合输出是：

\[
Z_U(\theta)
=
(F_\theta(x_1),\ldots,F_\theta(x_m)).
\]

同时，在这些输入上的hard函数是：

\[
H_U(\theta)
=
(h_\theta(x_1),\ldots,h_\theta(x_m)).
\]

对一个hard函数 \(f\)，定义参数cell：

\[
C_U(f)=\{\theta:H_U(\theta)=f\}.
\]

它表示所有在 \(U\) 上实现同一个hard函数的参数。

函数的参考质量是：

\[
p_U(f)=\mu_\Pi(C_U(f)).
\]

这就是“随机初始化下，函数 \(f\) 有多常见”。

需要特别注意，“完整函数”总是相对于输入表 \(U\) 而言。如果 \(U\) 是4bit世界的全部输入，它可以代表完整真值表。如果 \(U\) 只有训练样本，它只是训练集上的函数限制。

如果后来查询一个不在 \(U\) 中的输入 \(x_*\)，就必须扩展为：

\[
U'=U\cup\{x_*\}.
\]

原来的函数cell可能在 \(U'\) 上分裂成许多不同的延拓。只知道旧输入上的函数，不能自动知道新输入上的标签。

## 3. 加入训练集和loss以后发生什么

训练集写成：

\[
D=((x_i,y_i))_{i=1}^n.
\]

平均loss是：

\[
L_D(\theta)
=
\frac1n\sum_i\ell(F_\theta(x_i),y_i).
\]

给定阈值 \(\epsilon\)，低loss区域是：

\[
A_D(\epsilon)=\{\theta:L_D(\theta)\le\epsilon\}.
\]

它是参数空间的一个区域。降低 \(\epsilon\) 会让这个区域变小。

但我们真正关心的是：这个区域里面，不同函数cell各占多少质量。定义：

\[
\Omega_{U,D}(f;\epsilon)
=
\mu_\Pi(C_U(f)\cap A_D(\epsilon)).
\]

它表示：

> 实现函数 \(f\)，并且训练loss不超过 \(\epsilon\) 的参数质量。

整个低loss区域的质量是：

\[
V_D(\epsilon)=\sum_f\Omega_{U,D}(f;\epsilon).
\]

因此低loss条件下的静态函数分布是：

\[
Q^{\rm static}_{U,D,\epsilon}(f)
=
\frac{\Omega_{U,D}(f;\epsilon)}
{V_D(\epsilon)}.
\]

这就是严格意义上的：

> 已知参数已经进入低loss区域后，各个完整函数的条件概率。

这里有一个最容易被忽略的事实：

\[
L_D(\theta)\neq L_D(f).
\]

同一个hard function内部，不同参数可以有不同logit、不同margin和不同loss。因此loss是在函数cell内部进一步切分参数，而不是简单地给每个函数贴一个固定分数。

## 4. 三种“加入一个样本”不能混为一谈

之前最危险的概念错误，是把下面三种操作写成了同一个“加入样本”。

### 4.1 在旧系综中预测hard标签

先固定父事件：

\[
E=A_D(\epsilon).
\]

对查询输入 \(x_*\)，定义标签分支：

\[
B_y=\{\theta:h_\theta(x_*)=y\}.
\]

那么当前低loss系综对标签 \(y\) 的预测概率是：

\[
\mu_\Pi(B_y\mid E).
\]

这是“在原有候选中，多少参数预测这个标签”。

如果观察到的真实标签就是 \(y\)，再做hard conditioning：

\[
E'=E\cap B_y.
\]

这是一个真正的条件概率更新。

### 4.2 保留旧loss条件，再加入连续loss要求

hard标签只告诉我们符号，不告诉我们margin大小。

如果新样本的连续loss是 \(\ell_y(\theta)\)，加入后平均loss为：

\[
L_{D^+}(\theta)
=
\frac{nL_D(\theta)+\ell_y(\theta)}{n+1}.
\]

如果我们仍然要求原参数先满足 \(E\)，再要求新平均loss达标，得到的是：

\[
E\cap\{L_{D^+}\le\epsilon'\}.
\]

这需要知道旧loss和新样本连续输出之间的联合分布。同一个hard function内部，只有部分参数可能通过这个连续条件。

### 4.3 重新定义新训练集的平均loss事件

如果直接从整个参数空间重新测：

\[
\{\theta:L_{D^+}(\theta)\le\epsilon'\},
\]

那么它可能包含原来 \(E\) 外的参数。

例如原来 \(L_D>\epsilon\)，但新样本loss非常低，加入后平均值反而低于阈值。这些参数从旧父事件外部进入了新事件。

因此：

\[
\{L_{D^+}\le\epsilon'\}
\]

一般既不是旧事件的子集，也不是“旧事件加一个hard标签条件”。

这就是为什么“当前训练集低loss下对测试标签的预测概率”和“把测试样本加入后新训练集仍低于同一loss阈值的概率”不是同一个量。

## 5. 信息增益到底是什么

信息增益只有在固定父事件和明确新增事件以后才严格成立。

给定父事件 \(E\)，新增hard标签事件 \(B_y\) 的信息成本是：

\[
-\log_2\mu_\Pi(B_y\mid E).
\]

它可以理解成：

> 在当前候选参数系综中，标签 \(y\) 有多意外。

令 \(E'=E\cap B_y\)，并假设 \(\mu_\Pi(E')>0\)。严格的方向是：

\[
D_{\rm KL,2}\bigl(\mu_\Pi(\cdot\mid E')\|
\mu_\Pi(\cdot\mid E)\bigr)
=-\log_2\mu_\Pi(B_y\mid E).
\]

这是“新条件分布 \(\| \) 旧条件分布”的KL。反向KL一般不成立；连续soft约束时，它通常变成联合分布下的对数似然比期望，不再是这个单点hard条件化恒等式。

但它不等于：

- 加入样本后重新训练的实际变化；
- 新平均loss事件的负对数质量；
- SGD训练获得的有效信息；
- 识别规则所需的样本数量。

如果标签是在模型当前预测分布中随机产生的，那么对这个surprise取期望得到模型的预测熵。

如果标签来自另一个外部真实分布，期望得到的是交叉熵。只有真实分布与模型分布一致时，二者才相同。

## 6. 为什么参数KL和函数KL不同

参数分布比函数分布细得多。

想象两个参数点：

- 它们都实现同一个hard function；
- 但第一个logit margin是1；
- 第二个logit margin是10。

如果新的连续loss条件只保留其中一个，那么参数分布发生了变化，参数KL大于零。但hard function仍然只有同一个，函数分布没有变化，函数KL为零。

在函数ID是有限或可数分区、条件分布存在时，参数KL有严格的链式分解。设 \(P_\theta,Q_\theta\) 是两个参数分布，\(F(\theta)\) 是完整hard function ID，诱导的函数分布分别为 \(P_F,Q_F\)，则：

\[
D_{\rm KL}(Q_\theta\|P_\theta)
=D_{\rm KL}(Q_F\|P_F)
+\mathbb E_{f\sim Q_F}
D_{\rm KL}\bigl(Q(\theta\mid f)\|P(\theta\mid f)\bigr).
\]

因此：

- 函数cell之间的概率重新分配；
- 每个函数cell内部实现方式的重新分配。

只有当新增条件完全等于若干完整函数cell的并集时，第二部分才消失。

这也是“只研究hard function”不够解释BCE、MSE和幅度对齐的原因。

## 7. Full-target和fixed-D不是同一个Neural K profile

完整目标曲线测的是：

> 从参考参数空间中，直接实现整个目标真值表，并达到某个连续精度的参数质量。

fixed-D分布测的是：

> 已经满足部分训练集loss以后，剩余候选完整函数怎样分配。

完整目标事件通常比部分训练事件更严格。设 \(A_D\) 是父训练事件，\(A_{\rm extra}\) 是在目标cell内部对held-out连续输出施加的额外要求，并且完整事件确实满足 \(A_{\rm full}\subseteq C_U(f)\cap A_D\)。这时应写成：

\[
\mu_\Pi(A_{\rm full})
=
\mu_\Pi(C_U(f)\cap A_D)
\mu_\Pi(A_{\rm extra}\mid C_U(f)\cap A_D).
\]

等价地：

\[
\mu_\Pi(A_{\rm full})
=
\mu_\Pi(A_D)\,
Q^{\rm static}_{U,D,\epsilon}(f)\,
\mu_\Pi(A_{\rm extra}\mid C_U(f)\cap A_D).
\]

前一个乘法式使用的是目标cell在父事件中的**绝对质量**，不是条件概率。最后一个cell内部通过率不能默认是1。

所以完整规则的 \(K^{N,\mathrm{full}}\) 不能直接当作训练集条件下目标函数的概率，也不能直接当作泛化概率。它和 \(K^{N,\mathrm{FD}}\) 由同一个联合输出空间连接，但不是同一个测量对象。

## 8. Neural K和“复杂度”应该怎样理解

当前有三个不同的负对数质量：

1. 无训练约束的函数参考质量；
2. 函数在低loss区域中的绝对质量；
3. 函数在低loss候选中的条件概率。

“Neural K”是一个总称。论文原来称为Neural K-profile的对象是完整目标profile；fixed-D对象单独记为 \(K^{N,\mathrm{FD}}\)。

完整目标profile写为：

\[
K^{N,\mathrm{full}}_{\Pi,U}(f;\epsilon)
=-\log_2V^{\mathrm{full}}_{\Pi,U}(f;\epsilon).
\]

fixed-D函数cell profile写为：

\[
K^{N,\mathrm{FD}}_{\Pi,U,D}(f;\epsilon)
=-\log_2\Omega_{\Pi,U,D}(f;\epsilon).
\]

条件函数概率的负对数还要扣除整个低loss区域的总成本。

因此：

> Neural K是一种架构、参数化、参考测度、输入域和loss协议相对的实现质量码长。

它和算法概率、MDL、Kolmogorov complexity有实质结构联系，因为它们都把“多个实现的总质量”转成负对数成本。但这不等于已经证明它和机器无关的经典Kolmogorov complexity相同。

## 9. 捷径淘汰和复杂函数收缩是两种不同效应

降低loss时，可能发生两件事。

第一，某些候选函数在当前网络中无法把loss降到足够低，于是它们退出可行集合。这是捷径淘汰。

第二，多个仍然可行的函数都能继续降loss，但它们的保留比例不同。若复杂函数的保留比例更低，它们的相对质量就会继续下降。这才是复杂函数收缩。

第二件事不是集合论必然结果，需要实验观察。它不能仅由“loss降低了”推出。

## 10. 联合输出分布和loss的关系

联合输出分布是底层对象：

\[
Z_D(\theta)=(F_\theta(x_1),\ldots,F_\theta(x_n)).
\]

loss只是对这个高维联合输出施加的标量函数：

\[
L_D(\theta)=\mathcal L_D(Z_D(\theta)).
\]

所以：

- 架构和参考测度决定联合输出分布；
- loss决定从联合输出分布中切出什么区域；
- hard读取规则决定怎样把连续输出分成函数cell。

BCE、MSE和幅度正则相当于在同一个联合输出空间中使用不同的筛选方向。

BCE低loss允许样本margin很不均匀；MSE更像围绕目标向量的球；幅度对齐正则则进一步限制联合输出的幅度差异。它们会保留不同的函数cell内部实现，因此可能产生不同的函数质量重排。

这就是为什么当前理论不再把BCE当成唯一神秘的压缩机制，而是研究：

> 联合输出分布在不同loss截面下，函数质量如何变化。

## 11. Agreement和复杂度分布也必须分开

函数复杂度分布变窄，不等于函数分布凝聚。

例如32个函数都有完全相同的参考复杂度，每个占 \(1/32\)：

- 复杂度方差是0；
- 但两个独立抽样得到同一个完整函数的概率只有 \(1/32\)。

完整函数agreement应该用collision描述：

\[
\sum_fQ(f)^2.
\]

逐点agreement又是另一个量，因为不同函数可能只在少数输入上不同。

所以“复杂度分布变窄”“完整函数agreement升高”“逐点预测一致”不能相互替代。

## 12. 静态质量与真实训练

静态质量回答：

> 参考测度下，某个函数区域有多大。

真实训练还包含optimizer如何运输概率质量。令 \(\nu_{\mathcal T,t}\) 是指定训练协议在时间 \(t\) 后的参数分布。无论是否绝对连续，都可以定义：

\[
Q^{\rm opt}_{\mathcal T,t,U}(f)
=\nu_{\mathcal T,t}(C_U(f)).
\]

只有在

\[
\nu_{\mathcal T,t}\ll\mu_\Pi
\]

时，才能定义 \(w_t=d\nu_{\mathcal T,t}/d\mu_\Pi\)，并写成：

\[
Q^{\rm opt}_{\mathcal T,t,U}(f)
=p_{\Pi,U}(f)\,
\mathbb E_{\mu_\Pi}[w_t\mid C_U(f)].
\]

直观上才可以说，训练最终得到的函数概率是：

> 静态函数质量乘以optimizer对该函数区域的平均偏重。

如果绝对连续条件不成立，只能使用前面的测度交集定义，不能强行写密度比。因此静态体积可以解释训练选择的底图，但不能自动等于真实SGD分布。

## 13. 这一版理论到底保留了什么

保留的主线是：

\[
\text{联合输出分布}
\rightarrow
\text{训练事件}
\rightarrow
\text{函数cell质量}
\rightarrow
\text{条件函数分布}
\rightarrow
\text{loss下降时的函数重排}.
\]

信息增益、MDL、Neural K、Agreement和体积收缩都从这条主线中取出，但每一个量都必须注明：

- 参考测度是什么；
- 输入域是什么；
- 训练集是什么；
- 使用累计loss、薄层还是完整目标；
- 是hard条件化、soft条件化还是重新采样；
- 是静态分布还是optimizer分布。

如果缺少这些信息，公式即使局部正确，也不能保证描述的是同一个研究对象。

## 14. 对象与现有实验的对应

这套语言不是纯粹的定义体操。当前对象和实验的对应关系大致是：

| 理论对象 | 对应的实验或证据 |
|---|---|
| \(\Omega_{\Pi,U,D}(f;\epsilon)\) 与 \(Q^{\rm static}\) | 3bit/4bit函数曲线、SMC测量、跨架构联合输出曲线 |
| hard标签条件概率 | 规则样本添加、规则对决、agreement和信息增益分析 |
| \(E\cap A_y^+\) 与 \(A_y^+\) 的区别 | full-target/fixed-D/margin闭环与平均loss对象审计 |
| 参数KL与函数KL的区别 | 同一hard function内部的不同margin和连续loss分析 |
| \(K^{N,\mathrm{full}}\) | 完整真值表扫描得到的Neural K-profile |
| \(K^{N,\mathrm{FD}}\) | 固定训练集下的函数cell质量重排 |
| 静态质量与optimizer transport | SMC静态采样和多seed真实训练、grokking对照 |
| 换参考测度的赔率修正 | Gaussian、均匀球和欧式体积测量的排序比较 |
| 联合输出与loss截面 | BCE、raw MSE、幅度目标和幅度对齐正则实验 |

这些实验支持的是各自协议下的对象和关系；它们不自动证明架构无关的收缩定律，也不自动证明 \(K^{N,\mathrm{full}}\) 与 \(K^{N,\mathrm{FD}}\) 在所有loss水平相同。
