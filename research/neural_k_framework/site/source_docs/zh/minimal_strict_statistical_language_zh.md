# Neural K研究的最小严格概率语言

日期：2026-09-20。状态：本轮修订稿；含推导与有限例核验，未做形式化证明器认证。

用途：用可测对象和明确的更新规则承载现有实验。教学讲解见[教学版](../neural_k_framework/minimal_strict_statistical_language_tutorial_zh.md)；原“算法概率解释”对话提出的对象错位及本轮修改见[逐项审计](20260920_minimal_statistical_language_audit_zh.md)。本稿取代旧统计语言教程作为最小定义的依据。

## 0. 固定对象与简写规则

固定协议 \(\Pi\)：参数空间 \(\Theta\)、架构及参数化、输入输出编码、参考概率测度 \(\mu_\Pi\)、单样本loss \(\ell\)、hard读取规则 \(h\)。假设网络前向输出有限且可测，\(\ell\) 非负、有限且可测，hard读取在平局处有固定规则。

连续输出记为 \(F_\theta(x)\in\mathbb R^q\)，hard输出 \(h_\theta(x)\) 属于固定有限字母表 \(\mathcal Y\)。二分类可用原始logit \(z_\theta(x)\) 和 \(h_\theta(x)=\mathbf1[z_\theta(x)\ge0]\)。不同编码要同时写明读取规则和loss，不能默认用0作阈值。

固定有限、有序、无重复的输入表 \(U=(x_1,\ldots,x_m)\)。训练数据是带索引的样本表 \(D=((\tilde x_i,y_i))_{i=1}^n\)，\(n\ge1\)，输入包含在 \(U\) 中；重复输入按其出现次数计入平均loss。把 \((x_*,y)\) 追加到数据表记为 \(D^+=D\mathbin{\|}(x_*,y)\)。

以下完整定义写出下标；在一节内明确固定 \((\Pi,U,D)\) 后可以省略。比较不同实验时必须恢复相关下标。涉及查询输入时先统一使用包含该输入的 \(U'=U\cup\{x_*\}\)，再比较同一分区上的概率。

默认 \(\mu_\Pi(\Theta)=1\)。均匀球测度须除以球体积；未归一化Lebesgue体积另记 \(\bar\mu\)，不能直接当概率。所有条件概率分母必须正；零质量事件的负对数记为 \(+\infty\)，但条件分布本身不定义。质量为零不等于事件为空。

\(\log_2\) 的单位是bit；\(\ln\) 的单位是nat。一个以nat定义的负对数质量除以 \(\ln2\) 才是bit。

## 1. 联合输出、hard函数与函数簇

定义联合连续输出及其推前测度：

\[
Z_U(\theta)=(F_\theta(x))_{x\in U},\qquad
\nu_{\Pi,U}=(Z_U)_\#\mu_\Pi.
\]

hard函数是 \(U\) 上的完整离散输出表：

\[
H_{\Pi,U}(\theta)=(h_\theta(x))_{x\in U},\qquad
\mathcal F_{\Pi,U}=H_{\Pi,U}(\Theta).
\]

这里“完整”只相对于 \(U\)；当 \(U\) 恰好是完整任务域时，才是全域hard function。定义函数cell及参考质量：

\[
C_{\Pi,U}(f)=\{\theta:H_{\Pi,U}(\theta)=f\},\qquad
p_{\Pi,U}(f)=\mu_\Pi(C_{\Pi,U}(f)).
\]

这些cell互斥且覆盖 \(\Theta\)，故 \(\sum_f p_{\Pi,U}(f)=1\)。同一个cell内，logits、逐样本loss、平均loss和权重实现都可以不同。一般不存在仅由 \(f\) 决定的单一 \(L_D(f)\)。

若用固定映射 \(\psi:\mathcal F_{\Pi,U}\to\mathcal G\) 定义函数簇，簇 \(a\) 的cell是 \(\bigcup_{\psi(f)=a}C_{\Pi,U}(f)\)，其质量是对应函数质量之和。只有互斥分区才可直接求和；重叠的性质事件不能当作互斥函数簇。分簇规则需在比较前固定。

仅改变loss、保持网络输出、读取规则和参考测度不变时，\(C_{\Pi,U}\) 与 \(\nu_{\Pi,U}\) 不变；改变的是对联合输出施加的事件。

## 2. Loss约束、累计质量与薄层

平均loss、总loss及子水平事件分别是：

\[
L_D(\theta)=\frac1n\sum_{i=1}^n\ell(F_\theta(\tilde x_i),y_i),
\qquad T_D(\theta)=nL_D(\theta),
\qquad A_D(\epsilon)=\{L_D\le\epsilon\}.
\]

函数分支的累计质量、总质量和条件函数概率是：

\[
\Omega_{\Pi,U,D}(f;\epsilon)
=\mu_\Pi(C_{\Pi,U}(f)\cap A_D(\epsilon)),
\]

\[
V_{\Pi,D}(\epsilon)=\mu_\Pi(A_D(\epsilon))
=\sum_f\Omega_{\Pi,U,D}(f;\epsilon),
\qquad
Q^{\rm static}_{\Pi,U,D,\epsilon}(f)
=\frac{\Omega_{\Pi,U,D}(f;\epsilon)}{V_{\Pi,D}(\epsilon)}.
\]

这最后一个比值是严格条件概率；其事件是“平均loss不超过阈值”，包含阈值以下所有状态。

为避免把态密度当成总会存在的普通函数，先定义函数分支的loss测度：

\[
\xi_{\Pi,U,D,f}(B)
=\mu_\Pi(C_{\Pi,U}(f)\cap\{L_D\in B\}),
\qquad B\subseteq[0,\infty)\text{ 可测}.
\]

于是 \(\Omega_f(\epsilon)=\xi_f([0,\epsilon])\)。有限薄层 \((a,b]\) 的质量是 \(\xi_f((a,b])=\Omega_f(b)-\Omega_f(a)\)；约定 \(\Omega_f(a)=0\) 对 \(a<0\)。若薄层要包含 \(L_D=0\) 的原子，必须选包含0的区间，不能用严格下界0把它漏掉。

只有在所研究区间内 \(\xi_f\) 对Lebesgue测度绝对连续时，才写 \(\xi_f(dt)=\rho_f(t)\,dt\)。对总密度为正的几乎处处 \(t\)，密度定义的函数条件分布为：

\[
Q^{\rm density}_{U,D,t}(f)=\frac{\rho_f(t)}{\sum_g\rho_g(t)}.
\]

“精确等于某个loss”的普通条件概率可能因事件质量为0而不定义；上式是在密度存在时选定的条件版本。它也不等于轨迹首次经过该loss时的训练分布。

在上述可微条件且 \(\epsilon,\Omega_f(\epsilon)>0\) 时，定义局部收缩指数：

\[
\lambda_f(\epsilon)
=\frac{d[-\log_2\Omega_f(\epsilon)]}{d\log_2(1/\epsilon)}
=\frac{\epsilon\,\rho_f(\epsilon)}{\Omega_f(\epsilon)}.
\]

因此，在分母为正时有精确关系：

\[
Q^{\rm density}_{U,D,\epsilon}(f)
=\frac{Q^{\rm static}_{U,D,\epsilon}(f)\lambda_f(\epsilon)}
{\sum_{g:\Omega_g(\epsilon)>0}Q^{\rm static}_{U,D,\epsilon}(g)\lambda_g(\epsilon)}.
\]

此式先对 \(\Omega_f(\epsilon)>0\) 的函数使用；在上述普通可微点，零累计质量分支的导数也为0，不使用 \(0/0\) 的指数。薄层与累计分布的赔率还差一个局部指数比，不能只凭“进入深尾”就认定两者相同。以上主式用平均loss；若态密度改用总loss \(T_D\)，积分上限须改为 \(n\epsilon\)，密度转换也有对应的 \(n\) 因子。

## 3. 三种负对数质量与复杂度分布

保持本项目Neural Kolmogorov Complexity命名，但把“Neural K”总称下的两个profile写开：

\[
K^{\rm prior}_{\Pi,U}(f)=-\log_2p_{\Pi,U}(f),
\]

\[
K^{N,\mathrm{FD}}_{\Pi,U,D}(f;\epsilon)
=-\log_2\Omega_{\Pi,U,D}(f;\epsilon),
\qquad
K^{N,\mathrm{full}}_{\Pi,U}(f;\epsilon)
=-\log_2V^{\mathrm{full}}_{\Pi,U}(f;\epsilon),
\qquad
J_{\Pi,D}(\epsilon)=-\log_2V_{\Pi,D}(\epsilon).
\]

其中 \(K^{N,\mathrm{FD}}\) 是fixed-D函数cell质量的profile，\(K^{N,\mathrm{full}}\) 是论文中原Neural K-profile的完整目标体积。当 \(D=D_{\rm full}(f)\)、阈值相同且完整目标事件包含于 \(C_f\) 时，相应cell交集可省去，两个量相等。部分训练集的fixed-D量还需要第6.2节的事件桥接，不能仅凭hard-exact就与完整目标量等同；一般不能共用未加后缀的符号。

条件函数的剩余信息成本满足：

\[
-\log_2Q^{\rm static}_{\Pi,U,D,\epsilon}(f)
=K^{N,\mathrm{FD}}_{\Pi,U,D}(f;\epsilon)-J_{\Pi,D}(\epsilon).
\]

这是同一父事件内的fixed-D分支信息成本，不能拿 \(K^{N,\mathrm{full}}\) 直接替入。

另取一个在扫描前固定的成本标尺 \(c(f)\)，例如预先测定的参考深度Neural K、规则结构指标或另行规定的描述成本。它在当前扫描中不随 \(\epsilon\) 改变。对应的成本分布是推前分布：

\[
\mathcal P_{D,\epsilon}^{c}(B)
=\sum_{f:c(f)\in B}Q^{\rm static}_{U,D,\epsilon}(f).
\]

“函数数量的复杂度直方图”“按当前质量加权的复杂度分布”和“当前 \(-\log\Omega_f\) 的分布”是不同量。必须报告画的是哪一种。

### 3.1 固定D收紧loss的更新

固定协议、数据和读取规则，取 \(\epsilon_2<\epsilon_1\)，两次总质量都为正。简写 \(\Omega_i(f),Q_i(f)\)，并在 \(S=\{f:\Omega_1(f)>0\}\) 上定义：

\[
r_f=\frac{\Omega_2(f)}{\Omega_1(f)}\in[0,1].
\]

对 \(f\in S\)，直接由归一化得到：

\[
Q_2(f)=\frac{Q_1(f)r_f}{\sum_{g\in S}Q_1(g)r_g}.
\]

对 \(f\notin S\)，嵌套性给出 \(Q_1(f)=Q_2(f)=0\)，不定义 \(0/0\) 保留率。\(r_f=0\) 表示质量退出；较小的正 \(r_f\) 表示相对抑制。赔率变化取决于 \(r_f/r_g\)。

对在 \(Q_1\) 下可积的固定成本 \(c\)，有：

\[
\mathbb E_{Q_2}c-\mathbb E_{Q_1}c
=\frac{\operatorname{Cov}_{Q_1}(c,r)}{\mathbb E_{Q_1}r}.
\]

负协方差足以使平均固定成本下降；该协方差的符号是经验内容。这不自动保证方差下降或唯一函数占全部质量。所有 \(\Omega_f\) 减小导致当前 \(K^{N,\mathrm{FD}}_{\Pi,U,D}(f;\epsilon)\) 上升，与固定参考成本 \(c\) 的平均值下降可以同时发生。

## 4. 预测与新增样本的三种操作

固定父事件 \(E=A_D(\epsilon)\)，写 \(\mu_E=\mu_\Pi(\cdot\mid E)\)。为比较同一对象，本节函数均在包含查询点的 \(U'\) 上定义，简写 \(C_f=C_{\Pi,U'}(f)\)、\(Q_E(f)=\mu_E(C_f)\)。

### 4.1 预测hard标签与添加hard约束

定义互斥且覆盖父事件的查询标签分支：

\[
B_y=\{\theta:h_\theta(x_*)=y\},\qquad
p_E(y\mid x_*)=\mu_E(B_y)=\sum_{f:f(x_*)=y}Q_E(f).
\]

这是添加标签前的hard预测，\(\sum_y p_E(y\mid x_*)=1\)。观察到标签 \(y\) 后，若规定的更新就是 \(E_y^{\rm hard}=E\cap B_y\)，则：

\[
Q^{\rm hard}_{E,y}(f)
=\frac{Q_E(f)\mathbf1[f(x_*)=y]}{p_E(y\mid x_*)}.
\]

与标签一致的指定目标函数概率只能增加或不变。这条结论的前提是本次更新按hard分支条件化。

若之前只保留了不包含查询点的旧表 \(H_U\) 的边缘分布，就不足以计算 \(p_E(y\mid x_*)\)。旧cell在扩展到 \(U'\) 后会细分；必须保留这种细分或等价的联合信息。

### 4.2 保留旧条件，再添加连续loss要求

hard标签不决定样本的loss。二分类BCE有：

\[
s_y=2y-1,\qquad
m_{\theta,y}=s_yz_\theta(x_*),\qquad
\ell_{\theta,y}=\ln(1+e^{-m_{\theta,y}}).
\]

同一个hard标签可对应不同的正margin，因此要保留联合分布。对任意父事件 \(E\) 定义：

\[
\eta_E^{x_*}
=(H_{\Pi,U'}(\theta),L_D(\theta),F_\theta(x_*))_\#\mu_E.
\]

在这个分布中，用坐标 \((f,t,z)\) 表示函数、旧平均loss和查询连续输出。给定候选标签 \(y\) 和新阈值 \(\epsilon'\)，定义判定函数：

\[
a_y(t,z)=
\mathbf1\!\left[\frac{nt+\ell(z,y)}{n+1}\le\epsilon'\right],
\qquad
A_y^+=\{L_{D\|(x_*,y)}\le\epsilon'\}.
\]

在旧父事件内继续筛选的接受率及函数分布是：

\[
a_{E,y}=\int a_y(t,z)\,\eta_E^{x_*}(df,dt,dz)
=\mu_\Pi(A_y^+\mid E),
\]

\[
Q^\cap_{E,y}(f)
=\frac{\int\mathbf1[\tilde f=f]a_y(t,z)\,
\eta_E^{x_*}(d\tilde f,dt,dz)}{a_{E,y}}
=\mu_\Pi(C_f\mid E\cap A_y^+).
\]

必须保留符号 \(\cap\)：这个分布条件于“旧要求和新要求同时满足”。同一 \(f\) 内可能只有部分参数通过，所以更新系数一般不是hard标签的0/1指示。

### 4.3 加入数据后重新定义平均loss事件

实际另做一轮静态采样，若只要求新数据的平均loss不超过 \(\epsilon'\)，测的是：

\[
W_y=\mu_\Pi(A_y^+),\qquad
Q^{\rm new}_y(f)=\mu_\Pi(C_f\mid A_y^+).
\]

它可由不预先限制旧loss的 \(\eta_\Theta^{x_*}\) 计算。仅有 \(\eta_E^{x_*}\) 一般无法恢复它，因为新集合可能包含 \(E\) 外的参数。

精确的进入、退出分解是：

\[
\Omega^{\rm new}_y(f)
=\underbrace{\mu_\Pi(C_f\cap E)}_{\Omega^{\rm old}(f)}
-\underbrace{\mu_\Pi(C_f\cap E\setminus A_y^+)}_{\mathrm{out}_y(f)}
+\underbrace{\mu_\Pi(C_f\cap A_y^+\setminus E)}_{\mathrm{in}_y(f)}.
\]

将右侧对 \(f\) 求和得到 \(W_y\)，再归一化得到 \(Q_y^{\rm new}\)。若 \(\mu_\Pi(A_y^+\setminus E)=0\)，两种参数条件分布相同，因而 \(Q_y^{\rm new}=Q^\cap_{E,y}\)。函数边缘也可能偶然相同而参数分布不同，所以这不是函数边缘相等的必要条件。

一个足够的嵌套条件是 \((n+1)\epsilon'\le n\epsilon\)，因为loss非负。保留同一个平均阈值 \(\epsilon'=\epsilon>0\) 一般不满足它。

不同候选标签的 \(A_y^+\) 可以重叠，也可能留空隙。因此 \(W_y/V_D(\epsilon)\) 不必是概率，甚至可能大于1；即使把 \(W_y\) 跨标签重新归一化，也通常不等于 \(p_E(y\mid x_*)\)。把它作为一个单独预测评分协议可以，但不能援引hard分支恒等式证明它与原预测相同。softmax概率的平均 \(\mathbb E_{\mu_E}p_\theta(y\mid x_*)\) 又对应额外指定的随机标签模型，也不自动等于hard投票。

## 5. 信息成本、KL及其层级

对正质量参数事件 \(E\)，定义 \(J(E)=-\log_2\mu_\Pi(E)\)。若 \(E'=E\cap B\) 且 \(\mu_\Pi(E')>0\)，直接由密度比得到：

\[
I_E(B):=-\log_2\mu_\Pi(B\mid E)
=J(E')-J(E)
=D_{\rm KL,2}(\mu_{E'}\|\mu_E).
\]

这是参数分布之间的KL。若要换成hard函数分布，须使用有限分区的KL链式分解：

\[
D_{\rm KL,2}(\mu_{E'}\|\mu_E)
=D_{\rm KL,2}(Q_{E'}\|Q_E)
+\sum_{f:Q_{E'}(f)>0}Q_{E'}(f)
D_{\rm KL,2}\!\left(
\mu_{E'}(\cdot\mid C_f)\,\middle\|\,\mu_E(\cdot\mid C_f)
\right).
\]

推导：在每个正质量cell内，将参数密度比拆成“cell概率比”乘“cell内部条件密度比”，再积分。

当 \(B\) 是该分区中若干完整cell的并时，保留下来的cell内部条件分布不变，最后一项为0，参数KL才等于函数KL。连续loss筛选通常会改变cell内部，不能省去这一项。

反例：两个等质量参数点具有同一hard function，筛选后只留一个。参数KL是1 bit；函数始终只有一个，函数KL是0。这正是“hard function相同但loss不同”在信息公式中的体现。

### 5.1 预测熵与信息链

对4.1节的互斥hard标签分支，在标签按当前预测分布 \(p_E\) 取值时：

\[
\mathbb E_{Y\sim p_E}[-\log_2p_E(Y\mid x_*)]
=H_{\rm Sh}(p_E).
\]

这里 \(H_{\rm Sh}(p_E)=-\sum_y p_E(y)\log_2p_E(y)\) 是Shannon熵。

若按外部真实标签分布 \(r\) 取期望，结果是交叉熵 \(H_{\rm Sh}(r)+D_{\rm KL,2}(r\|p_E)\)，不一定等于模型预测熵。

对固定交集序列 \(E_{j+1}=E_j\cap B_j\)，每一步质量都正，则：

\[
\sum_{j=0}^{N-1}I_{E_j}(B_j)
=-\log_2\frac{\mu_\Pi(E_N)}{\mu_\Pi(E_0)}.
\]

固定最终交集时总成本与加入顺序无关。它不是对重新定义平均loss事件、重新训练或动态选择不同最终数据的无条件顺序不变量。

### 5.2 最小KL原则

对任意满足 \(P\ll\mu_\Pi\)、\(P(E)=1\) 的概率测度，密度分解给出：

\[
D_{\rm KL,2}(P\|\mu_\Pi)
=J(E)+D_{\rm KL,2}(P\|\mu_E),
\]

\[
J(E)=\min_{P\ll\mu_\Pi,\ P(E)=1}D_{\rm KL,2}(P\|\mu_\Pi).
\]

最小值由 \(\mu_E\) 达到。这是分布信息成本；它未给出训练时间、真实新增数据bit数或SGD能达到的最小计算成本。

## 6. 信息缺口与完整目标曲线

先只考虑hard标签事件 \(B_D=\{\theta:h_\theta(\tilde x_i)=y_i,\ \forall i\}\)。定义：

\[
G^{\rm hard}_{\Pi,U}(D)=-\log_2\mu_\Pi(B_D).
\]

若目标 \(f_*\) 与数据一致、\(p_{\Pi,U}(f_*)>0\)，则 \(C_{f_*}\subseteq B_D\)，所以：

\[
-\log_2\mu_\Pi(C_{f_*}\mid B_D)
=K^{\rm prior}_{\Pi,U}(f_*)-G^{\rm hard}_{\Pi,U}(D).
\]

对一般连续loss父事件 \(E\)，正确的式子仍是：

\[
-\log_2\mu_\Pi(C_{f_*}\mid E)
=-\log_2\mu_\Pi(C_{f_*}\cap E)+\log_2\mu_\Pi(E).
\]

不能删除分子中的 \(E\)。这一“剩余多少bit”是指定参考分布下的目标概率成本，不自动换算成还需要多少随机样本。

### 6.1 Full-target不能替代fixed-D

固定完整任务域 \(U\)，对外部目标表 \(g\) 定义 \(D_g^{\rm full}=((x,g(x)))_{x\in U}\)，以及旧完整规则扫描测量的：

\[
V^{\rm full}_{\Pi,U}(g;\epsilon)
=\mu_\Pi\{L_{D_g^{\rm full}}\le\epsilon\},
\qquad
K^{N,\rm full}_{\Pi,U}(g;\epsilon)=-\log_2V^{\rm full}_{\Pi,U}(g;\epsilon).
\]

这与部分训练集的 \(K^{N,\mathrm{FD}}_{\Pi,U,D}(g;\epsilon)\) 是两种定义。高loss下，不同目标的full-target事件可重叠，也可包含hard表并不等于 \(g\) 的参数。

对 \(M\) 个二分类BCE标量项的等权平均、标准0阈值读取，\(\epsilon<\ln2/M\) 是保证完整hard表正确的充分条件：一个错分类或零margin项就至少贡献 \(\ln2\)。这是充分条件，不是必要条件。

### 6.2 连续loss的严格桥

为保证嵌套，设完整表分成不重叠的 \(D\) 和 \(H_g\)，分别有 \(n\) 和 \(k\) 个样本，并使用同一个总loss预算 \(\tau>0\)：

\[
E_D^\tau=\{T_D\le\tau\},\qquad
E_g^\tau=\{T_D+T_{H_g}\le\tau\}\subseteq E_D^\tau.
\]

于是严格有：

\[
\mu_\Pi(E_g^\tau)
=\mu_\Pi(E_D^\tau)
\mu_\Pi(E_g^\tau\mid E_D^\tau).
\]

换回平均loss，左右对应阈值分别是 \(\tau/(n+k)\) 与 \(\tau/n\)，不能使用同一个数字而省去样本数转换。

如果预算还保证 \(E_g^\tau\subseteq C_g\)（标量二分类BCE可取 \(\tau<\ln2\)），则：

\[
\mu_\Pi(E_g^\tau)
=\mu_\Pi(C_g\cap E_D^\tau)
\underbrace{\mu_\Pi(E_g^\tau\mid C_g\cap E_D^\tau)}_{b_g(\tau)}.
\]

这最后的 \(b_g\) 是“已经完整hard正确之后，额外满足held-out连续精度的比例”。它一般不等于1。因此full-target质量不能直接当作fixed-D中的目标函数质量。即使已经知道每一个候选完整表的full-target质量，通常仍缺少这些内部精度因子。

## 7. 最低可行复杂度与捷径淘汰

固定一般 \((\Pi,U,D)\) 和固定成本 \(c(f)\)。令：

\[
r_D^{\min}(f)=\inf_{\theta\in C_f}L_D(\theta),\qquad
\mathcal F_D^\exists(\epsilon)
=\{f:\exists\theta\in C_f,\ L_D(\theta)\le\epsilon\},
\]

\[
c_D^*(\epsilon)=\inf_{f\in\mathcal F_D^\exists(\epsilon)}c(f).
\]

空集合的下确界记为 \(+\infty\)。\(r_D^{\min}(f)<\epsilon\) 保证可行，\(r_D^{\min}(f)>\epsilon\) 保证不可行；等号处须检查下确界是否取到。可采样的正质量候选另取 \(\mathcal F_D^+(\epsilon)=\{f:\Omega_f(\epsilon)>0\}\)。

由子水平集嵌套：

\[
\epsilon_2<\epsilon_1
\quad\Longrightarrow\quad
c_D^*(\epsilon_2)\ge c_D^*(\epsilon_1).
\]

它描述“允许误差降低后，最低可行成本不能更低”。是否恰好有低成本捷径退出、其后复杂候选是否相对收缩更快，需要测量；集合论不保证这两种效应的相对强弱或平均成本的单调性。

## 8. Agreement、混合修正与概率预算

固定一套函数分布 \(Q\)，可选 \(Q^{\rm static}\) 或规定训练协议后的 \(Q^{\rm opt}\)，每次计算都要说明。以下两次抽样独立且来自同一个 \(Q\)。

完整函数碰撞率、Rényi-2熵、最大函数概率分别是：

\[
\operatorname{Coll}(Q)=\Pr[f=g]=\sum_fQ(f)^2,\qquad
H_{\rm R2}(Q)=-\log_2\operatorname{Coll}(Q),\qquad
p_{\max}=\max_fQ(f).
\]

静态条件分布的混合修正定义为：

\[
\Delta_{\rm mix}=-\log_2p_{\max}
=\log_2\frac{\sum_f\Omega_f}{\max_f\Omega_f}.
\]

因为最大项的平方不超过平方和，而 \(\sum_fQ(f)^2\le p_{\max}\sum_fQ(f)\)，有：

\[
2^{-2\Delta_{\rm mix}}
\le\operatorname{Coll}(Q)\le2^{-\Delta_{\rm mix}}.
\]

\(\Delta_{\rm mix}\to0\) 等价于完整函数碰撞率趋1。这里选的是最大质量函数，未证明它等于独立复杂度 \(c\) 的最小者。

在输入 \(x\in U\) 上定义 \(p_x(y)=\sum_{f:f(x)=y}Q(f)\)。逐点成对一致率是 \(\sum_y p_x(y)^2\)，多数标签比例是 \(\max_y p_x(y)\)。对预先固定的probe分布平均，才得到相应总体指标；多bit输出还须区分逐bit和整向量一致。

32个不同函数各占 \(1/32\)、且 \(c(f)\) 完全相同，成本方差是0，但完整碰撞率只有 \(1/32\)。因此成本分布收窄不等于函数凝聚；高agreement也没有涉及外部真值是否正确。

若数据集本身随机，应先定义数据分布 \(\mathcal D\)。平均条件碰撞率 \(\mathbb E_D\sum_fQ_D(f)^2\) 与混合分布碰撞率 \(\sum_f(\mathbb E_DQ_D(f))^2\) 一般不同。

对同一父事件内互斥分支 \(\Omega_f\)，记 \(V=\sum_f\Omega_f\le1\)、\(k_f=-\log_2\Omega_f\)，则对有限 \(k\)：

\[
\#\{f:k_f\le k\}\le V2^k.
\]

每个被计数分支至少占 \(2^{-k}\)，故可直接证明。彼此重叠的full-target事件不满足这个计数前提。

## 9. 静态质量、训练分布与换测度

固定训练协议 \(\mathcal T\)，包含训练数据、初始化、优化器、随机性和停止规则。其时间 \(t\) 的参数分布记为 \(\nu_{\mathcal T,t}\)，永远可以定义：

\[
Q^{\rm opt}_{\mathcal T,t,U}(f)=\nu_{\mathcal T,t}(C_f).
\]

如果 \(\nu_{\mathcal T,t}\ll\mu_\Pi\)，令 \(w_t=d\nu_{\mathcal T,t}/d\mu_\Pi\)。对正参考质量的cell：

\[
Q^{\rm opt}_{\mathcal T,t,U}(f)
=p_{\Pi,U}(f)\mathbb E_{\mu_\Pi}[w_t\mid C_f].
\]

对同一个参数事件 \(E\)，若两种测度都赋予它正质量，并且 \(\mu_\Pi(C_f\cap E)>0\)，则：

\[
Q^{\rm opt}_{\mathcal T,t,U}(f\mid E)
=Q_E(f)\,
\frac{\mathbb E_{\mu_\Pi}[w_t\mid C_f\cap E]}
{\mathbb E_{\mu_\Pi}[w_t\mid E]}.
\]

零参考质量cell在绝对连续前提下也有零训练质量，应单独处理。若绝对连续不成立，例如训练分布支撑在参考零测度集合上，则保留测度定义，不使用密度比分解。这些式子没有预测运输因子。

换参考测度也类似：同一参数空间、同一事件 \(A,B\)，若 \(d\widetilde\mu=w\,d\mu_\Pi\)，并且 \(\mu_\Pi(A),\mu_\Pi(B),\widetilde\mu(B)>0\)，有：

\[
\frac{\widetilde\mu(A)}{\widetilde\mu(B)}
=\frac{\mu_\Pi(A)}{\mu_\Pi(B)}
\frac{\mathbb E_{\mu_\Pi}[w\mid A]}
{\mathbb E_{\mu_\Pi}[w\mid B]}.
\]

排序是否保持取决于该修正因子。重新参数化时，若同时推前测度及事件，质量保持；换坐标后重新指定“均匀测度”通常构成另一个协议。

## 10. 输出幅度与收缩律的条件命题

本节给出明确的欧式协议作为条件推导的例子。令标量输出为 \(z_i=g_\phi(x_i)+b\)，其中 \(b\) 是公共输出bias。取有限欧式体积区域 \(S\) 限制 \(\phi\)，对 \(b\in\mathbb R\) 用Lebesgue测度。对二值目标 \(f\) 设：

\[
y_f(l,r)_i=\begin{cases}l,&f(x_i)=0,\\r,&f(x_i)=1.\end{cases}
\]

固定平均MSE容差 \(\delta>0\)，定义几何体积：

\[
V_f^{\rm geo}(l,r;\delta)
=\int_S\int_{\mathbb R}
\mathbf1\!\left[\frac1m\|g_\phi(U)+b\mathbf1-y_f(l,r)\|^2\le\delta\right]db\,d\phi.
\]

它不超过 \(2\sqrt{\delta}\operatorname{Vol}(S)\)，但总参考域体积无限，不能用它定义 \(\theta\sim\bar\mu\) 的全域均匀概率。对正有限事件体积，定义 \(J_f=-\ln(V_f^{\rm geo}/v_0)\)，其中 \(v_0>0\) 是固定同单位体积尺度，消除对有量纲体积直接取对数的歧义。

变量替换 \(b'=b+c\) 严格给出：

\[
V_f^{\rm geo}(l+c,r+c;\delta)=V_f^{\rm geo}(l,r;\delta),
\qquad J_f(l,r)=H_f(r-l),\quad H_f(d):=J_f(0,d).
\]

这个性质依赖事件对平移协变，以及同一参考测度对该变换不变。固定高斯或固定球的概率测度一般不满足。对任意可测双射 \(T_c\)，同时推移事件和测度满足 \((T_c)_\#\mu(T_cA)=\mu(A)\)，仅是换元恒等式。

再假设在一个固定网格窗口，端点混合差分有统一界：

\[
\Gamma_f(l,r)
=J_f(l-h,r+h)-J_f(l-h,r)-J_f(l,r+h)+J_f(l,r),
\quad |\Gamma_f|\le\eta,\quad h>0.
\]

平移约化后 \(\Gamma_f=H_f(d+2h)-2H_f(d+h)+H_f(d)\)。对所有相邻网格都满足该界时，令 \(d_j=d_0+jh\)、\(D_0=H_f(d_1)-H_f(d_0)\)，逐项累加得到：

\[
|H_f(d_n)-H_f(d_0)-nD_0|\le\eta n(n-1)/2.
\]

因此在误差可接受的窗口内，\(V_f^{\rm geo}(d_n)\) 近似于 \(V_f^{\rm geo}(d_0)e^{-nD_0}\)。网格线性不自动控制网格之间；连续版本还需要可微性与二阶导数界。称为“收缩”还需 \(D_0>0\)。移动目标的MSE邻域一般不嵌套，其质量比不能叫条件存活率。

平移对称性本身没有推出 \(|\Gamma_f|\) 小，更没有推出复杂函数系数更大。近似平移时的误差累积见[条件推导](20260919_volume_contraction_conditional_theorem_zh.md)。

平均BCE子水平集、最小margin事件与等幅MSE邻域也不同。另取固定参考概率 \(\mu_\Pi\)，定义 \(b_{\rm BCE}(a)=\ln(1+e^{-a})\)、\(s_i=2f(x_i)-1\)、\(W_f(a)=\mu_\Pi\{\min_i s_i z_i\ge a\}\)。对 \(\epsilon>0\)，非负性给出：

\[
W_f(b_{\rm BCE}^{-1}(\epsilon))
\le \mu_\Pi\{m^{-1}\sum_i b_{\rm BCE}(s_i z_i)\le\epsilon\}
\le W_f(b_{\rm BCE}^{-1}(m\epsilon)).
\]

这桥接BCE与最小margin，未将最小margin事件替换成等幅输出邻域。固定有界参数域的连续网络还可能有正的最低BCE，不能默认 \(\epsilon\to0\) 的事件始终存在。

## 11. AIT与MDL保留到什么程度

固定最优通用前缀机 \(\mathsf M\)，以及有限函数表的固定二进制编码 \(\langle f\rangle\)。算法概率和前缀复杂度满足：

\[
m_{\mathsf M}(\langle f\rangle)
=\sum_{p:\mathsf M(p)=\langle f\rangle}2^{-|p|},
\qquad
-\log_2m_{\mathsf M}(\langle f\rangle)
=K_{\mathsf M}(\langle f\rangle)+O(1).
\]

这里的经典编码定理只说明算法概率与经典复杂度的关系；本项目的 \(K^{N,\mathrm{FD}}\) 与 \(K^{N,\mathrm{full}}\) 都保留“实现质量聚合后取负对数”的结构，但本稿没有证明它们与经典Kolmogorov复杂度相等。

这条coding theorem见[Gács讲义1.6.4至1.6.5节](https://www.cs.bu.edu/faculty/gacs/papers/ait-notes.pdf)。常数与固定机器和编码有关，统一界不随 \(f\) 变化；它不是神经loss扫描中的可调混合修正。Neural K保留“实现质量聚合后取负对数”的结构，但本稿未给出二者相等的神经编码定理。

Chaitin停机概率 \(\Omega_{\mathsf M}^{\rm halt}=\sum_{\mathsf M(p)\downarrow}2^{-|p|}\) 是另一个对象；不能用它替换函数分支质量 \(\Omega_{\Pi,U,D}\)，也不能从其不可计算性直接推出有限网络的agreement不可计算。具体不完备性表述和前提见[已核查的AIT记录](20260920_ait_omega_mixture_agreement_audit_zh.md)，不进入本稿核心定理。

MDL须先给出编码模型。例如两段式码长是 \(\mathcal L(f)+\mathcal L(D\mid f)\)。3节的负log质量首先是事件信息成本；对互斥cell的概率或次概率质量，可以联系Kraft预算与取整后的前缀编码。一般full-target事件可重叠，其负log质量不能未经归一化或另行编码就当作区分全部函数的理想码长。7节的最低可行成本是受误差约束的优化问题；它没有证明SGD会取到该最小值。

## 12. 可保留的实质与验证范围

定义与概率恒等式负责准确描述：研究的是哪个事件、哪个分布、哪个条件更新。经验主张包括：不同函数有不同的loss质量曲线、一些收紧方向增强简单性偏置、幅度对齐改变候选质量、若干静态测量能预测训练偏好。这些主张各自需要实验，不能由重命名推出。

“复杂度分布会变窄”“唯一规则最后必然占优势”“收缩系数等于经典Kolmogorov复杂度”“静态分布等于SGD分布”均不属于定义必然成立的结果。

本轮核验对象是公式对应的有限概率模型、条件事件和代数误差界；它帮助发现定义与推导错误，不证明神经网络普遍满足任何经验前提。核验脚本与结果见逐项审计。
