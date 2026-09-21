# 有限宽 CGF 预测器：推导与适用边界

## 原模型

原 3bit/4bit 实验均为单隐藏层宽 m=16 的 tanh 网络。输入为 r=2^b 个 0/1 向量，Gaussian 神经坐标相互独立，标准差均为 1。联合输出写成

\[
Z=\frac1{\sqrt m}\sum_{j=1}^{m}a_j h_j
+\frac{b_o}{\sqrt m}\mathbf1,
\qquad
h_j(x)=\tanh\left(\frac{w_j\cdot x+b_j}{\sqrt b}\right).
\]

积分掉独立的 Gaussian 读出 a_j 和 b_o 后，有限宽网络的联合累积量生成函数是

\[
\Lambda(t)=\log\mathbb E e^{t^TZ}
=\frac{(\mathbf1^Tt)^2}{2m}
+m\log\mathbb E_{w,b}\exp\left[\frac{(t^Th)^2}{2m}\right].
\tag{1}
\]

这一步是精确恒等式，不使用无限宽近似，也没有从 SMC 数据标定参数。单神经元的积分只有 b+1 个独立 Gaussian 坐标，程序用多维 Gauss-Hermite 数值积分；整体反号对称性将节点数减半。积分节点不是 SMC 粒子。

与平均 Gaussian 近似的区别：后者只保留 Lambda 的二次展开，式 (1) 则保留有限宽表示涨落的所有阶。本文受到 [Fischer 等的有限网络 Gaussian 混合分解](https://proceedings.mlr.press/v235/fischer24a.html) 启发，进一步利用本项目单隐藏层的独立性简化计算；不是声称论文原样提供了这个 BCE 预测器。

## BCE 事件的凸对偶

标签符号 s_i=2y_i-1，事件 E 为 r 个输入上平均 BCE 不超过 epsilon。定义

\[
I(E)=\sup_t\{\inf_{z\in E}t^Tz-\Lambda(t)\}.
\]

由 Chernoff 不等式有 Pr(E)<=exp[-I(E)]，即 K(E)>=I(E)/ln(2)。I 本身不是完整 K。

对 BCE，可令 t=s*omega、omega_i>=0，引入 alpha>max(omega_i)，得到有限维凹最大化：

\[
I(E)=\sup_{\omega,\alpha}
\left\{\alpha\sum_i H(\omega_i/\alpha)
-r\alpha\epsilon-\Lambda(s*\omega)\right\},
\tag{2}
\]

其中 H 是二元熵，使用自然对数。对代码中固定有限正求积权重，Lambda 仍为凸函数，因此不是任意核自由拟合标签的过程。最优输出与乘子满足

\[
z_* = \nabla\Lambda(t_*),\qquad
s_i z_{*,i}=\log(\alpha/\omega_i-1),\qquad L(z_*)=\epsilon.
\]

程序同时检查平均 BCE、KKT 残差和 Legendre 恒等式。数学上的 Chernoff 界是严格的，但浮点求积尚未经区间认证，所以输出不能称为已认证的严格数值界。

约束始终是原来的平均 BCE，没有换成固定输出幅度、MSE 或最大 BCE，也没有加入幅度对齐正则。初值采用均匀方向只是求解初始化，最终各样本的 logits 可以不同。

## 表示代价确实被计入

令 u=t^T h，psi=log E exp[u^2/(2m)]，nu_t 为按此指数倾斜后的单神经元分布。则

\[
I=\underbrace{m\operatorname{KL}(\nu_t\Vert\nu_0)}_{\text{隐藏表示倾斜代价}}
+\underbrace{\tfrac12\mathbb E_{\nu_t}u^2
+\tfrac1{2m}(\mathbf1^Tt)^2}_{\text{读出倾斜代价}},
\]

其中第一项等于 E(u^2)/2-m*psi。它是指数倾斜后的整体参数分布相对原先验的 KL 分解，不是忽略表示罕见程度的任意最小范数解。

## 从作用量到概率近似

在最优指数倾斜分布下，Z 的均值是 z_*，协方差是 Sigma_*=Lambda''(t_*)。以下才开始引入近似：用局部 Gaussian 描述倾斜后输出，用 BCE 边界的二阶展开计入切向曲率。

设 A A^T=Sigma_*、a=||A^T t_*||，T 为垂直于 A^T t_* 的正交切向基，lambda=r*alpha。定义

\[
B=I+\lambda T^TA^T\nabla^2 L(z_*)AT.
\]

本轮在读取完整比较结果前固定的概率近似是

\[
\widehat{\log P(E)}=-I+\frac{a^2}{2}+\log\Phi(-a)-\frac12\log\det B.
\tag{3}
\]

其中平直边界的因子来自 Gaussian 半空间积分；非线性边界和非 Gaussian 倾斜分布仍有近似误差。式 (3) 不是严格恒等式，不能从求积收敛推导出它对真实体积的误差界。

程序还保留不加式 (3) 修正的原始 Chernoff 作用量，防止把主导代价和完整概率混为一谈。未通过全空间参数重要性采样重新估计 P(E)，因此这是一轮理论近似对历史 SMC 的回测，不是换个采样器重测后自我比较。

## 技术核验

- CGF 零点、偶对称性、梯度反对称性。
- 梯度与 Hessian 的有限差分核验。
- 零点 Hessian 对照原先验的高精度协方差积分。
- 将激活临时设为线性，仅用于与有精确 CGF 的数学模型做单元测试；正式预测仍然全部 tanh。
- 将 CGF 替换为精确 Gaussian，仅用于对照旧 BCE 凸优化器。
- 五个命名规则的不同初值、输出反号核验。
- 正式所有节点提高求积阶数检查，不按与 SMC 拟合好坏选择精度。

## 什么尚未成立

- 没有证明式 (3) 在任意有限宽、任意规则和 loss 深度上误差小。
- 没有把 K 的小相对误差等同于概率的小相对误差；1 bit 的 K 误差就是质量倍率 2。
- 这里依赖单隐藏层中不同神经元的先验独立性。多层、CNN/RNN 共享参数和 Transformer 不能直接沿用式 (1) 的乘积形式。
- 这不是 Kolmogorov 复杂度的直接计算，也不证明 SGD 严格按该分布采样。
