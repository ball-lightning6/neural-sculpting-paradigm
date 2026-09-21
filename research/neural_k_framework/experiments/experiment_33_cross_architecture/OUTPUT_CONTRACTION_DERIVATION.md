# 跨架构输出体积收缩：Gaussian代价与输出放大阶数

日期：2026-09-13。状态：面向现有实验协议的条件数学推导与确定性数值核验；没有新训练、SMC或网络先验统计。证明尚未外部审阅或形式化验证，不宣称大偏差理论、Gaussian混合或相关范数恒等式的首创。

## 0. 本轮实际推进了什么

可以不先定义不可观测的最细微管线，而从两件明确的东西出发：

1. 白化Gaussian参数的负对数先验代价为`||theta||²/2`。
2. 网络怎样把参数放大转换为指定输入集合上的联合输出放大。

在下文给出的条件下，能够证明：

\[
V_{A,s}(a)=\exp\{-C_{A,s}a^{2/r_A}+o(a^{2/r_A})\}.
\tag{1}
\]

这里V是所有指定输出都具有至少a的正确方向margin的**联合事件质量**；r是适当参数块的输出放大阶数；C由一个独立的最小代价问题给出，不是根据曲线反过来定义的拟合系数。

特别地，实际采样代码中“最后的LayerNorm可随机变化的affine参数 + Gaussian线性读出”属于二阶放大结构，满足可行性条件时得到`-ln V(a) ~ C a`。有界tanh/RNN/LSTM隐藏输出加Gaussian线性读出则得到`-ln V(a) ~ C a²`。

这解释了一大类架构为什么存在有规律的收缩形式，但没有证明所有架构同一个幂次，也没有证明C自动等于算法信息论复杂度，或现有有限窗口已经到达最终渐近区。

## 1. 先固定对象，避免把不同曲线混起来

固定有限输入集合X、架构A及Gaussian参考测度。令z=F_A(theta)是所有待观察raw logits拼成的m维向量，s_i取正负1。

本文主要对象：

\[
V_{A,s}(a)=P\{s_iF_A(\theta)_i\ge a,\ i=1,\ldots,m\}.
\tag{2}
\]

另一个对象是等幅点密度`D_s(a)=p_Z(as)`；其存在和渐近形式需要额外正则条件，后文单独处理。二者的系数一般不同，不能因为都是联合输出问题就混称。

现有跨架构首轮实测主要是16个固定8bit输入、128个格点输出的平均BCE条件质量K_D，不是全部256输入约束的完整规则K_f，也不是等幅点密度。部分架构有完整低loss窗口，部分仍截断；不能把十种架构全部写成精密深尾验证。[原始审计](E:/code/clean_github/neural-sculpting-paradigm/research/notes/20260912_cross_architecture_rule_volume_results_zh.md)

旧3→16→1 tanh等幅密度在a约3～6的窗口接近一次律，但那个模型与跨架构实验中的GELU+LayerNorm MLP不同。[旧输出重分析](E:/code/clean_github/neural-sculpting-paradigm/research/notes/20260910_output_amplitude_old_data_analysis_zh.md)

## 2. 一个统一的条件命题

### 命题A：紧支撑内部表示与渐近齐次读出

设H取值于紧集K，其概率分布的支撑为K；U为独立的p维标准Gaussian。H可以由任意复杂的随机网络主体产生，**并没有把主体固定为一个隐藏层快照**。

假定联合输出可写为F(H,U)，存在连续函数G，使对每个有限M都有

\[
\sup_{h\in K,\|u\|\le M}
\left\|\frac{F(h,Ru)}{R^r}-G(h,u)\right\|\to0,
\quad R\to\infty,
\tag{3}
\]

且`G(h,tu)=t^r G(h,u)`，r>0。定义

\[
C_s=\inf_{h\in K,u}
\left\{\frac12\|u\|^2:\ s_iG(h,u)_i\ge1\ \forall i\right\}.
\tag{4}
\]

假定该约束可行。由紧性、连续性及正齐次性，有`0<C_s<infinity`。那么

\[
\lim_{a\to\infty}\frac{-\ln V_s(a)}{a^{2/r}}=C_s.
\tag{5}
\]

这是一种Gaussian大偏差/收缩原理的具体形式。下面给出不依赖复杂网络细节的直接证明。

### 上界：小于必要范数的参数不能产生足够大的输出

令`R=a^(1/r)`。在有界u区域，若`F(h,Ru)/a`的所有正确方向分量至少为1，由(3)，G的分量至少为`1-delta`。由齐次性和(4)，

\[
\tfrac12\|u\|^2\ge C_s(1-\delta)^{2/r}.
\]

选取足够大的固定M，区域外的u范数本来就超过这个下界。因此整体事件包含于

\[
\{\|U\|\ge R\sqrt{2C_s}(1-\delta)^{1/r}\}.
\]

固定维数Gaussian的径向尾满足`-ln P(||U||>=R b)=R²b²/2+o(R²)`，故

\[
\liminf\frac{-\ln V_s(a)}{a^{2/r}}
\ge C_s(1-\delta)^{2/r}.
\]

### 下界：一个近乎最低代价的可行实现及其邻域已经贡献足够质量

选h_0,u_0使代价接近C_s。把u_0稍微放大，令所有margin严格大于1。连续性保证存在h_0在K中的相对邻域和u_0附近的球，使G在这整个邻域严格可行；(3)保证大a时F也可行。

H邻域的概率是一个与a无关的正数。U/R落入上述球的概率至少是`exp[-R²(C_s+small error)+o(R²)]`。于是得到对应的limsup上界。最后令放大幅度、球半径和代价误差趋零，得到(5)。

**为什么不是忽略隐藏层？** 它仍通过K决定哪些单位输出形状可实现，并通过邻域概率影响有限幅度质量。只是固定架构、a真正趋无穷时，这个与a无关的概率在主导指数尺度上不贡献成本。即使该概率极小，仍不能在有限幅度下忽略它。

若主体也是Gaussian参数eta的连续映射，K为其表示的支撑闭包，上述证明也说明：全参数最小代价

\[
I_s(a)=\inf_{s_iF_A(\theta)_i\ge a}\tfrac12\|\theta\|^2
\]

在这些条件下满足`I_s(a)/a^(2/r) -> C_s`。因此这里的C是独立的参数优化量，不是把测得的体积斜率重新命名。

## 3. 整个网络齐次时，公式尤其简单

若整个白化参数输出映射满足`F(t theta)=t^r F(theta)`，令H为常量、U为全部参数即可。此时

\[
C_s=\min_{s_iF(\theta)_i\ge1}\tfrac12\|\theta\|^2,
\qquad
K_s(a)=\frac{C_s}{\ln2}a^{2/r}+o(a^{2/r}).
\tag{6}
\]

- 一层线性读出：r=1，二次幅度代价。
- L层无偏置线性网络或ReLU网络：r=L，得到`a^(2/L)`。
- 参数共享不会破坏齐次性，但会改变最小代价C。例如一个参数重复使用L次与L个独立参数相乘，虽然放大阶数相同，达到单位输出的Gaussian代价不同。
- 残差、偏置或随机归一化参数不能简单按层数数r；需要检查真正的渐近主项及其可行性。若最高阶主项不能实现该方向，不能用该阶数冒充答案。

### 为什么常数斜率也可以出现

精确齐次且零均值Gaussian时，可进一步证明在坐标`b=a^(2/r)`上的局部斜率趋于C_s。写`theta=R omega`，其中R²服从chi-square，omega为独立球面方向；令`g(omega)=min_i s_i F(omega)_i`。

在g>0的方向，事件等价于`R² >= b/g(omega)^(2/r)`。其b边界密度为

\[
f(b)=c_p b^{p/2-1}\int q^{p/2}e^{-bq}\,\sigma(d\omega),
\quad q=\frac1{2g(\omega)^{2/r}}.
\]

q的最小支撑点正是C_s。指数加权后q向C_s集中，故`d ln f/db -> -C_s`。积分尾的hazard亦趋C_s，即`d[-ln V]/db -> C_s`。对于一般仅渐近齐次的版本，(5)本身不自动保证每个局部窗口斜率都收敛，需要额外正则控制。

## 4. 这不是只能说单输出：等幅联合密度的版本

对精确齐次F，若m不大于p、目标s的fiber具有适当coarea表示，令

\[
\nu_s(du)=\frac{d\mathcal H^{p-m}(u)}{J_F(u)},\qquad F(u)=s.
\]

假定该表示合法、加权积分在某个正温度有限，且最低代价附近有正的fiber测度，则

\[
p_Z(as)=(2\pi)^{-p/2}a^{p/r-m}
\int_{F(u)=s}e^{-a^{2/r}\|u\|^2/2}\nu_s(du).
\tag{7}
\]

由Laplace原理，`-ln p_Z(as) / a^(2/r)`趋于fiber上的最低Gaussian代价。这里用的是**等幅fiber**，不是所有margin至少为1的区域，因此一般

`C_margin(s) <= C_equal(s)`。

等号需要条件，不能把平均BCE、最低margin和等幅密度的系数直接互换。共享输出头、ReLU原子质量或结构约束也可能让普通全维密度不存在，此时应直接研究事件质量，而不是给协方差随意加jitter。

## 5. 为什么带末端归一化的不同架构会有一次幅度律

### 一个不依赖主体结构的确定性界

LayerNorm在affine变换之前，设归一化组有d个数，则

\[
h_j=\frac{x_j-\bar x}{\sqrt{v+\varepsilon}},\qquad
\sum_j h_j^2=\frac{d v}{v+\varepsilon}\le d.
\tag{8}
\]

因此，在有限输入集合上，把这些标准化值拼成H后，其支撑闭包是紧的。GroupNorm有同类组内界。对已经有界的H施加固定池化或ReLU仍然有界；affine变换之后的ReLU则作为后续齐次主项的一部分处理。

我们的SMC代码令最后归一化的gain和bias分别服从`N(1,0.1²)`及`N(0,0.1²)`，线性读出及bias是独立Gaussian。把这些末端参数写成白化U，主体归一化值记H。典型输出形式为

\[
Z=W\{(\gamma_0+D_\gamma u_\gamma)\odot H+D_\beta u_\beta\}+b.
\tag{9}
\]

当这些末端Gaussian坐标统一放大R时，主项是R²乘一个双线性输出映射；gain的均值1、单独的输出bias等只贡献R阶或更低项。由于H有界，收敛在H及有界U上是一致的，满足命题A，r=2。

于是，在双线性主项可以实现指定共同margin的条件下：

\[
-\ln V_s(a)=C_s a+o(a).
\tag{10}
\]

主体可以是MLP、attention、卷积或其复杂组合。共同结构是**最后的有界标准化表示、仍参与Gaussian参考测度的affine参数、以及Gaussian线性读出**，并不是要求主体采用同一种计算。

系数为

\[
C_s=\inf_{H\in\operatorname{supp}(H_A),u}
\{\tfrac12\|u\|^2:s_iG_A(H,u)_i\ge1\}.
\tag{11}
\]

对线性读出后的共同bias，满行秩读出加自由归一化bias通常可将其吸收，帮助验证主项可行性。含ReLU、共享读出或特殊结构时，应具体检查，不能无条件认定每条规则都可行。

**关键适用条件：** 末端参数必须与H独立。输出权重和输入embedding绑定的模型不直接落入此命题。默认训练初始化把归一化gain固定为1，也不等于本项目采样时的Gaussian参考测度。

### 一个明确、可检验的数学预测

若把最后一次真归一化的affine gain与bias都固定，而之后仅剩一个Gaussian线性读出，则特征有界，r降为1，真正远尾变成二次幅度代价。不能只固定gain、却忘记随机bias与读出相乘还能放大常数输出。

这说明归一化参数是否参与参考测度是实质条件，不是可以忽略的实现细节。本轮仅提出该预测，没有新增网络实验。

## 6. 一个完全可积的联合分布：一次律怎样从平方代价产生

取`Y ~ N(0,Sigma)`、`G ~ N(0,tau²)`独立，输出向量`Z=G Y`。这里Y不是把整个真实网络宣称为Gaussian，而是一个可精确求解的共享增益模型。

令`Q_s=s^T Sigma^-1 s`，条件密度的主要代价为

\[
\frac{G^2}{2\tau^2}+\frac{a^2Q_s}{2G^2}.
\]

对G优化，最优`G²=a tau sqrt(Q_s)`，总代价为

\[
\min_G\left(\frac{G^2}{2\tau^2}+\frac{a^2Q_s}{2G^2}\right)
=\frac{a\sqrt{Q_s}}{\tau}.
\tag{12}
\]

直观上，并不需要某一组权重独自扩大a倍；两组各扩大sqrt(a)倍，就能产生a倍输出，而Gaussian平方代价变成a量级。

积分不只给近似。若输出维数为m，令`nu=(1-m)/2`，则精确有

\[
p_Z(as)=\frac{2(a^2Q_s\tau^2)^{(1-m)/4}}
{(2\pi)^{(m+1)/2}\tau|\Sigma|^{1/2}}
K_{(1-m)/2}\left(\frac{a\sqrt{Q_s}}{\tau}\right),
\tag{13}
\]

其中K是修正Bessel函数。其大参数渐近给出

\[
-\ln p_Z(as)=\frac{a\sqrt{Q_s}}{\tau}
+\frac m2\ln a+O(1).
\tag{14}
\]

这是一个真正多输出联合密度的收缩公式。不同方向的系数由Sigma和s算出，而不是从密度曲线上拟合。

若gain为`1+tau V`，平方代价的最优正gain g满足

`g³(g-1)=tau² a² Q_s`。

其代价在小幅度时接近Gaussian二次形式，大幅度时为

\[
I_s(a)=\frac{a\sqrt{Q_s}}\tau
-\frac{\sqrt a Q_s^{1/4}}{\tau^{3/2}}+O(1).
\tag{15}
\]

因此有限窗口内的幅度斜率可以仍在增加，随后趋于常数。这个例子说明为何不能把“已经接近线性”误写成“导数早已严格恒定”；它不是对整个真实LayerNorm网络系数的数值拟合。

## 7. 有界tanh与RNN/LSTM为什么不同

若末层前的特征H有界，而所有无界放大只来自独立Gaussian线性读出U，则直接用命题A的r=1版本：

\[
-\ln V_s(a)\sim C_s a^2,
\quad C_s=\inf_{H,u:s_iB(H)u\ge1}\tfrac12\|u\|^2.
\tag{16}
\]

没有固定隐藏层；H仍遍历其整个可达支撑。只要联合要求可行，特征的统一上界保证C_s严格为正。

这适用于旧3→16→1 tanh模型，以及当前compact RNN/LSTM在有界hidden state后的Gaussian线性头。**所以旧tanh曲线在a约3～6时近似一次，不能无条件外推为a趋无穷时仍一次。** 它可以是重要的有限幅度规律，但不是这个有界架构的最终指数。

## 8. 对当前十种架构的适用性

依据项目[物理模型与参数先验实现](E:/code/clean_github/neural-sculpting-paradigm/research/function_information_conservation/experiment_cross_architecture_output_volume.py:159)及其`parameter_prior`，并检查本地torchvision末端读出结构：

| 架构 | 相关末端结构 | 本轮可以给出的判断 |
|---|---|---|
| compact MLP | GELU、LayerNorm、线性头 | 命题A的r=2结构；可行方向有一次幅度主导式 |
| Symbol Transformer | 最终LayerNorm、共享线性头 | r=2结构；共享头不妨碍联合事件版本，但条件Gaussian可退化 |
| ViT | encoder最终LayerNorm、线性分类头 | r=2结构，需符合独立参数假设 |
| ConvNeXt | 池化后LayerNorm2d、线性头 | r=2结构 |
| Swin分类器 | 最终LayerNorm、池化、线性头 | r=2结构 |
| RNN | 最终tanh hidden、线性头 | r=1，二次幅度主导式 |
| LSTM | sigmoid门乘tanh memory、线性头 | hidden有界；r=1，二次幅度主导式 |
| 改进U-Net | 最后GroupNorm、ReLU、1×1卷积 | r=2形式可用，须核实特定目标的主项可行性 |
| 普通U-Net | 采样时BatchNorm在eval模式 | 不是输入依赖的有界标准化；需按其实际渐近主项分析 |
| Swin-Unet | 编码器后还有卷积/BN/转置卷积链 | 本轮未证明完整主项及其可行性，不能直接套最终LayerNorm结论 |

普通U-Net中的BatchNorm buffers固定，在eval模式下只是使用固定统计量的affine变换。不能因为名字含Norm，就把式(8)套上去。随机BN gain还可能构成额外的乘法放大路径。

以上是结构适用表，不是宣称十种架构的原实测斜率全部已被定量解释。当前实验中的曲线系数仍有明显有限窗口变化。

## 9. 怎样回到平均BCE，且不假定每个样本等幅

记`ell(z)=ln(1+exp(-z))`，`a(epsilon)=-ln(expm1(epsilon))`。m个非负BCE项给出精确包含关系：

\[
V_{\min}(a(\epsilon))
\le P\{\operatorname{mean}BCE\le\epsilon\}
\le V_{\min}(a(m\epsilon)).
\tag{17}
\]

当epsilon趋零时，`a(epsilon) ~ ln(1/epsilon)`，而`a(m epsilon)=a(epsilon)-ln m+o(1)`。若

`-ln V_min(a) ~ C a^p`，则由夹逼有

\[
-\ln V_{\mathrm{mean\ BCE}}(\epsilon)
\sim C\left(\ln\frac1\epsilon\right)^p.
\tag{18}
\]

尤其p=1时，

\[
V_{\mathrm{mean\ BCE}}(\epsilon)=\epsilon^{C+o(1)},
\qquad K_D(\epsilon)=C\log_2(1/\epsilon)+o(\log(1/\epsilon)).
\tag{19}
\]

这给出了“BCE深度坐标上近似恒定收缩系数”的理论入口，**不要求平均BCE等同于所有样本同步等幅**。但它是主导渐近结论，不保证每个有限窗口的局部导数已经稳定。m=128时，ln(m)约4.85，对当前a约7的有限区间绝不是可随意忽略的小偏移。

上述幂律外形与SLT有限极小值的小loss体积形式相似，但这里可以来自参数无穷远处的Gaussian大偏差，不能直接把C命名为原始BCE极小值的RLCT。

## 10. 与以前Gaussian混合推导的衔接

所有有独立零均值Gaussian线性读出的架构，都能在测度层面写成

`Z | eta ~ N(0, B_eta B_eta^T)`，再对全部主体eta积分。

在条件协方差满秩、积分和求导成立时，等幅密度为

\[
D_s(a)=\int A_\eta e^{-tQ_s(\eta)}\mu(d\eta),
\quad t=a^2/2,
\]

其中`Q_s(eta)=s^T Sigma_eta^-1 s`是该表示下单位等幅输出的最小白化读出范数平方。于是

\[
\frac{d[-\ln D_s]}{dt}=E_t Q_s,\qquad
\frac{d^2[-\ln D_s]}{dt^2}=-\operatorname{Var}_t(Q_s)\le0.
\tag{20}
\]

这与用户的实现成本分布图像有直接联系；它不是只计算一个冻结主体。若Q的最小支撑点q_*正，则最终为`-ln D_s ~ q_* t`。若q_*=0，主导式可能是t的分数幂。

例如低Q区域的加权密度具有`exp(-b/Q^gamma)`级稀缺代价时，平衡`tQ+b/Q^gamma`得到`t^(gamma/(gamma+1))`；gamma=1对应a的一次律。这是解释不同放大机制的另一种写法，但没有测得该Q分布时，不能拿一个自由分布去任意拟合后就宣称证明了原网络。

共享头造成条件协方差退化时，式(20)的全维密度版本不自动成立；式(2)及命题A不要求条件协方差满秩，因此更适合跨架构统一论述。

## 11. 一个必须保留的边界：主导系数未必严格区分复杂度

不能从上述定理推出“独立定义的Kolmogorov复杂度越高，任何架构下C一定更高”。一个明确例子是足够宽的两隐藏层tanh网络。

在固定Boolean输入集合上，第一隐藏层宽度至少为输入点个数，可用顶点分离的tanh单元构造近似查表特征；第二隐藏层的m个单元都能趋近于同一符号函数s_f。所有参数仍采用full-support Gaussian先验，复制单元的附近是正质量区域，不要求精确相同参数拥有正概率。

若输出为`z=(v·h+b)/sqrt(m)`，v和b为标准Gaussian，对任意非恒定标签，正负输入两点满足

\[
2a\le\frac{v\cdot(h_+-h_-)}{\sqrt m}\le2\|v\|,
\]

所以单位margin读出代价至少1/2。若所有末层隐藏单元都趋近s_f，则取`v_j=1/sqrt(m)`、b=0，可将代价逼近1/2。由有界读出命题，**所有可这样实现的非恒定规则，真正a²远尾主导系数都为1/2**。

恒定规则还可利用输出bias，系数为`m/[2(m+1)]`。复杂规则与简单非恒定规则的差异可以仍存在于极小的主体先验质量、次阶项及巨大有限区间中，但不能保证主导系数严格区分它们。

这不是旧单隐藏层3→16→1的直接结论。它说明“存在收缩律”与“主导系数提供不退化的算法复杂度排名”是不同命题。

## 12. 本轮数值核验

核验脚本没有训练网络，也没有估计原架构的新SMC体积：

- 96组维数、幅度、Q、增益方差组合，式(13)与独立log-gain积分的最大对数密度差4.55e-13。
- 一个实际两输入、四Gaussian参数的模型`f(x)=u1*v1+0.5*u2*v2*x`，x取正负1。共同margin的最小代价分别为常数规则1、copy规则2，线性规划与显式因子分解核对一致。a=1000时确定性积分得到`-ln V/a`约1.00464、2.00545，趋向上述两个系数。
- 后一积分的初步192/256阶二维Laguerre结果，对数差最大约4.77e-4；随后针对外层的可积对数奇点改用自适应积分，内层96/128阶的对数结果差约3.55e-15。最终表值使用这一改进积分。仍不称为区间算术认证。
- 矩阵双线性分解的标准恒等式`inf_{UV=M} (||U||_F²+||V||_F²)/2=||M||_*`，SVD构造误差约1.78e-15。它是上述最小代价能对应明确函数/算子范数的另一例子。
- LayerNorm确定性上界在跨多个输入尺度的代数测试中通过。
- 显式3→8→16→1 tanh查表构造中，copy与parity在a=10000时总Gaussian代价除以a²都约0.500166，与第11节匹配下界一致。这是可行构造的代价，不是体积采样值。
- 对`U`、`U*tanh(V)`、`UV`、`U*(1+0.1V)`的单边尾做独立确定性积分，分别观察到二次、二次、一次、由中等幅度向一次过渡的渐近形状。

![确定性积分与放大结构](E:/code/clean_github/neural-sculpting-paradigm/tmp/output_contraction_math_20260913/tail_mechanisms.png)

- [核验脚本](E:/code/clean_github/neural-sculpting-paradigm/research/function_information_conservation/verify_output_contraction_derivation.py)
- [结构化结果](E:/code/clean_github/neural-sculpting-paradigm/tmp/output_contraction_math_20260913/verification.json)
- [积分曲线](E:/code/clean_github/neural-sculpting-paradigm/tmp/output_contraction_math_20260913/curves.csv)

## 13. 文献定位与不应越过的引用边界

Vladimirova等2019年给出Gaussian参数、ReLU-like激活下单元随深度变重的尾部结论。这支持乘法组合改变尾部幂次的方向，但不是本项目多输入、逐规则系数的完整计算。[论文](https://proceedings.mlr.press/v97/vladimirova19a.html)

该文Remark 3.2对归一化尾部作了宽泛说明；不能据此把固定affine的逐样本LayerNorm也认定为保留相同最优尾指数，因为式(8)直接给出有界性。本文按实际是否采用输入依赖统计、是否随机化affine参数区分，而不把那句话当作当前协议的定理。[正文](https://proceedings.mlr.press/v97/vladimirova19a/vladimirova19a.pdf)

Noci等以及Zavatone-Veth、Pehlevan在2021年研究有限宽Gaussian参数网络的精确先验分布，说明有限网络不能直接被单一Gaussian代替；后者明确限定主要结果为单个输入。本文沿这些已知概率工具推导指定有限输入集合的margin事件和条件系数，不宣称这些经典工具本身是新发现。[Noci等](https://arxiv.org/abs/2106.06615)、[Zavatone-Veth与Pehlevan](https://arxiv.org/html/2104.11734v3)

仍未完成：对全部十种真实模型逐一认证完整主项；直接算出原规则的系数数值；证明已测有限窗口就是严格渐近区；证明系数与独立算法复杂度等价；从静态结论推出SGD必然按该质量采样。最近Mod3的可达性结论不因本推导消失。

**可以使用的总结：在明确先验和放大结构下，跨架构输出收缩律有统一的概率-几何来源；架构决定放大阶数及最小代价，Gaussian先验把代价转成指数质量。存在这类定律并不神秘，但它的幂次、有限幅度修正以及“复杂度”含义必须具体区分。**
