# Neural K 理论核心

> 本页只写最终理论框架，不重述研究过程。实验编号仅作为证据锚点。

## 一句话版本

固定网络架构、编码、参数测度和损失以后，神经网络本来就更容易表示某些简单函数；继续降低训练 loss，又会让不同函数及其内部实现以不同速度失去参数体积，通常使神经网络相对复杂度更低、可复用程度更高的描述获得优势。因此，在结构化数据上，把训练 loss 压得更低，往往等价于要求网络用更经济的程序描述训练集。数据决定哪些程序仍与样本相容，网络结构决定什么叫“经济”，优化器决定这些低-loss 程序能不能从初始化真正到达。

## 1. 两种简单性是一条原则的两个尺度

### 1.1 先验简单性

随机抽取网络参数时，不同函数的概率高度不均匀。一个函数若能由更多参数配置实现，它在网络诱导的函数先验中就有更大质量。大量已有研究和本项目的先验采样都表明，结构较简单、与架构更对齐的函数通常具有更大先验质量。

这不是说所有人类看来简单的函数都更容易。Parity 就是典型反例：它在人类的异或语言里很短，在普通 tanh MLP 的表示语言里却可能很昂贵。

### 1.2 Loss-resolved 简单性

这里所说的 loss-resolved 简单性，最干净的定义和直接证据是：**对每个候选函数使用它的全部输入输出样本**，再分别测量该函数的 full-target loss-volume profile。对有限布尔空间，这意味着使用完整真值表；对不能穷举的空间，则必须预先规定一个足够完整、对所有候选一致的目标集。它不是从一小部分训练样本出发，再把研究者知道的数据生成器自动当成唯一目标函数。

在完整目标下，参数即使已经给出全部正确 hard 输出，仍可处在不同 loss 深度。

> **核心结论：继续收紧 raw loss 时，各函数的可用参数体积不会等比例缩小。较复杂、依赖大量不可复用例外或与架构不对齐的实现，往往在深 loss 中收缩得更快；更经济的实现相对获得优势。**

只给部分训练集 $D$ 时，测量的是“在满足 $D$ 的低-loss 参数中，各种完整延拓分别占多少质量”。这个 fixed-$D$ 候选质量与 full-target volume 有严格关系，但不是同一个量；[E20](experiments/e20.html)通过条件概率和 margin bridge 建立了两者的测度联系。部分训练集下哪个函数占优，还会受到样本覆盖、不平衡和其他兼容延拓影响，不能把完整目标的体积排序未经桥接直接搬过去。

这条经验规律不能写成“某个预先指定的简单函数在所有 loss 区间都单调上升”。[E12](experiments/e12.html)、[E14](experiments/e14.html)和[E24](experiments/e24.html)已经证明，赢家和局部收缩速度可以随 loss 换序。真正稳定的理论对象是一整条复杂度曲线，而不是永久标量。

### 1.3 与“压缩即智能”的关系

普通训练代码显式优化的只有训练 loss，并没有第二项机器无关的程序复杂度。压缩是网络结构、参数测度和 loss 几何产生的结果：若复用特征、中间计算和规则能用更少独立自由度同时满足更多样本，它通常更容易继续达到深 loss。

因此，在干净、结构化、数据充分的任务中，降低 loss 基本上可以理解为不断寻找对训练集更经济的描述。这直接连接“压缩即智能”：智能行为来自发现可复用规律，而不是为每个输入保存独立答案。[E08](experiments/e08.html)给出了共享中间计算的因果证据，早期百余规则实验则提供了广度证据。

但这不是无条件等号。训练早期从常量输出学会任务时，绝对程序复杂度可以上升；面对随机标签或噪声，继续降低 loss 也可能增加例外记忆。

> **核心边界：loss 是显式目标，压缩是结构化数据和有限网络语言下常见的经济路径。**

## 2. 复杂度是网络相对的 Profile

把架构和参数化、输入输出编码、参考参数测度以及逐样本损失合在一起，记作神经参考协议：

$$
\Pi=(\mathcal A,\varphi,\mu,\ell).
$$

对完整目标函数 $f$，定义在 loss 阈值 $\epsilon$ 以下的参数体积：

$$
V^{\mathrm{full}}_\Pi(f;\epsilon)
=
\mu_\Pi\{\theta:L_{D_{\mathrm{full}}(f)}(\theta)\le\epsilon\}.
$$

Neural K-profile 定义为：

$$
K^N_\Pi(f;\epsilon)
=
-\log_2V^{\mathrm{full}}_\Pi(f;\epsilon).
$$

它不是机器无关 Kolmogorov complexity，而是“当前神经语言把函数 $f$ 实现到精度 $\epsilon$ 有多稀有”。更换架构、激活、宽度、编码、初始化尺度或损失，都会改变这条曲线。

复杂度首先是一条 profile：先验概率只是浅层起点，某个固定 loss 的体积只是一个截面，局部收缩率只是斜率。不同曲线可以交叉。并且 hard function 仍然太粗；即使所有输出符号已经相同，margin、logit 与内部表示的连续几何仍可继续收缩和换序。[E24](experiments/e24.html)是直接证据。

### 独立操作量：随机训练集的辨识与恢复相变

Neural K-profile 是完整目标的静态几何对象。实际训练还可以定义一个直接可测、但**不能与 Neural K-profile 偷换**的样本复杂度。为此固定神经参考协议 $\Pi$、训练集抽样分布 $q$，再固定完整训练审计协议 $\mathcal T$：网络容量、初始化、optimizer、训练预算、seed 数、完整 probe 范围和恢复阈值都必须预先写定。

对完整目标函数 $f$ 和样本数 $n$，从 $D_{\mathrm{full}}(f)$ 中均匀随机抽取多份大小为 $n$ 的训练集 $D_j$，每份训练多个独立 seed。定义数据集级恢复判据 $\mathcal R_{\Pi,\mathcal T}(D_j,f)$；它不能只看 validation accuracy，而应同时要求：

1. 绝大多数 seed 已拟合训练集；
2. 已拟合 seed 中，目标完整函数占足够高质量；
3. 目标函数是 modal hard function；
4. 完整函数 collision / agreement 已超过预注册阈值。

样本数 $n$ 下的恢复率为：

$$
\widehat r_{\Pi,\mathcal T}(f,n)
=
\frac{1}{M}
\sum_{j=1}^{M}
\mathbf 1\!\left[
\mathcal R_{\Pi,\mathcal T}(D_j,f)=1
\right].
$$

对恢复率做单调估计后，定义：

$$
n_q^{\mathrm{id}}(f;\Pi,q,\mathcal T)
=
\min\left\{
n\in\mathcal G,\ n<N:
\widehat r_{\Pi,\mathcal T}^{\,\uparrow}(f,n)\ge q
\right\},
\qquad q\in\{0.5,0.9\}.
$$

这里 $N=|D_{\mathrm{full}}(f)|$，$\mathcal G$ 是预注册的非完整样本数网格。`n50` 和 `n90` 分别是网格上随机训练集有 50% 和 90% 概率稳定恢复完整规则的首次 crossing。点估计 `n_q=n_k` 实际只把连续相变定位到区间 $(n_{k-1},n_k]$；应同时报告 bootstrap 区间。若直到最大非完整样本数 $N-1$ 仍未 crossing，则记为 `n_q>N-1` 或 right-censored，不能把 $n=N$ 的完整真值表拟合当作泛化相变。

因此它给出的是数据量轴上的 grokking / 规则恢复相变**区间**。网格相邻、置信区间重叠的规则出现 tie 或一个档位错序，不足以否定统计关系。尤其在 4-bit 这类只有 16 个状态的空间里，一个样本已占 6.25%，holdout 位置和对称性影响很大，最难规则还会直接撞到全空间上限；这时 `n50/n90` 往往不可辨识，只能依靠 full profile 与局部 leave-one-out 旁证。

这个量更准确的名称是**协议相对辨识/恢复样本复杂度**。它以“样本数”为单位，并直接回答“在抽样分布 $q$ 下，当前训练系统需要多少随机约束，才会稳定选择这条完整规则”。它同时包含至少四项：完整目标本身的 Neural K-profile、固定部分训练集下竞争延拓的分母、样本覆盖/捷径造成的可辨识性，以及 optimizer 的可达性。因此它可以作为 Neural K 的外部判据和经验锚点，却不能未经额外假设直接定义为同一个量。

更明确地说，完整目标体积测量的是分子；随机训练集相变测量的是这个分子在 target-dependent 竞争分母和 optimizer 运输中何时占优。只有当这些额外因素跨任务受控，或者已经证明保持单调时，`n50/n90` 排序才能被当作 full-target Neural K 排序的代理。

### 这些实验怎样修正了原来的结论

我们最初有一个很自然的猜想：**网络认为越复杂的规则，就需要越多训练样本才能被稳定识别。** 对 parity1 到 parity4 的预注册实验严格符合这个顺序，说明完整规则的 Neural K 确实具有前瞻预测力，而不只是看完结果后的重新命名。

随后，parity2 和 MUX3 出现了一个无法忽略的反例。按完整规则体积看，MUX3 更容易；但用普通随机样本训练时，它反而需要更多数据。这迫使我们承认：只知道目标规则本身有多难，还不够。

关键缺失项是**当前训练集还允许哪些替代解释**。对 MUX3 来说，大量样本既符合真实 selector 规则，也符合“复制某一位，再记住少数例外”的捷径。这样的样本虽然数量不少，却没有真正帮助网络区分这些函数。

第一个补充实验只改变采样方式。当我们增加那些恰好能区分真实 MUX3 和 copy 捷径的样本时，MUX3 立刻变得比 parity2 更早恢复。目标规则和网络都没有改变，改变的只是样本有没有击中竞争函数真正分歧的位置。

第二个补充实验完全拿掉 optimizer，直接观察同一批训练样本允许的低-loss 函数。结果仍然相同：普通样本留下大量错误捷径，而冲突样本会系统地排除它们，让真实 MUX3 成为主导。这说明上述现象首先来自训练集本身塑造的静态函数竞争，optimizer 主要影响到达速度和具体路径。

> **修正后的结论：一个规则需要多少数据，不只取决于规则本身有多难，还取决于当前有哪些函数在和它竞争，以及采到的样本能否有效排除这些竞争函数。完整规则的 Neural K 仍是重要的一阶预测量，但它不等于 grokking 相变点。**

详细数字、三项补充实验和数值边界都放在 [E23](experiments/e23.html)，理论核心只保留这个修正后的因果关系。

[E24](experiments/e24.html)还补充了另一种情况：即使两个完整映射已经固定，随着 loss 继续降低，它们的相对体积仍可能换序。这说明“规则难度”本身也应看整条 loss profile，而不是某一个浅层截面。

### 与 MDL 和算法信息论的关系

标准 MDL 把学习写成两部分码长：

$$
L_{\mathrm{MDL}}(H,D)=L(H)+L(D\mid H).
$$

$L(H)$ 是描述假设或程序的成本，$L(D\mid H)$ 是用该假设编码尚未解释残差的成本。算法概率和 Kolmogorov complexity 则从另一个方向表达同一原则：短程序应获得更大先验质量。

Neural K 把这套语言改写成一个可测、但依赖具体神经协议的版本。参考测度 $\mu_\Pi$ 已经给不同实现分配了不均匀质量；一个完整函数在精度 $\epsilon$ 下拥有的体积 $V^{\mathrm{full}}_\Pi(f;\epsilon)$，对应码长 $-\log_2V^{\mathrm{full}}_\Pi(f;\epsilon)$。因此，Neural K-profile 可以理解为：**当前神经语言要把目标描述到不同残差精度时，需要付出怎样的精度依赖码长。**

这并不表示训练程序显式同时最小化“loss 加程序长度”。训练代码通常只降低经验 loss；MDL 式选择来自架构参考测度与 loss 亚水平几何共同造成的质量差异。它也不是机器无关 Kolmogorov complexity，因为更换网络、编码、参数化或参考测度会改变码长。当前还没有神经版 coding theorem 证明两者等价。

这里可证伪的内容不只是给体积换一个名字：[E22](experiments/e22.html)把逐样本广义 surprise 严格累加为顺序无关的端点码长，[E23](experiments/e23.html)又用预先冻结的 full-target profile 预测了独立随机训练集的 `n50/n90`。这把参数质量、预测编码和辨识样本复杂度接成了一条经验链。

## 3. 静态几何与优化动力学必须分开

固定训练集 $D$ 和 loss 阈值 $\epsilon$，某个完整函数的静态质量为：

$$
\Omega_{\Pi,D}(f;\epsilon)
=
\mu_\Pi\{\theta:L_D(\theta)\le\epsilon,\ h_\theta=f\}.
$$

归一化后得到静态函数分布：

$$
Q^{\mathrm{static}}_{\Pi,D,\epsilon}(f)
=
\frac{\Omega_{\Pi,D}(f;\epsilon)}
{\sum_g\Omega_{\Pi,D}(g;\epsilon)}.
$$

真实优化器从初始化分布出发，经训练映射产生另一个分布：

$$
Q^{\mathrm{opt}}_{\Pi,D,t}
=
(T_{\Pi,D,t})_\#\mu_\Pi.
$$

> **关键分层：静态体积决定哪些候选区域大、哪些区域随 loss 更稳定，是实际训练的重要一阶影响；优化器还决定入口、流管、速度、连通性、数值噪声和历史。因此静态体积不能被直接叫作 SGD 后验。**

Parity scaffold 表明：一个低-loss 终点可以存在、局部稳定，却从随机初始化无法到达。[E17](experiments/e17.html)则表明：同一网络、训练集和相近 loss 下，SMC 与 AdamW 仍可选择明显不同的函数。

[E28](experiments/e28.html)提供了另一侧的真实任务锚点：在完整50k MNIST和同一
小CNN上，整网HMC与plain Adam分别达到99.03%和98.95%，处于同一性能水平，
但函数多样性和时间平均收益并不相同。[E29](experiments/e29.html)又在dead-bit
反事实中给出更严格的matched-loss判决：无衰减Adam与静态SMC的五函数平均差
只有0.0010，而L2/MAP通过主动压缩未约束权重形成另一种运输结果。两项结果共同
说明静态几何可以是强预测底图，但“预测接近”与“分布相等”仍是两回事。

任何任务都应分开检查三件事：网络是否能表示目标，数据是否足以辨识目标，优化器是否能够到达目标。

### 解耦、Scaffold 与优化可达性不等于复杂度

端到端任务的原始目标可以写成 $L_{\mathrm{end}}(\theta)$。解耦训练、中间监督或 scaffold 通常加入辅助目标：

$$
L_{\lambda}(\theta)
=
L_{\mathrm{end}}(\theta)
+
\lambda L_{\mathrm{aux}}(\theta),
$$

并在训练后期把 $\lambda$ 降到零，或完全撤去辅助输入与中间标签。这个操作首先改变的是梯度向量场和进入低-loss 区域的路径：

$$
-\nabla L_{\lambda}
=
-\nabla L_{\mathrm{end}}
-
\lambda\nabla L_{\mathrm{aux}}.
$$

原始端点 loss 可能在随机初始化附近几乎不给出有用方向，或者不同计算阶段的梯度互相抵消。辅助目标把远距离、复合或对称的计算拆成局部可学习步骤，使网络先形成中间表示，再逐步内化为端到端计算。这说明搜索路径得到改善，不等于原始目标函数的 Neural K 必然降低。

若 scaffold 撤除后，网络仍能只靠 $L_{\mathrm{end}}$ 保持解、从扰动中恢复，说明原始端点任务确实存在受自身 loss 支持的稳定低-loss 区域；最初失败主要是入口或梯度可达性问题。[E16](experiments/e16.html)的 parity scaffold—撤除—扰动恢复就是这种判据。Mod 3、乘法、递归/搜索和汉诺塔等早期解耦实验也提供了同方向现象。

反过来，辅助监督成功不能单独证明原始函数很简单；端到端失败也不能单独证明函数很复杂。必须分开四个对象：

1. **表示能力**：网络中是否存在实现目标的参数；
2. **数据辨识性**：训练样本是否足以排除竞争延拓；
3. **协议相对复杂度**：目标在原始神经参考协议中的 Neural K-profile；
4. **优化可达性**：从指定初始化和优化器出发，梯度是否能进入该区域。

若辅助变量、模块或输入在最终系统中永久保留，那么神经参考协议 $\Pi$ 已经改变，目标在新语言下确实可能更简单。若辅助信息只用于训练、最后完全撤除，则它主要是 continuation / curriculum 路径；最终复杂度仍应在原始端点协议下测量。比较解耦与端到端结果时，必须先声明属于哪一种情况。

## 4. 统计物理语言

### 4.1 对应表

| 统计物理 | 神经网络中的对象 |
|---|---|
| 微观态 | 一组具体参数 $\theta$ |
| 宏观态 | 完整 hard function，或进一步细分的 function-margin/representation cell |
| 能量 | 训练集上的总损失 |
| 态密度 | 某个能量附近有多少参数微观态 |
| 熵 | 态密度或累计体积的对数 |
| 温度 | 对训练误差的容忍尺度；不是物理温度 |
| 自由能 | 能量要求与参数熵之间的综合代价 |
| 相变/换相 | 不同函数宏观态的优势随数据量、温度或 loss 深度换手 |

为保持样本可加性，把总能量写成逐样本损失之和：

$$
E_D(\theta)=\sum_{i=1}^{n}\ell(h_\theta(x_i),y_i)=nL_D(\theta).
$$

对函数 $f$ 定义能量态密度：

$$
\rho_{D,f}(E)
=
\int \delta(E-E_D(\theta))\,\mathbf 1[h_\theta=f],d\mu_\Pi(\theta).
$$

Microcanonical 视角直接截取某个 loss 以下的累计体积：

$$
\Omega_{D,f}(\epsilon)
=
\int_{-\infty}^{n\epsilon}\rho_{D,f}(E),dE.
$$

其对数可以看作累计参数熵。一个函数在同样 loss 下拥有更多实现方式，就有更高参数熵和更大静态质量。

### 4.2 Canonical 系综、配分函数与自由能

给定逆温度 $\beta$：

$$
Z_D(\beta)
=
\int e^{-\beta E_D(\theta)}d\mu_\Pi(\theta),
$$

函数限制下的配分函数为：

$$
Z_{D,f}(\beta)
=
\int \mathbf 1[h_\theta=f]e^{-\beta E_D(\theta)}d\mu_\Pi(\theta).
$$

于是函数宏观态的 Gibbs 质量为：

$$
Q_\beta(f\mid D)=\frac{Z_{D,f}(\beta)}{Z_D(\beta)}.
$$

自由能定义为：

$$
F_D(\beta)=-\frac1\beta\log Z_D(\beta),
\qquad
F_{D,f}(\beta)=-\frac1\beta\log Z_{D,f}(\beta).
$$

两个函数的相对赔率由受限自由能差决定：

$$
\log\frac{Q_\beta(f\mid D)}{Q_\beta(g\mid D)}
=
-\beta\bigl(F_{D,f}-F_{D,g}\bigr).
$$

$\beta$ 越大，系综越强调低能量，也就是更低训练 loss。复杂函数若低能态密度更少，其受限自由能会更快变差，质量就被压低。这是“随 loss 下降，复杂函数体积通常收缩更快”的统计物理表达。

Microcanonical 的 loss 截面和 canonical 的 Gibbs 加权描述的是同一个态密度；后者是前者的 Laplace 型汇总。本项目两种口径都做过测量。

### 4.3 这里不是什么物理主张

训练不是一个自动达到热平衡的物理系统，$\beta$ 也不是 GPU 的真实温度。除非另有 Langevin 或平衡假设，SGD/AdamW 不等于 Gibbs 采样器。统计物理在这里首先是一套描述静态态密度、能量—熵竞争和宏观态换手的数学语言。

Grokking 可以表现为规则宏观态在更深 loss 或更大数据量下接管，但有限网络中的平滑 crossover 不必是严格热力学奇点。本文尚未从网络结构解析推导 $\rho_{D,f}(E)$ 的形状；这正是“为什么两种简单性成立”的更深问题，目前不在研究范围内。

## 5. 怎样用体积预测一个未见样本

给定训练集 $D$、未见输入 $x$ 和候选标签 $y$，构造扩展训练集：

$$
D_y=D\cup\{(x,y)\}.
$$

在 microcanonical 口径下，标签分支质量为：

$$
P_{\mathrm{hard}}(y\mid x,D,\epsilon)
=
\frac{\mu_\Pi\{\theta\in A_D(\epsilon):h_\theta(x)=y\}}
{\mu_\Pi(A_D(\epsilon))}.
$$

直观上，在所有已经把训练集做到当前 loss 的参数中，看有多少会把新输入预测成标签 $y$。质量更大的分支就是当前静态预测。

也可以分别计算每个假设标签扩展数据集的配分函数：

$$
\widetilde P_\beta(y\mid x,D)
=
\frac{Z_{D_y}(\beta)}{Z_D(\beta)},
$$

再在全部候选标签上归一化：

$$
P_\beta(y\mid x,D)
=
\frac{Z_{D_y}(\beta)}{\sum_{y'}Z_{D_{y'}}(\beta)}.
$$

当逐样本 loss 是标准负对数似然且 $\beta=1$ 时，这就是普通 posterior predictive evidence。更一般地，它仍是一个协议相对的 Gibbs 分支分数。

> **预测原则：某个标签对应的体积随 loss 深入收缩得更慢，意味着它在越来越严格的训练精度下保留更多兼容实现，因此预测倾向增强。**

不同标签的体积曲线仍可能交叉，所以预测必须声明使用哪个 loss 深度或 $\beta$。MNIST [E25](experiments/e25.html)直接检验了这种静态分支预测。

[E26](experiments/e26.html)把同一原则推进到标签组合层：在只知道一个anchor和
5:5比例时，自然MNIST 0/1划分在两个盲测panel中都从浅loss的末位升到深loss
top1，并在loss 0.3占据约99.97%的归一化体积质量。它支持“体积可预测未见
标签组合”，但不等于取消类别比例约束后的完全无监督聚类。

## 6. Surprise、信息增益与样本顺序不变量

### 6.1 已观察标签的 Surprise

对已经观察到的标签 $y$，其预测惊讶度为：

$$
s(y\mid x,D)=-\log_2P(y\mid x,D).
$$

定义数据集状态成本：

$$
C(D)=-\log_2Z(D).
$$

加入一个样本的广义 surprise 就是端点成本差：

$$
\Delta C_t=C(D_t)-C(D_{t-1})
=
-\log_2\frac{Z(D_t)}{Z(D_{t-1})}.
$$

沿任意样本顺序求和：

$$
\sum_{t=1}^{m}\Delta C_t
=
C(D_m)-C(D_0).
$$

中间项望远镜消去，因此总成本只依赖起点和最终训练集，不依赖样本添加顺序。**每一步的 surprise 仍然依赖顺序；不变的是总和。** [E22](experiments/e22.html)对 256 条规则和每条规则全部 40,320 个顺序进行了数值闭合。

这个不变量属于同一个静态配分函数，不代表 SGD 轨迹守恒。

### 6.2 Hard-conditioning 下：实际 Surprise 精确等于函数分布信息增益

这里的概率不要求样本本身来自一个随机生成器。令当前完整函数为随机变量

$$
F\sim Q_D(f),
$$

而候选输入 $x$ 上的标签只是这个函数随机变量的投影：

$$
Y_x=F(x),
\qquad
P_D(y\mid x)
=
\sum_fQ_D(f)\mathbf 1[f(x)=y].
$$

观察标签 $y$ 并做纯 hard conditioning 后，新的函数分布为：

$$
Q_{D'}(f)
=
\frac{Q_D(f)\mathbf 1[f(x)=y]}
{P_D(y\mid x)},
\qquad
D'=D\cup\{(x,y)\}.
$$

因此，以2为底计算 KL 时有精确恒等式：

$$
D_{\mathrm{KL},2}\!\left(Q_{D'}\Vert Q_D\right)
=
-\log_2P_D(y\mid x)
=
s(y\mid x,D).
$$

> **Hard-conditioning 精确定理：真实标签在当前函数分布下越不可能，它删除的候选函数质量越多；已观察标签的预测 surprise，精确等于函数 posterior 更新的信息增益。**

这就是 [Itti 与 Baldi 的 Bayesian surprise](https://pmc.ncbi.nlm.nih.gov/articles/PMC2782645/) 在完整函数空间中的具体化：改变的不是“样本分布”，而是我们对哪个完整函数仍可能成立的信念分布。该等式要求同一个静态 $Q_D$ 上的连贯 hard 更新；把多 seed optimizer 终点分布重新训练一遍，并不自动满足这个 Bayes 公式。

### 6.3 实际 Surprise 不等于标签未知时的期望信息增益

Surprise 是标签已经观察以后产生的实际编码代价。选择下一个尚未标注的输入时，更相关的是期望信息增益：观察其标签以后，函数或参数分布平均缩小多少。

一般定义为：

$$
\mathrm{IG}(x\mid D)
=
\mathbb E_{y\sim P(y\mid x,D)}
\left[
\mathrm{KL}\bigl(q(\theta\mid D,x,y)\,\|\,q(\theta\mid D)\bigr)
\right]
=
I(\Theta;Y_x\mid D).
$$

等价的 [BALD](https://arxiv.org/abs/1112.5745) 形式为：

$$
I(\Theta;Y_x\mid D)
=
H[Y_x\mid D]
-
\mathbb E_{\theta\sim q(\theta\mid D)}H[Y_x\mid\theta].
$$

若每个 hard function 对 $x$ 给出确定标签，第二项为零，期望信息增益就等于候选标签分布的预测熵。此时最接近 50:50、agreement 最低的输入，平均最有信息量。这是主动 disagreement 实验 [E21](experiments/e21.html)的理论基础。

二分类时，若 $p=P_D(Y_x=1)$，则

$$
A(x\mid D)=p^2+(1-p)^2.
$$

所以在标签尚未知时，低 agreement 与高预测熵、从而高**期望**信息增益单调对应。但标签一旦观察，读数变成 $-\log_2P_D(y\mid x)$：一个原本接近50:50的标签平均很有信息；一个 agreement 极高却与共识相反的标签，则可能产生更大的**实际** surprise。两种选样直觉并不冲突，它们分别回答“查询前平均能学到多少”和“这个已经发生的标签究竟排除了多少函数”。

需要保留一个实际边界：hard function 后验中“所有模型都答对”并不保证重新训练时该样本完全没有作用。它的 soft margin 仍可不同，加入样本也会从初始化开始改变优化路径。

在 soft Gibbs 更新中，这个边界可以写得更明确。若

$$
q'_{D}(\theta)
\propto
q_D(\theta)e^{-\beta\ell(\theta;x,y)},
$$

则广义 surprise 是

$$
\Delta C
=
-\log\mathbb E_{q_D}
\left[e^{-\beta\ell(\theta;x,y)}\right],
$$

而 posterior KL 为

$$
D_{\mathrm{KL}}(q'_D\Vert q_D)
=
\Delta C
-
\beta\,\mathbb E_{q'_D}\ell.
$$

因此 soft loss 下“样本编码代价”“posterior KL”和单网络 BCE 彼此相关但不相等；hard indicator 极限才恢复上一小节的精确等式。

### 6.4 加入样本以后：查询点收束，不代表全局 Agreement 单调

Hard conditioning 后，所有幸存函数都在已加入的查询点 $x$ 上给出标签 $y$，所以该点的局部 agreement 立即变为1。但在另一个 probe 点 $x'$ 上，agreement 可以上升，也可以下降：被删除的函数簇可能原本维持某个全局共识，也可能正是造成分歧的来源。于是全局平均 agreement、完整函数 collision 和目标函数质量都没有逐样本单调定理。

这给“收束—分叉—再次收束”一个直接解释：一个高 surprise 标签可能先摧毁原来的主导函数簇，使剩余簇暂时更加均衡、全局 agreement 下降；后续信息样本再继续删除竞争簇，函数分布才凝聚到新规则。局部 KL 信息增益始终非负，不等于全局 agreement 必须上升。

MUX3 的 cell/conflict 实验把这一区分变成了可见机制。普通格 $x_1=x_2$ 上，真实 MUX3、`copy x1` 和 `copy x2` 给出相同标签，因而这些样本几乎不能区分三者；selector 冲突格 $x_1\ne x_2$ 才把捷径与真实规则分开。Uniform fixed-$D$ SMC 在普通格上的后验目标准确率为0.993，在冲突格仅0.777；把冲突样本提高到75%后，冲突格准确率升至0.995，完整 MUX3 目标质量从0.000214升至0.782。这个干预提高的不是抽象“样本数量”，而是单位样本削减当前竞争函数分母的效率。

## 7. Agreement 猜想

对某个输入 $x$，若候选标签概率为 $P(y\mid x,D)$，两次独立函数抽样给出同一标签的 pairwise agreement 为：

$$
A(x\mid D)=\sum_yP(y\mid x,D)^2.
$$

对完整函数分布 $Q(f\mid D)$，完整函数 collision 为：

$$
C(D)=\sum_fQ(f\mid D)^2.
$$

平均逐点 agreement 很高，不保证完整函数 collision 很高；许多函数可以只在少数不同位置分歧。因此应用时必须同时报告代表性 probe 上的逐点 agreement、完整函数模态质量、collision 或 Hamming-ball mass。

> **Agreement 的关键优点是它不依赖研究者知道外部生成规则。它测量的是当前训练集和网络协议下，经验函数分布是否已经收束。Agreement 小于 1 表示仍有函数延拓在竞争；agreement 接近 1 表示网络对当前训练集所诱导的某个完整规则高度确信。在形成这一猜想的早期规则实验中，我们只在过拟合、训练样本数低于 grokking 相变区时观察到 agreement 明显小于 1；而 agreement 接近 1 的情况，恰好都是训练样本已经足以支持规则泛化的时候。这组配对观察使我们猜想：对于足够大的问题空间和非平凡训练集，只有当训练集稳定辨识出某条规则、样本数跨过相应相变区时，完整函数分布才会趋于收束；因此，在 agreement 接近 1 的训练集中，应当很可能存在一条可以进一步发现和提取的人类可读规则。**

[E27](experiments/e27.html)直接检查了这段直觉最容易混淆的时间顺序。Mod97的
四档数据在hard fit时，未见输入Agreement只有0.027--0.033；它没有因为训练集
已经记住就接近1。进入grokking后，Agreement主要随不同seed共同恢复正确目标
而上升，剩余错误仍大多seed特异。因此当前证据不支持“grokking前已经形成一个
共享完整错误函数”，也没有观察到独立于准确率的第二次Agreement突变。

高 agreement 不保证网络恢复了研究者预设的外部生成规则。单样本可以让网络一致收敛到某个常数延拓；但常数函数本身正是一条极短且人类可读的规则，所以这不是“高 agreement 对应可读规则”猜想的反例。它只说明这个猜想必须是 teacher-free 的：agreement 指向训练集与网络协议共同选出的规则，不保证该规则等于研究者藏在数据背后的 generator。我们的猜想是更具体的：

> 在足够大的问题空间中，如果一个非平凡训练集能让**完整函数分布**在严格 fresh-seed 审计下收束到接近 1，那么该训练集通常对应一个可被提取为较短、人类可读的逻辑规则。发现这个规则本身仍可能很困难。

需要更多样本才达到高 agreement，通常意味着可辨识规则更复杂，或竞争延拓更难排除。这个方向可以把 agreement 收束所需样本数当作协议相对复杂度的另一个操作量。

该猜想已经通过两轮初步压力测试：[E18](experiments/e18.html)的严格高共识终点全部落入 signed threshold 族；[E21](experiments/e21.html)用 anti-consensus 前缀推高所需样本后，终点从线性阈值扩展到更复杂但仍可读的二次 polynomial threshold。两项结果都支持猜想，但还没有证明“所有高 agreement 函数必然人类可读”。

这个猜想还有一个更深的引申：如果高 agreement 在更多任务、架构和更大问题空间中仍系统对应人类可读短规则，那么神经网络诱导的复杂度排序与人类直觉中的复杂度排序就不是彼此无关的。二者可能都在利用局部性、对称性、组合性、低阶交互和计算复用等同一批可压缩结构。早期百余确定性规则实验也提供了同方向的广度旁证：当目标存在精确、可复用的生成规则，且数据与优化可达性充分时，网络往往能同时把未见样本 loss 和跨 seed 函数分歧压得很低；欠约束或噪声条件则不会自动产生这种完整凝聚。

> **若这一方向继续经受压力测试，它暗示的不是一套唯一、绝对的“宇宙编码”，而是一类在多种有效表示语言中都相对经济的稳健压缩结构。神经网络和人类可能因为面对相同结构与有限计算资源，而部分收敛到相似的复杂度判断。**

这仍然是强猜想而非结论。架构依赖性和 parity 等反例已经说明两种复杂度不会完全重合；百余规则实验也没有全部采用 E18/E21 的严格 fresh-seed 符号审计，因此只能作为支持性现象，不能替代跨架构预注册检验。

潜在应用是规则发现和数据充分性诊断：当 agreement 接近 1 时，可以认为训练集已经稳定约束出某条规则，随后再用符号回归、程序搜索、可解释模型或人工分析去提取它；当 agreement 仍低时，继续争论唯一规则通常还为时过早。

## 8. 符号概念为什么可能出现

OOD 规则组合实验表明，网络可以把 rule code 或 role bit 解释成可复用语义，并在训练未出现的组合上继续执行。共享中间计算实验 [E08](experiments/e08.html)又显示：当多个任务需要同一昂贵中间状态时，复用表示能用更少容量和样本达到更低 loss。

因此，符号和概念可能不是连接主义系统外部强加的东西，而是多任务压缩压力下的经济宏观表示。一个稳定符号可以绑定不同角色、组合不同规则，并避免为每个任务重新保存整套计算。

目前证据主要是行为和计算经济性证据。它没有直接证明隐藏层中存在唯一、离散、与人类符号一一对应的变量，也没有证明所有语义都由同一机制形成。

## 9. 与已有 AGI 和深度学习理论的关系

### 9.1 一个共同数学核心：Gibbs 变分原理

设 $\mu_\Pi$ 是架构诱导的参考测度，$E_D(\theta)$ 是训练数据能量，则：

$$
\Phi_{\Pi,\beta}(D)
=
-\log Z_{\Pi,\beta}(D)
=
\min_q
\left[
\beta\,\mathbb E_qE_D(\theta)
+
D_{\mathrm{KL}}(q\Vert\mu_\Pi)
\right].
$$

第一项要求解释数据、降低 loss；第二项衡量候选分布 $q$ 偏离架构参考测度需要付出的复杂度。这个“数据拟合 + 描述偏离”的结构，是本框架与多条 AGI、认知和统计学习理论真正共享的数学核心。

把参数空间进一步按完整函数 $f$ 分解，就得到函数限制的配分函数和自由能。这一步是本项目最关心的分辨率：不是只问整体 evidence，而是问每个完整函数宏观态怎样随 loss、样本和架构竞争。

### 9.2 Solomonoff 归纳与 AIXI：通用先验的资源受限对应物

[Solomonoff 归纳](https://mlanthology.org/misc/1964/solomonoff1964misc-formal/)按程序长度给短程序更高先验权重。[Hutter 的 AIXI](https://arxiv.org/abs/cs/0701125)再把通用序列归纳与行动和奖励最大化结合，形成不可计算的理想通用智能体。

本框架的对应物是：

$$
P_\Pi(f)\propto e^{-C_\Pi(f)},
\qquad
C_\Pi(f)=-\log P_\Pi(f),
$$

其中 $\Pi$ 不是通用图灵机，而是一个具体、有限、资源受限的神经网络协议。因此它可通过采样和训练实验测量，却不具有 Solomonoff 先验或 AIXI 的通用最优性。

最准确的联系是：神经网络可以被看作一个**可计算、资源受限、架构相对的归纳器**。当前理论只描述从数据中选择函数；尚未加入状态、行动、未来历史和奖励，所以还不是 AIXI 意义上的智能体理论。

### 9.3 Schmidhuber 的 Compression Progress：从静态压缩到好奇心

[Schmidhuber 的 compression progress 理论](https://people.idsia.ch/~juergen/driven2008.pdf)认为，真正产生好奇心奖励的不是一个对象已经多可压缩，而是压缩器对数据的压缩能力取得了多少进步。令人感兴趣的经验应当既有惊讶，又包含可以被学会的新规律。

本框架已经能测一个样本带来的自由能增量：

$$
\Delta\Phi(z\mid D)
=
\Phi(D\cup\{z\})-\Phi(D).
$$

若样本早已被当前函数系综预测，增量很小；若它排除大量候选函数，增量很大；若学习它以后还能让大量未来样本不再惊讶，就产生真正的 compression progress。

E21 的 disagreement 查询与主动补样可以被看作这个思想的被动学习版本：先寻找当前最有希望改变函数分布的样本。要成为好奇心智能体，还需要让系统主动选择环境行动并把未来压缩进步作为内在奖励。

### 9.4 Free Energy Principle、Predictive Coding 与 Active Inference

[Friston 与 Kiebel（2009）的 predictive coding / Free Energy Principle](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/)把感知写成对隐藏世界原因的近似 Bayesian inversion，并用 variational free energy 给出可计算目标。两套框架共享“预测误差/能量 + 后验偏离/复杂度”的结构，但变量含义不同：

| 本框架 | Free Energy Principle / Active Inference |
|---|---|
| $q(\theta)$ 是参数或程序实现分布 | $q(z)$ 是环境隐藏原因的近似后验 |
| 数据是训练样本 | 数据是感官观测 |
| 函数自由能衡量规则实现质量 | variational free energy 衡量内部生成模型解释观测的能力 |
| disagreement 找高信息训练样本 | prediction error 驱动感知更新 |
| 目前没有行动闭环 | active inference 通过行动改变未来观测 |

本项目的 $-\log Z$ 是给定参考测度下的 evidence；FEP 通常优化 evidence 的 variational upper bound。引入近似分布 $q$ 后，两者由上面的 Gibbs 变分式连接。

这只是数学和功能层联系，不证明 Free Energy Principle 的生物学版本，也不证明大脑按本站测量的函数宏观态工作。

### 9.5 PAC-Bayes 与 Gibbs Posterior

[Catoni（2007）](https://arxiv.org/abs/0712.0248)直接用温度、Gibbs posterior 和相对熵把经验风险与泛化界联系起来。标准形式与本框架共享：

$$
q_\beta(\theta\mid D)
\propto
\mu_\Pi(\theta)e^{-\beta E_D(\theta)}.
$$

PAC-Bayes 主要问给定 posterior 后，KL 复杂度和经验风险如何控制测试风险。本项目进一步按完整函数 $f$ 分解 $q$，测量各函数的 $Z_f(\beta)$、排名换序和 loss profile，并将其与真实 optimizer 分布直接比较。

因此两者共享 Gibbs 数学，但研究分辨率和目标不同。PAC-Bayes 界不能自动推出某条 Neural K-profile，也不能证明 SGD 无偏采样 Gibbs posterior。

### 9.6 Singular Learning Theory

[Watanabe 的 Singular Learning Theory](https://doi.org/10.1017/CBO9780511800474)用 RLCT 取代普通参数计数，描述神经网络等奇异模型的 Bayesian evidence 与自由能渐近。[WBIC](https://jmlr.csail.mit.edu/papers/volume14/watanabe13a/watanabe13a.pdf)和后来的 [Local Learning Coefficient](https://proceedings.mlr.press/v258/lau25a.html)使这种复杂度更接近可估计对象。

最直接的候选对应是：

$$
\text{RLCT / LLC}
\quad\longleftrightarrow\quad
\text{低-loss 体积的局部或渐近收缩指数}.
$$

但 SLT 主要研究大样本 Bayesian 渐近和局部奇异结构；本项目测有限数据、有限网络、具体完整函数宏观态、profile 交叉和 optimizer 可达性。SLT 可能成为未来严格推导 volume slope 的工具，目前不能直接替代这些实验。

### 9.7 经典学习统计力学、Flat Minima 与 Local Entropy

[Seung、Sompolinsky 与 Tishby（1992）](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.6056)早已用 Gibbs ensemble 研究样本数、温度、泛化曲线和学习相变；经典工作主要集中于 teacher–student perceptron、大系统平均和少量可解析 order parameters。

[Flat Minima](https://doi.org/10.1162/neco.1997.9.1.1)和 [Entropy-SGD](https://arxiv.org/abs/1611.01838)关注一个参数解附近的宽度或 local entropy。本框架则把参数空间按完整函数合并成宏观态，问同一函数在整个参数空间有多少实现，以及其总体积怎样随 loss 收缩。

因此函数体积比单个 minimum 更粗粒化，也更接近程序选择；但它仍依赖参考测度与参数化，不是坐标无关的绝对复杂度。

### 9.8 Information Bottleneck 与 MDL Probing

[Information Bottleneck](https://arxiv.org/abs/physics/0004057)在表示变量中压缩关于输入的无关信息，同时保留与目标相关的信息。它和“压缩即智能”语言相邻，但压缩对象是随机变量互信息，不是完整函数的参数体积；[Saxe 等（2018）](https://openreview.net/forum?id=ry_WPG-A-)与[Kolchinsky、Tracey、Van Kuyk（2018）](https://arxiv.org/abs/1808.07593)还指出了深网“压缩阶段”及确定性任务中的退化和测量边界。

[Voita 与 Titov（2020）的 MDL probing](https://aclanthology.org/2020.emnlp-main.14/)不只问某层能否线性预测标签，而是问给定表示以后，用多少 online codelength 才能学会标签。这与用隐藏层表示作为输入、测量剩余标签映射自由能的想法高度相邻。

二者可以成为表示层面的辅助测量，但不能把 probe codelength、互信息、参数 norm、Neural K 和机器无关 Kolmogorov complexity直接画等号。

### 9.9 为什么当前仍不是完整 AGI 理论

> **AGI 定位：当前框架解释的是给定数据和神经协议时，学习系统怎样在函数空间中归纳；它目前是学习与归纳理论，不是完整 AGI 理论。**

完整智能体理论还至少需要：

$$
\text{状态}
+
\text{行动}
+
\text{未来预测}
+
\text{奖励}.
$$

一个可能的扩展是让行动选择兼顾预期自由能下降、信息增益和外部奖励：

$$
a^*
=
\arg\max_a
\mathbb E
\left[
\Phi(D)-\Phi(D\cup z_a)
+
\lambda R(a)
\right].
$$

这会同时邻近 AIXI 的序贯决策、active inference 的 expected free energy、compression progress 的好奇心奖励以及主动学习的信息增益。但目前没有实验建立这个行动闭环，所以它只能作为扩展方向。

### 9.10 本框架可能真正新增的连接

不能声称重新发明了 free energy、Gibbs posterior、MDL、AIXI 或学习统计力学。更可能属于本项目的组合是：

1. 按完整函数宏观态分解参数空间；
2. 测量每个函数整条 low-loss density-of-states / Neural K-profile；
3. 把逐样本 surprise 严格连接到同一个端点自由能；
4. 用 full-rule profile 前瞻预测随机数据集样本相变；
5. 把 optimizer 对静态函数体积的偏离单独视为非平衡运输；
6. 用 agreement 在不知道 teacher 的情况下测量函数凝聚，并检验其符号可读性。

所以它更像一个可实验的共同坐标系：

<div class="theory-link-chain" aria-label="共同理论坐标系">
  <span>Solomonoff / MDL</span>
  <span>Bayesian evidence / PAC-Bayes</span>
  <span>统计物理自由能</span>
  <span>predictive coding</span>
  <span>神经网络函数体积</span>
</div>

## 10. 为什么这不是循环定义

把

$$
K^N_\Pi(f;\epsilon)=-\log_2V^{\mathrm{full}}_\Pi(f;\epsilon)
$$

**定义**成协议相对体积复杂度，本身当然不可证伪；它只是给测量结果命名。真正可证伪的是：这个定义能否预测它自身之外的量。

目前有四类独立锚点：

1. [E10](experiments/e10.html)在实验前用线性可分性构造严格简单/复杂函数对，再看低 loss 赔率，避免事后给赢家改名。
2. [E23](experiments/e23.html)先测完整规则 volume score 并冻结哈希，再预测独立随机训练集的 `n50/n90`；parity 家族顺序严格命中。
3. [E25](experiments/e25.html)用静态分支预测未见 MNIST 标签和 NLL 转折，而不是只解释训练目标自身。
4. [E18](experiments/e18.html)与 [E21](experiments/e21.html)用独立的符号复杂度审计检查高 agreement 终点是否人类可读。

反例也真实改变了理论：AND shortcut 否定预设赢家，weighted rule-bit 否定单函数全程单调，random/parity 的浅层错序否定单截面标量，deep crossing 否定 hard ID 足够，MUX3/parity2 又否定“完整规则难度单独决定样本相变”。正因为理论会被这些结果迫使修改，它不是“网络选了什么，什么就叫简单”的循环解释。

## 11. 数据量、Loss 深度与相区

同一规则在不同数据量下可以出现四个典型区域：

1. **欠约束区。** 样本太少，多个巨大延拓满足训练集；agreement 可以很低，也可能错误地高在常数或 shortcut 上。
2. **临界/Grokking 区。** 数据已经允许规则辨识，但规则只在深 loss 占优；训练先拟合，再在继续降 loss 时由规则接管。
3. **充分数据/直接学习区。** 规则在较浅 loss 就压倒替代函数，train 和 validation 从早期同步下降；早期百余成功实验大多属于这里。
4. **有限精度与噪声区。** 任意有限训练集只支持有限外推精度；继续压 loss 可能强调训练集特异残差或错误标签，导致 NLL、accuracy 或外部规则性能分离。

在纯 hard conditioning 下，增加一个与目标规则一致的样本不会删除目标 cell，却会删除部分竞争 cell，因此目标的归一化 hard 质量不下降。对固定 raw-BCE 截面没有同样无条件的单调定理，因为新样本同时改变平均 loss 和 margin 约束。

数据量、网络容量和训练时间承担不同角色：容量决定能否表示，数据决定能否辨识，训练动力学决定能否到达。把三者混成一个“任务难度”会掩盖 parity、Mod 3、grokking 和过拟合之间的真实差别。

## 12. 当前理论边界

当前没有解释或严格推导以下问题：

- 为什么广泛网络架构会产生先验简单性偏置；
- 为什么特定函数的低-loss 态密度具有当前形状；
- 能否从网络结构解析预测完整 Neural K-profile；
- 任意函数对的 profile 是否最终稳定、交叉次数是否有限；
- 高 complete-function agreement 是否普遍蕴含人类可读规则；
- 静态体积怎样与具体 SGD/AdamW 运输形成定量闭式关系。

因此，本理论目前是一套由受控实验支持的静态—动态分层框架，不是完成的深度学习统一定律。

> **最强主张：训练 loss 的连续下降具有函数选择意义；这种选择可以用协议相对的函数体积、自由能、预测分支和样本相变来测量。**
