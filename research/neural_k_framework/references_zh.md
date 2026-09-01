# 参考文献与本站引用关系

> 本页只收录当前网页明确提及或直接依赖的方法与理论工作。论文地址优先使用期刊、会议、PMLR、JMLR、OpenReview、ML Anthology 或 arXiv 原始页面。

## 一、函数先验、参数到函数映射与简单性偏置

### R01 · Dingle、Camargo 与 Louis（2018）

Kamaludin Dingle, Chico Q. Camargo, Ard A. Louis. **Input–Output Maps Are Strongly Biased Towards Simple Outputs.** *Nature Communications* 9, 761 (2018). DOI: 10.1038/s41467-018-03101-6.

- [Nature Communications 原文](https://www.nature.com/articles/s41467-018-03101-6)
- **本站引用关系**：一般 input-output map 的简单输出偏置前史；同时提醒这种关系主要给出概率上界，简单输出不保证一定高概率。

### R02 · Valle-Pérez、Camargo 与 Louis（2019）

Guillermo Valle-Pérez, Chico Q. Camargo, Ard A. Louis. **Deep Learning Generalizes Because the Parameter-Function Map Is Biased Towards Simple Functions.** *ICLR 2019*.

- [arXiv:1805.08522](https://arxiv.org/abs/1805.08522)
- **本站引用关系**：神经网络参数到函数映射产生非均匀函数先验的直接前作；也是函数空间 PAC-Bayes 解释的重要来源。

### R03 · Mingard 等（2019）

Chris Mingard, Joar Skalse, Guillermo Valle-Pérez, David Martínez-Rubio, Vladimir Mikulik, Ard A. Louis. **Neural Networks Are a Priori Biased Towards Boolean Functions with Low Entropy.** arXiv (2019).

- [arXiv:1909.11522](https://arxiv.org/abs/1909.11522)
- **本站引用关系**：说明 Boolean 函数先验的低熵偏置，并在固定输出熵后继续观察结构复杂度偏好。

### R04 · Mingard 等（2021）

Chris Mingard, Guillermo Valle-Pérez, Joar Skalse, Ard A. Louis. **Is SGD a Bayesian Sampler? Well, Almost.** *Journal of Machine Learning Research* 22(79):1–64 (2021).

- [JMLR 正式页面](https://www.jmlr.org/papers/v22/20-676.html)
- **本站引用关系**：与本站多 seed 函数分布最直接的前作；支持静态 Bayesian function posterior 的一阶预测力，同时明确保留 optimizer 和超参数造成的二阶偏差。

### R05 · Mingard 等（2025）

Chris Mingard, Henry Rees, Guillermo Valle-Pérez, Ard A. Louis. **Deep Neural Networks Have an Inbuilt Occam’s Razor.** *Nature Communications* 16, 220 (2025).

- [Nature Communications 原文](https://www.nature.com/articles/s41467-024-54813-x)
- **本站引用关系**：E04 复用其 7-bit Boolean、深层 tanh 与 advSGD 核心协议，并延长首次零训练错误后的观察时间。

### R06 · Mingard 等（2025）

Chris Mingard, Lukas Seier, Niclas Göring, Andrei-Vlad Badelita, Charles London, Ard A. Louis. **Characterising the Inductive Biases of Neural Networks on Boolean Data.** arXiv (2025).

- [arXiv:2505.24060](https://arxiv.org/abs/2505.24060)
- **本站引用关系**：最接近“架构是一种程序语言”的近期布尔函数工作；其离散网络与 DNF 对应为 Neural K 的架构相对性提供直接对照。

## 二、Grokking、核极限与特征学习

### R07 · Power 等（2022）

Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra. **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.** arXiv (2022).

- [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- **本站引用关系**：确立小型算法数据上“先记忆、后延迟泛化”的原始现象和数据量轴。

### R08 · Nanda 等（2023）

Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt. **Progress Measures for Grokking via Mechanistic Interpretability.** arXiv (2023).

- [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- **本站引用关系**：把模加法 grokking 分解为记忆、Fourier 电路形成和 cleanup 的连续过程；是本站“表面突变可以有连续内部坐标”的重要对照。

### R09 · Jacot、Gabriel 与 Hongler（2018）

Arthur Jacot, Franck Gabriel, Clément Hongler. **Neural Tangent Kernel: Convergence and Generalization in Neural Networks.** *NeurIPS 2018*.

- [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)
- **本站引用关系**：E05 的解析无限宽固定核基线；用于判断固定 kernel 已足够时，真实有限网络是否仍会重组特征。

### R10 · Soudry 等（2018）

Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, Nathan Srebro. **The Implicit Bias of Gradient Descent on Separable Data.** *Journal of Machine Learning Research* 19(70):1–57 (2018).

- [JMLR 正式页面](https://www.jmlr.org/papers/v19/18-188.html)
- **本站引用关系**：说明零分类错误以后 logistic/cross-entropy 仍会推动 margin 方向；它是“post-fit loss 继续下降”必须区分的窄机制。

### R11 · Göring 等（2025）

Niclas Alexander Göring, Charles London, Abdurrahman Hadi Erturk, Chris Mingard, Yoonsoo Nam, Ard A. Louis. **Feature Learning Is Decoupled from Generalization in High Capacity Neural Networks.** arXiv (2025).

- [arXiv:2507.19680](https://arxiv.org/abs/2507.19680)
- **本站引用关系**：直接触发 E05 对 feature-learning strength 与泛化收益的拆分；提醒表示变化大不等于变化对目标有用。

### R12 · Kornblith 等（2019）

Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton. **Similarity of Neural Network Representations Revisited.** *ICML 2019*, PMLR 97:3519–3529.

- [PMLR 正式页面](https://proceedings.mlr.press/v97/kornblith19a.html)
- **本站引用关系**：E05 使用 centered kernel alignment（CKA）比较隐藏表示和经验 NTK；CKA 只是相似性仪器，不自动等于压缩或任务相关性。

## 三、算法信息、MDL、PAC-Bayes 与自由能

### R13 · Solomonoff（1964）

Ray J. Solomonoff. **A Formal Theory of Inductive Inference, Part I and Part II.** *Information and Control* 7 (1964).

- [ML Anthology 原文入口](https://mlanthology.org/misc/1964/solomonoff1964misc-formal/)
- **本站引用关系**：为“有限证据下偏好短程序”提供理想化算法概率参照；本站不主张有限神经网络或 SGD 实现 Solomonoff mixture。

### R14 · Kolmogorov（1965）

A. N. Kolmogorov. **Three Approaches to the Quantitative Definition of Information.** *Problems of Information Transmission* 1(1):1–7 (1965; Russian original 3–11).

- [MathNet 原文页面](https://www.mathnet.ru/eng/ppi68)
- **本站引用关系**：提供参考机相对的描述复杂度与不变性定理背景；Neural K 只被称为协议相对候选量。

### R15 · Grünwald（2004）

Peter Grünwald. **A Tutorial Introduction to the Minimum Description Length Principle.** arXiv (2004).

- [arXiv:math/0406077](https://arxiv.org/abs/math/0406077)
- **本站引用关系**：用于区分 two-part code、NML、prequential coding 与 Bayesian 方法，避免把 MDL 当成“SGD 自动最小化程序长度”的口号。

### R16 · Blier 与 Ollivier（2018）

Léonard Blier, Yann Ollivier. **The Description Length of Deep Learning Models.** *NeurIPS 2018*.

- [NeurIPS 正式页面](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3b712de48137572f3849aabd5666a4e3-Abstract.html)
- **本站引用关系**：用 variational 与 prequential code 实测深网描述长度；是 E22 把逐样本 surprise 与端点成本连接时的重要方法参照。

### R17 · McAllester（1999）

David A. McAllester. **PAC-Bayesian Model Averaging.** *COLT 1999*.

- [ACM DOI 页面](https://doi.org/10.1145/307400.307435)
- **本站引用关系**：函数先验路线用 PAC-Bayes 把先验/后验质量与泛化界连接起来；本站没有从该界推出 loss-profile 排名。

### R18 · Levin、Tishby 与 Solla（1989）

Esther Levin, Naftali Tishby, Sara A. Solla. **A Statistical Approach to Learning and Generalization in Layered Neural Networks.** *COLT 1989*, pp. 245–260. DOI: 10.1016/B978-0-08-094829-4.50020-9.

- [ML Anthology 正式条目与原文](https://mlanthology.org/colt/1989/levin1989colt-statistical/)
- **本站引用关系**：早已把固定架构网络写成 Gibbs ensemble，并连接预测、自由能和 predictive MDL；本站不把“神经网络自由能”作为首创。

## 四、平坦性、局部熵与奇异学习理论

### R19 · Hochreiter 与 Schmidhuber（1997）

Sepp Hochreiter, Jürgen Schmidhuber. **Flat Minima.** *Neural Computation* 9(1):1–42 (1997). DOI: 10.1162/neco.1997.9.1.1.

- [DOI / MIT Press 页面](https://doi.org/10.1162/neco.1997.9.1.1)
- **本站引用关系**：把宽低误差权重区域与 MDL/泛化联系起来；本站强调参数平坦度具有坐标依赖性，不能直接替代函数体积。

### R20 · Chaudhari 等（2017）

Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, Riccardo Zecchina. **Entropy-SGD: Biasing Gradient Descent Into Wide Valleys.** *ICLR 2017*.

- [arXiv:1611.01838](https://arxiv.org/abs/1611.01838)
- **本站引用关系**：以 local entropy 偏好宽谷；与本站静态参数质量相邻，但测量对象和优化协议不同。

### R21 · Watanabe（2009）

Sumio Watanabe. **Algebraic Geometry and Statistical Learning Theory.** Cambridge University Press (2009). DOI: 10.1017/CBO9780511800474.

- [Cambridge DOI 页面](https://doi.org/10.1017/CBO9780511800474)
- **本站引用关系**：Singular Learning Theory / RLCT 的基础来源；用于比较奇异模型证据和自由能渐近，不被等同为函数 Kolmogorov complexity。

### R22 · Lau 等（2025）

Edmund Lau, Zach Furman, George Wang, Daniel Murfet, Susan Wei. **The Local Learning Coefficient: A Singularity-Aware Complexity Measure.** *AISTATS 2025*, PMLR 258:244–252.

- [PMLR 正式页面](https://proceedings.mlr.press/v258/lau25a.html)
- **本站引用关系**：提供可扩展估计的局部学习系数；是未来比较 Neural K-profile 局部斜率、RLCT/LLC 与 margin core 的直接方法邻居。

## 五、主动学习的经典来源

### R23 · Seung、Opper 与 Sompolinsky（1992）

H. Sebastian Seung, Manfred Opper, Haim Sompolinsky. **Query by Committee.** *COLT 1992*. DOI: 10.1145/130385.130417.

- [ACM DOI 页面](https://doi.org/10.1145/130385.130417)
- **本站引用关系**：E21 以多 seed 分歧选择信息量高的未见输入，是版本空间 committee 查询思想的神经函数分布版本。

## 六、有限宽神经网络的现代统计力学

### R24 · Pacelli 等（2023）

R. Pacelli, S. Ariosto, M. Pastore, F. Ginelli, M. Gherardi, P. Rotondo. **A Statistical Mechanics Framework for Bayesian Deep Neural Networks Beyond the Infinite-Width Limit.** *Nature Machine Intelligence* 5 (2023).

- [Nature Machine Intelligence 原文](https://www.nature.com/articles/s42256-023-00767-6)
- **本站引用关系**：给出有限宽 Bayesian 深网超越 NNGP/无限宽极限的统计力学修正；用于说明现代定量理论可以解析特定模型，但不能自动替代本站对普通优化器和完整函数分布的实验测量。

## 七、AGI、自由能与认知理论

### R25 · Hutter（2007）

Marcus Hutter. **Universal Algorithmic Intelligence: A Mathematical Top-Down Approach.** In *Artificial General Intelligence*, Springer, pp. 227–290 (2007).

- [arXiv:cs/0701125](https://arxiv.org/abs/cs/0701125)
- **本站引用关系**：AIXI 把 Solomonoff 通用归纳与序贯行动、奖励最大化结合；用于说明本站目前只有归纳层，没有完整智能体闭环。

### R26 · Schmidhuber（2008）

Jürgen Schmidhuber. **Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes.** arXiv preprint (2008).

- [作者提供的原始 PDF](https://people.idsia.ch/~juergen/driven2008.pdf)
- **本站引用关系**：把好奇心奖励定义为压缩能力的进步，而非静态可压缩度；与本站的逐样本自由能增量和主动 disagreement 查询相邻。

### R27 · Friston 与 Kiebel（2009）

Karl Friston, Stefan Kiebel. **Predictive Coding under the Free-Energy Principle.** *Philosophical Transactions of the Royal Society B* 364:1211–1221 (2009). DOI: 10.1098/rstb.2008.0300.

- [PubMed Central 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/)
- **本站引用关系**：Free Energy Principle / predictive coding 的代表性入口；与本站共享 variational free-energy 数学，但隐藏变量、感知与行动对象不同。

### R28 · Catoni（2007）

Olivier Catoni. **PAC-Bayesian Supervised Classification: The Thermodynamics of Statistical Learning.** IMS Lecture Notes–Monograph Series 56 (2007).

- [arXiv:0712.0248](https://arxiv.org/abs/0712.0248)
- **本站引用关系**：用 Gibbs posterior、温度和 KL 复杂度连接经验风险与泛化；本站进一步做完整函数分解，但不由 PAC-Bayes 自动推出 profile。

### R29 · Watanabe（2013）

Sumio Watanabe. **A Widely Applicable Bayesian Information Criterion.** *Journal of Machine Learning Research* 14:867–897 (2013).

- [JMLR 原文 PDF](https://jmlr.csail.mit.edu/papers/volume14/watanabe13a/watanabe13a.pdf)
- **本站引用关系**：WBIC/RLCT 是奇异模型自由能渐近的可计算入口；可能用于未来推导低-loss 体积收缩指数。

### R30 · Seung、Sompolinsky 与 Tishby（1992）

H. Sebastian Seung, Haim Sompolinsky, Naftali Tishby. **Statistical Mechanics of Learning from Examples.** *Physical Review A* 45:6056–6091 (1992). DOI: 10.1103/PhysRevA.45.6056.

- [Physical Review A 原文页面](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.6056)
- **本站引用关系**：经典 teacher–student 学习统计力学；用于界定“温度、样本数与泛化相变”早已有严格先例。

### R31 · Voita 与 Titov（2020）

Elena Voita, Ivan Titov. **Information-Theoretic Probing with Minimum Description Length.** *EMNLP 2020*.

- [ACL Anthology 原文](https://aclanthology.org/2020.emnlp-main.14/)
- **本站引用关系**：用 online codelength 代替单点 probe accuracy；与未来按隐藏层测条件自由能/计算压力的路线相邻。

### R32 · Tishby、Pereira 与 Bialek（2000）

Naftali Tishby, Fernando C. Pereira, William Bialek. **The Information Bottleneck Method.** Allerton Conference / arXiv (2000).

- [arXiv:physics/0004057](https://arxiv.org/abs/physics/0004057)
- **本站引用关系**：定义输入信息压缩与目标相关信息保留的互信息权衡；压缩对象不同于完整函数参数体积。

### R33 · Saxe 等（2018）

Andrew M. Saxe et al. **On the Information Bottleneck Theory of Deep Learning.** *ICLR 2018*.

- [OpenReview 原文](https://openreview.net/forum?id=ry_WPG-A-)
- **本站引用关系**：说明所谓逐层“压缩阶段”依赖激活和测量条件，不是所有深网训练的普适现象。

### R34 · Kolchinsky、Tracey 与 Van Kuyk（2018）

Artemy Kolchinsky, Brendan D. Tracey, Steven Van Kuyk. **Caveats for Information Bottleneck in Deterministic Scenarios.** arXiv (2018).

- [arXiv:1808.07593](https://arxiv.org/abs/1808.07593)
- **本站引用关系**：指出确定性连续映射和确定性标签下 Information Bottleneck 可退化；本项目的精确规则任务正需要这一边界。

## 八、Bayesian Surprise 与主动信息增益

### R35 · Itti 与 Baldi（2009）

Laurent Itti, Pierre Baldi. **Bayesian Surprise Attracts Human Attention.** *Vision Research* 49(10):1295–1306 (2009). DOI: 10.1016/j.visres.2008.09.007.

- [PubMed Central 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC2782645/)
- **本站引用关系**：用更新前后模型分布的 KL 定义 Bayesian surprise；理论核心第6节把它具体化为完整 hard-function 分布，并证明实际标签 surprisal 等于 hard posterior 信息增益。

### R36 · Houlsby 等（2011）

Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, Máté Lengyel. **Bayesian Active Learning for Classification and Preference Learning.** arXiv (2011).

- [arXiv:1112.5745](https://arxiv.org/abs/1112.5745)
- **本站引用关系**：BALD 用标签与参数/假设之间的互信息选择查询；本站在确定性函数极限下把它连接到 predictive entropy、低 agreement 与 E21 的 disagreement 选样。

## 九、Grokking熵地形补充

### R37 · Zhang 等（2025）

Xiaotian Zhang, Yue Shang, Entao Yang, Ge Zhang. **Is Grokking a Computational Glass Relaxation?** *NeurIPS 2025*.

- [NeurIPS正式论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/92f67b9047fa7a43d7506054b5f0ec6a-Paper-Conference.pdf)
- [arXiv:2505.11411](https://arxiv.org/abs/2505.11411)
- **本站引用关系**：用Wang--Landau方法观测模运算Transformer的训练loss--测试准确率熵图，并用无显式正则的WanD找到高范数泛化解；支持“grokking不需要weight decay、泛化态可具有静态熵优势”的外部对照。E30只在更小系统中澄清显式L2如何改变同一静态函数竞争。
