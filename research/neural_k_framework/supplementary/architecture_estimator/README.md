# 当前最成熟的 Neural K 估计方法

本目录只保留当前推荐估计器及其最终回测，不上传前面的精度修正、双层失败和任务预测迭代。

## 方法

当前方法是：

1. 从固定单隐藏层 `tanh` 网络和 Gaussian 参数先验直接计算有限宽联合输出的 cumulant generating function（CGF）；
2. 对全真值表平均 BCE 事件求 Chernoff 作用量；
3. 用固定的鞍点/局部 Gaussian 边界曲率修正近似完整事件概率；
4. 对条件积分通过数值诊断的节点使用条件积分，否则回退到鞍点估计。

核心对象、适用条件和近似边界见[估计器方法说明](ARCHITECTURE_ESTIMATOR_METHOD.md)。

## 最终回测

- 3-bit：全部40个对称轨道、5档BCE；
- 4-bit：预先固定64个轨道、5档BCE；
- 总计520个规则×loss节点；
- 不用SMC K拟合预测器参数；
- 3-bit和4-bit均明显优于旧经验式，但深尾误差仍会增长。

最终结果见[520节点回测报告](NEURAL_K_LARGE_BENCHMARK_REPORT.md)。它是历史数据回测，不是完全盲测，也不是任意架构的通用精确算法。

## 原论文依据

本项目方法受到以下数学工作的启发，但不是它们的直接定理：

- Fischer et al., *Critical feature learning in deep neural networks*, ICML 2024：有限宽随机特征/核分布与表示涨落；[PMLR](https://proceedings.mlr.press/v235/fischer24a.html)。
- Bauer et al., *A unified theory of feature learning in RNNs and DNNs*, 2026：有限宽表示分布与作用量视角；[arXiv:2602.15593](https://arxiv.org/abs/2602.15593)。

这些论文提供数学背景；本目录中的 CGF、BCE凸对偶和局部边界修正是针对本项目单隐藏层协议的具体化，不能直接推广到多层、CNN、RNN或Transformer。

## 后续研究

最终估计器在形成过程中参考了下列后续研究，但均只借用接口或作诊断设计，不声称使用其定理：

- Oketch, Fewster & Goodman (2026), *A general framework for computation and estimation using the saddlepoint approximation*：鞍点三、四阶导数修正的实现接口（[arXiv:2607.17464](https://arxiv.org/html/2607.17464v1)，作者 R 包 saddlepoint）；配套基础为同一作者 *What is the price of approximation?*（[arXiv:2504.07324](https://arxiv.org/html/2504.07324v1)）。
- Katsevich (2025), *Saddle Point Approximation and Central Limit Theorem for Densities in high dimensions*：高维鞍点乘法误差研究，用于诊断设计与证明缺口定位，当前不能为本目录 K 值加严格误差条（[arXiv:2510.21545](https://arxiv.org/html/2510.21545v1)）。
- Voldoire, Chopin, Rateau & Ryder (2024), *Saddlepoint Monte Carlo and its Application to Exact Ecological Inference*：指数倾斜加特征函数积分的数值校正候选；与本项目参数空间 sequential Monte Carlo 名字相近但算法不同（[arXiv:2410.18243](https://arxiv.org/html/2410.18243v1)）。
- Papagiannouli, Trevisan & Zito (2026), *Beyond NNGP: Large Deviations and Feature Learning in Bayesian Neural Networks*：逐层核代价递推，是本估计器向多隐藏层扩展的候选路线（[arXiv:2602.22925](https://arxiv.org/html/2602.22925v1)，作者 JAX 代码 ldnn）。
- Macci et al. (2026), *Functional Large Deviations for Wide Deep Neural Networks with Gaussian Initialization and Lipschitz Activations*：从有限输入到整个输出函数的大偏差图景（[arXiv:2601.18276](https://arxiv.org/html/2601.18276v1)）。

历史来源说明：有限宽 CGF 于 2026-08-31 已在本项目中推导实现过，本轮是重新接上并扩展该路线，详见[历史来源更正](PROVENANCE_CORRECTION.md)。

## 保留的源文件

`current_method/`中保留最小复现脚本和协议：有限宽CGF、最终520节点回测、验证和绘图。SMC checkpoint、历史迭代和中间缓存不在发布树中。
