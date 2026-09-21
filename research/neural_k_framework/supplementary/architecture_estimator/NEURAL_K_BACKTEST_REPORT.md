# 从有限网络分布直接预测 Neural K：3bit 与 4bit 回测

日期：2026-09-15。状态：计算、比较、绘图和审计已完成。

历史来源补充：后续查重发现，8月31日已有同一个有限宽 CGF 和倾斜重要性积分。本轮新增的是快速鞍点近似、完整五档回测及4bit迁移，不是首次引入 CGF。详见 [来源更正](E:/code/clean_github/neural-sculpting-paradigm/research/architecture_bias_theory_20260915/neural_k_backtest_v1/PROVENANCE_CORRECTION.md)。下列数值结果不变。

## 结论

得到了一次实质性的量化预测改进，而且不只是重新拟合经验式。

主方法没有用 SMC K 标定参数，从原单隐藏层 tanh 架构和 Gaussian 先验计算有限宽联合输出的 CGF，再对原平均 BCE 事件求 Chernoff 作用量，加入固定的鞍点/边界曲率近似。它在全部 40 个 3bit 轨道上明显优于旧经验式，原样迁移到预先固定的 64 个 4bit 轨道后仍明显改善。

这不是任意架构的通用复杂度定理，不是精确概率算法，也没有重新训练网络或重跑参数 SMC。

## 1. 比较协议

- 原网络分别是 3→16→1 和 4→16→1，tanh，输入 0/1。
- 所有参数的白化神经坐标是独立标准 Gaussian；第一层权重和 bias 除以 sqrt(bits)，读出权重和 bias 除以 sqrt(16)。
- 约束仍是全部真值表行的平均 BCE，不加幅度对齐，不换成 MSE。
- 3bit：全部 40 个严格对称轨道，5 个阈值，共 200 个预测节点。
- 4bit：8 个预先指定的命名锚点，加固定种子抽取的 56 个真正 4 变量轨道；共 64 个，其中 60 个真正依赖全部 4 个变量。5 个阈值，共 320 个节点。
- 新预测器不读取 SMC K。比较脚本在预测与 hash 冻结后才读取参考值；4bit 未重新标定。
- 旧对数和 Hill 的 3bit 比较使用原轨道留出预测；4bit 比较使用原先冻结的 3bit 参数直接迁移结果。
- 旧数据此前已经研究过，故称回测和跨规模迁移，不称从未见过历史结果的盲测。

## 2. 3bit：未标定预测

| BCE 上限 | K 平均绝对误差 / bit | K 中位相对误差 | R² | Spearman |
|---:|---:|---:|---:|---:|
| 0.5 | 0.4754 | 3.0966% | 0.99682 | 0.99981 |
| 0.2 | 0.3223 | 0.3329% | 0.99993 | 0.99981 |
| 0.08 | 0.3366 | 0.1376% | 0.99997 | 0.99981 |
| 0.01 | 0.4502 | 0.0762% | 0.99998 | 0.99981 |
| 0.001 | 1.0892 | 0.0544% | 0.99991 | 0.99962 |

最深节点的命名例子，K 单位均为 bit：

| 规则 | 原 SMC | 新预测，无标定 |
|---|---:|---:|
| 常数 | 80.3086 | 80.3494 |
| copy | 199.8383 | 199.8673 |
| majority3 | 360.9023 | 361.0600 |
| parity2 | 712.0828 | 711.3568 |
| parity3 | 872.0422 | 861.3441 |

这些不是选择出来的唯一成功规则；上述总指标包含全部 40 条曲线。parity3 是最深节点最大的偏差，约 10.70 bit。

## 3. 4bit：原样迁移

| BCE 上限 | K 平均绝对误差 / bit | K 中位相对误差 | R² | Spearman |
|---:|---:|---:|---:|---:|
| 0.5 | 1.2286 | 2.2347% | 0.98558 | 0.99936 |
| 0.2 | 1.8487 | 0.4443% | 0.99345 | 0.99954 |
| 0.08 | 1.7909 | 0.1396% | 0.99652 | 0.99840 |
| 0.01 | 3.7332 | 0.2165% | 0.99873 | 0.99895 |
| 0.001 | 9.8935 | 0.6188% | 0.99767 | 0.99748 |

最深节点并列比较：

| 方法 | 3bit，40 轨道 MAE / bit | 4bit，64 轨道 MAE / bit |
|---|---:|---:|
| 旧对数经验式 | 31.6178 | 195.6268 |
| 旧 Hill 经验式 | 27.4410 | 263.4283 |
| 新方法，无标定 | **1.0892** | **9.8935** |

其中 60 个真正依赖全部 4 个变量的规则：新方法最深节点 MAE=10.1325 bit、中位相对误差=0.6532%、R²=0.99637。因此改进并非只发生于嵌入的旧 3bit 家族。

这批 4bit 最深参考中，30 条带原 SMC 报警。全部保留；另报告原先无报警的 34 条：新方法 MAE=4.2987 bit、中位相对误差=0.2458%、R²=0.99962；旧对数式对应 MAE=163.6786 bit。没有以与新预测的吻合程度筛选参考值。

无报警不代表参考真值已认证；有报警也不能把全部偏差归咎于 SMC。新预测在深尾整体略低于原参考，仍有系统性近似误差。

![深尾预测比较](E:/code/clean_github/neural-sculpting-paradigm/research/architecture_bias_theory_20260915/neural_k_backtest_v1/deep_prediction_comparison.png)

## 4. 收缩系数也改善

统一使用有限窗口

`c = [K(BCE=0.001)-K(BCE=0.01)] / log2(10)`。

这不是宣称已求得 epsilon 严格趋零时的最终渐近系数。

| 对象 | 新预测系数 MAE | 系数中位相对误差 | R² |
|---|---:|---:|---:|
| 3bit 全部 40 轨道 | 0.2400 | 0.1192% | 0.99960 |
| 4bit 固定 64 轨道 | 1.8587 | 1.0403% | 0.99288 |
| 4bit 原参考无报警的 34 轨道 | 0.9075 | 0.4809% | 0.99846 |

![收缩系数比较](E:/code/clean_github/neural-sculpting-paradigm/research/architecture_bias_theory_20260915/neural_k_backtest_v1/contraction_comparison.png)

## 5. 为什么这是新信息，而不只是增加拟合自由度

主预测器没有从 SMC K 拟合比例、偏置或逐规则参数。它保留有限网络输出 CGF 中的高阶结构，显式计入隐藏表示倾斜的 KL 成本。

只使用 Chernoff 主导作用量、不计边界与涨落近似时，最深 3bit 的 MAE 仍为 21.11 bit，4bit 为 47.61 bit。加入事先固定的鞍点近似后，分别降至 1.09 和 9.89 bit。因此必须区分“主导代价”和“区域概率”，不能把前者直接改名为 K。

另有按原协议做的新作用量两参数标定版，完整记录在 JSON/CSV 中。主结论始终用无标定版，没有根据 4bit 表现选择另一个模型作为主结果。

方法是受到有限宽 Gaussian 混合/核分布文献启发后，针对本项目单隐藏层独立性作出的 CGF 化简，不是直接照搬 Bauer 的四层线性模型，也不宣称这些概率工具本身是首次发现。

完整公式与近似边界见 [METHOD.md](E:/code/clean_github/neural-sculpting-paradigm/research/architecture_bias_theory_20260915/neural_k_backtest_v1/METHOD.md)。

## 6. 误差与核验

### 尚未解释完的偏差

- parity4，BCE=0.001：参考 1657.7889 bit，新预测 1598.8841 bit，低估约 **58.90 bit**；参考带副本离散等报警，不能据此免除预测器的误差。
- parity4 在较浅节点也有较明显偏差，所以剩余问题不能仅归因于最深采样。
- 4bit 原参考无报警子集仍有最大约 21.30 bit 的误差。
- K 的相对误差小不等于概率倍率误差小：1 bit 就是倍率 2。当前不宜把全部预测直接当成高精度候选函数概率。

### 数值精度

- 核心 59 项技术检查通过：协方差、梯度、Hessian、线性可解模型、旧 Gaussian 优化器对照、初值和反号对称。
- 3bit 全部 200 节点由 32 阶提升到 48 阶后，预测变化小于 0.1 bit。
- 4bit 先 24 阶、再 32 阶；42 个不满足数值门槛的节点再用等价的分块实现提高到 40 阶。
- 4bit 最终 318/320 节点满足 0.1 bit 阶数差门槛。`0x0EF0` 和 `0x188E` 在 BCE=0.001 处的末档差分别为 0.1651、0.1357 bit，保留标记，没有删去。
- 所有 520 个事件的优化约束和 KKT 门槛通过。
- 独立最终审计 145 项通过；这指文件、实现、覆盖范围和指标核对，不代表鞍点近似误差已被数学认证。
- 数值序列化曾发生两次 NumPy 标量类型报错，均只涉及汇总/打印，已修复并校验保存文件；预测核心源码未因此更改，已计算节点未重新拟合。

### 资源与历史完整性

使用本地 GPU 1、RTX 3080 10GB，float64，TF32 关闭。未使用远程 GPU。

3bit 两档求积的初次计算约 50 秒；4bit 两档计算约 645 秒，42 节点复算约 171 秒。合计不含开发和检查的主要 4bit 数值工作约 14 分钟。3bit 汇总恢复只用了缓存，manifest 中恢复调用耗时不冒充初次计算耗时。

旧 sampler 主要用 float32 前向、float64 累加 log 质量；本轮是对同一数学网络的 float64 概率积分近似，不声称模拟了每一次 float32 舍入。

原始测量、旧经验公式及其冻结参数均未改写。参考文件 hash、预测 hash、额外 4bit SMC 复测节点及原报警全部保留。

## 7. 本轮可以支持的结论

**在这个单隐藏层 tanh、Gaussian 先验的网络族中，有限宽联合输出分布确实提供了足以显著改善 Neural K 和有限窗口收缩系数预测的信息；无需先用 SMC 标定每条规则，而且这种改进能迁移到新 4 变量规则。**

未完成：全部 1992 个 4bit 轨道的新预测；其他隐藏宽度/深度；CNN、RNN、Transformer；精确极小概率误差认证；真实 SGD 的函数频率预测。

## 文件

- [预测核心](E:/code/clean_github/neural-sculpting-paradigm/research/architecture_bias_theory_20260915/neural_k_backtest_v1/finite_cgf.py)
- [3bit 全部比较](E:/code/clean_github/neural-sculpting-paradigm/research/architecture_bias_theory_20260915/neural_k_backtest_v1/three_bit_comparison.csv)
- [4bit 全部比较](E:/code/clean_github/neural-sculpting-paradigm/research/architecture_bias_theory_20260915/neural_k_backtest_v1/four_bit_comparison.csv)
- [3bit 结构化汇总](E:/code/clean_github/neural-sculpting-paradigm/research/architecture_bias_theory_20260915/neural_k_backtest_v1/three_bit_summary.json)
- [4bit 结构化汇总](E:/code/clean_github/neural-sculpting-paradigm/research/architecture_bias_theory_20260915/neural_k_backtest_v1/four_bit_summary.json)
- [最终审计](E:/code/clean_github/neural-sculpting-paradigm/research/architecture_bias_theory_20260915/neural_k_backtest_v1/final_audit.json)

`PROTOCOL.md` 是启动前快照，保留原文和 hash；当前状态见 `PROGRESS.md`。
