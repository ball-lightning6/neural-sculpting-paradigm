# Dead bit：静态SMC、无衰减Adam、L2与NNGP

## 1. 问题与协议

输入为 (z,x0,x1,x2)。训练集完整覆盖三个有效bit的8个状态，但dead bit
z恒为0；反事实测试把z依次改为0.25、0.5、1和2，目标标签保持不变。

目标函数为 projection、XOR、majority、parity和MUX五种。网络是
4->16->1 tanh，共97参数。静态分支使用4个独立replica、每条2,048粒子的
Gaussian constrained SMC；优化器分支使用512个配对初始化；NNGP-like基线
由131,072个prior网络估计输出协方差。

主预注册比较是z=1时，无weight decay Adam与matched-loss静态SMC的严格正确
质量。平均绝对差不高于0.10且最坏函数不高于0.20为通过。

## 2. 主结果

原包的自动汇总报告：

- 平均绝对差：0.001001；
- 最大绝对差：0.004150；
- primary_static_prediction_pass=true。

自动汇总使用Adam终点与SMC最深层近似匹配；两者loss仍相差约20--60倍。
利用包内保存的Adam轨迹重新严格匹配后，五个函数的log-loss距离均不超过
0.057，z=1结论不变：

| 函数 | SMC z=1严格质量 | Adam z=1严格质量 | 绝对差 |
|---|---:|---:|---:|
| projection | 1.000000 | 1.000000 | 0 |
| XOR | 1.000000 | 1.000000 | 0 |
| majority | 0.999878 | 1.000000 | 0.000122 |
| parity | 0.999268 | 1.000000 | 0.000732 |
| MUX | 0.995850 | 1.000000 | 0.004150 |

因此在冻结的主口径上，静态低-loss质量几乎精确预测了真实无衰减Adam的
dead-bit反事实行为。

## 3. 不是dead权重被偷偷压成零

SMC最深层的dead列方差为0.975--1.077，16维dead列平方范数均值为
15.68--17.41，与N(0,I) prior的理论值16一致。无衰减Adam中dead列的最大
绝对变化严格为0，方差约1、平方范数约16。

尽管dead权重始终保持prior，SMC的z=1严格正确质量随loss深入系统提高：

- epsilon约0.2时，各函数约0.53--0.90；
- epsilon约0.02时，各函数约0.95--1.00；
- epsilon约0.001时，各函数约0.996--1.00。

机制不是识别并删除dead bit，而是有效分支在更低loss获得更大margin，使
prior尺度的dead扰动不足以改变hard function。

## 4. L2/MAP机制对照

显式L2把Adam的dead列方差和平方范数都压到近0；z=1严格质量为：

- projection、XOR、majority、MUX：1.000；
- parity：0.998。

与其较高终点loss匹配的静态SMC，在XOR、majority、parity、MUX上分别只有
0.934、0.955、0.843、0.794。这个差异复现了Izmailov等人的机制：MAP式
正则化主动选择dead权重为0，而BMA/静态质量保留其prior随机性。

## 5. 更强shift暴露边界

在z=2且严格匹配loss时：

| 函数 | SMC严格质量 | 无衰减Adam严格质量 |
|---|---:|---:|
| projection | 0.9827 | 0.9824 |
| XOR | 0.7361 | 0.8828 |
| majority | 0.8984 | 0.9316 |
| parity | 0.6472 | 0.8281 |
| MUX | 0.7456 | 0.9062 |

所以静态与优化器的近乎精确一致有明确作用范围：标准binary翻转z=1通过，
把未见输入幅度外推到2后，复杂函数出现约0.15--0.18的optimizer重加权优势。
这不是主判据失败，但限制了结论外推。

## 6. NNGP结果

对能够拟合z=0训练分支的小ridge，NNGP在五个函数的z=1分支全部严格正确。
因此本实验没有建立NNGP不能处理dead bit。z=2时，XOR和parity等高频函数
开始失败，说明它同样存在shift尺度边界。

## 7. 采样诊断与结论

SMC最深层pCN接受率为0.298--0.302，四个replica的z=1 posterior-predictive
分歧为0；运行耗时410.18秒，峰值显存约65 MB。

可以声称：

1. loss-resolved静态体积在binary dead-bit反事实上准确预测无衰减Adam；
2. 深loss通过margin而非dead方向posterior contraction恢复不变性；
3. 显式L2/MAP通过另一种机制获得更强鲁棒性；
4. NNGP也能处理本轮z=1任务，有限宽方法未形成对NNGP的独占优势；
5. 更强shift会重新放大静态与optimizer差异。

不能声称该Boolean结果推翻了真实MNIST/CIFAR协变量偏移中的BMA失败，也不能
声称静态体积在任意dead feature或任意扰动幅度上都等于SGD。

## 8. 标准温度1 HMC目标分布

为回答Izmailov等人所用标准Bayesian HMC在本任务上的行为，额外从Gaussian
prior抽取1,048,576个参数样本，并按精确Bernoulli likelihood
exp(-8*mean_BCE)重加权。该重要性积分与温度1 HMC具有完全相同的目标分布；
各函数ESS仍为总样本的62%--93%，不存在深尾退化。

| 函数 | ESS比例 | posterior平均loss | z=0 BMA准确率 | z=1 BMA准确率 | z=0严格样本质量 | z=1严格样本质量 |
|---|---:|---:|---:|---:|---:|---:|
| projection | 0.621 | 0.6706 | 1.000 | 1.000 | 0.0473 | 0.0362 |
| XOR | 0.924 | 0.7234 | 0.750 | 0.750 | 0.000004 | 0.000005 |
| majority | 0.737 | 0.6920 | 0.625 | 0.625 | 0.00508 | 0.00455 |
| parity | 0.932 | 0.7245 | 0.500 | 0.500 | 0 | 0 |
| MUX | 0.779 | 0.6998 | 0.750 | 0.750 | 0.000171 | 0.000141 |

因此标准温度1 HMC在这个8样本、97参数的缩放任务上会失败，而且对四个复杂
函数连训练分支的BMA都未充分拟合。失败不是采样器没有找到posterior，而是
posterior本身没有收缩到深loss区域。继续运行同温度HMC不会解决；重复数据、
降低温度或显式进入更低epsilon才会把分布推向前述成功的静态层。

这个结果与原论文并不矛盾。原论文的大数据模型在分布内已经有高准确率，主要
暴露OOD dead-pixel退化；本缩放任务的数据太少，使温度1 posterior的欠收缩
先在ID上出现。它反而清楚展示了loss-resolved profile相对单一温度posterior
多出的信息。

## 9. 文件

- 实验脚本：
  research/function_information_conservation/experiment_dead_bit_static_sgd_nngp.py
- 原始结果：
  E:/Downloads/results_dead_bit_static_sgd_nngp_package.zip
