# MNIST 50k：HMC与Adam结果

## 1. 预测质量

| 指标 | HMC | plain Adam | MAP Adam |
|---|---:|---:|---:|
| ensemble test accuracy | 99.03% | 98.95% | 98.82% |
| ensemble test NLL | 0.03291 | 0.03415 | 0.03775 |
| 单参数/单seed平均accuracy | 98.363% | 98.180% | 98.315% |
| train-test gap | 0.442 pp | 0.768 pp | 0.466 pp |
| 平均函数分歧 | 0.01820 | 0.02049 | 0.01646 |

HMC与plain Adam只相差8个测试样本。逐样本McNemar检验中，仅HMC正确19张、
仅Adam正确11张，双侧p=0.2005。因此应表述为同一性能水平，HMC有微弱但不
显著的数值优势。

## 2. 成员数与时间平均

HMC有480个参数样本，Adam有32个seed。把HMC分成15个互不重叠的32成员组：

- 32-member HMC accuracy均值99.012%；
- 标准差0.033个百分点；
- 范围98.96%--99.08%；
- Adam 32-seed ensemble为98.95%。

15/15个HMC组的hard-modal accuracy均高于plain Adam，但优势只有约0.06个
百分点。只取最后一个16-chain HMC快照时，HMC为98.93%、Adam为98.95%，说明
完整HMC收益部分来自跨时间posterior averaging。

## 3. 静态系综凝聚

HMC 480个样本在10,000张测试图上产生480个不同完整函数，但逐点平均agreement
达到0.9877：

- agreement不低于0.99覆盖91.32%测试集，准确率99.989%；
- agreement等于1覆盖87.32%，准确率99.989%。

高agreement是极强可靠性信号，但不是逻辑保证：有一张测试图被全部480个样本
一致预测错误。

按正确的snapshot-major保存顺序，同链平均函数分歧为0.01190，异链为0.01860。
异链仍系统性更远，因此结果支持高质量预测与函数凝聚，不证明HMC已在全局权重
空间完全混合。

## 4. 计算与解释边界

- 50k HMC约5,590秒；
- 全部四个Adam条件约141秒，其中50k plain Adam约57秒；
- HMC是测量静态函数系综的工具，不是更实用的训练算法。

本实验建立：整网静态HMC在真实50k十分类任务上可达到99.03%，并与同架构
真实Adam达到相同水平。

本实验不建立：

- HMC普遍优于Adam；
- HMC已全局精确混合；
- 静态分布等于optimizer终点分布；
- 该结果相对NNGP或其他kernel具有独占accuracy优势。

## 5. 文件

- experiment_mnist10_small_cnn_full_hmc_demo.py
- experiment_mnist10_small_cnn_adam_same_protocol.py
- README.md
- RESULTS_AND_CONCLUSION.md

