# 跨架构规则体积：首轮完整队列分析

日期：2026-09-12。来源：`E:\Downloads\results_cross_architecture_rule_volume_compact_curves_v1_package.zip`，约2.51GB。

## 总体判断

本轮已经取得实质的跨架构测量结果，不再只是测速。
最清楚的信号是：不同架构在相同训练约束下给出差异很大的函数偏好；若干架构的完整低loss曲线能够在四个独立副本中复现；可共同比较的规则对呈现一致方向的体积/收缩差异。
但这仍是128粒子的有限loss初测，部分路线只有时间截断结果或明显副本分歧，不称为全部架构已通过精确深尾验证。

**必须保留对象定义：这里测的是16个固定输入对应的`K_D`，不是以全部256输入约束得到的完整规则`K_f`。**
Rule204/90是生成两份约束标签的规则名称；每条曲线内部仍含所有满足该训练集的候选bit映射。
图像模型的“完整bit映射”只读取每个输入图像的8个指定格点，整图指标另列。

## 覆盖与可靠性

- 十种compact架构，固定8bit输入/8bit输出、16个约束输入，图像编码为32×32黑白格点。
- 平均BCE共20条曲线、80个副本；图像native-MSE共6条曲线、24个副本。
- 104个活跃副本都已尝试，54个达到目标，50个触发单项时间限制。没有运行错误。
- 等价为13条曲线四副本全部达到目标，Swin-Unet/Rule90/格点BCE另有2个副本达到目标。
- 活跃任务累计记录用时3.693小时，包含运行中的诊断；另有两条已隔离最大BCE各约3分钟。
- 50个未完成项并非漏跑或都不能达到目标，只是本次单项预算已用完。
- 最大BCE的2条原记录保留但隔离。包含关系报警仍成立，不混入以下结论。
- 直接先验校准未触发现有启发式报警，但这不能认证深尾混合。

## 规则对的共同方向

对每个架构，只在**两条规则、四个副本都到达的最深共同节点**比较。
十种架构均为`K_D90 > K_D204`，而且各自四个Rule90估计全都高于四个Rule204估计。
其中Transformer只能共同到0.6、ViT到0.3、普通U-Net到0.4，不能把这些比较说成它们的深尾结论。

两规则都四副本到BCE=0.001的四个架构，最后一次减半窗口的系数为：

`c = [K_D(0.001)-K_D(0.002)] / log2(2)`，先在同一副本内做差，再取副本中位数；±为副本SD。

| 架构 | Rule204的c | Rule90的c |
|---|---:|---:|
| MLP | 348.39 ± 10.82 | 405.43 ± 5.50 |
| RNN | 636.19 ± 6.34 | 677.03 ± 7.13 |
| ConvNeXt | 343.12 ± 6.28 | 370.81 ± 13.67 |
| 改进U-Net | 17.46 ± 1.25 | 57.16 ± 3.57 |

这支持一个有限范围的跨架构结果：在这对预选规则生成的训练约束上，较复杂的Rule90条件不仅体积较小，在最后窗口也收缩得更快。
不能把它升级为全部完整函数都满足同一复杂度定律；这里只测一对规则，且对象是训练集条件体积。

![平均BCE曲线](E:/code/clean_github/neural-sculpting-paradigm/tmp/cross_architecture_rule_volume_analysis_20260912/01_bce_curves.png)

## 架构偏好与函数凝聚

在Rule204、平均BCE≤0.001、四副本均完成的条件下，全部约束格点都已答对：

| 架构 | K_D中位数 | 未见逐bit准确率均值 | 完整256输入bit映射等于Rule204的粒子比例 |
|---|---:|---:|---:|
| Swin-Unet | 37.13 | 100% | 100% |
| Transformer | 64.11 | 约99.998% | 99.41%（509/512） |
| 普通U-Net | 96.35 | 100% | 100% |
| 改进U-Net | 165.01 | 98.71% | 0/512 |
| ConvNeXt | 2207.68 | 90.87% | 0/512 |
| MLP | 2223.98 | 87.92% | 0/512 |
| RNN | 4427.77 | 96.12% | 0/512 |

数字是相关SMC粒子的经验分布，不是512个独立训练模型。0/512不意味着真实条件概率严格为零。
“改进U-Net”只是旧代码的版本名称，不表示它应在这项测量上全面优于普通U-Net。

Swin-Unet在更浅的区域就发生了清楚的Rule204凝聚：

| 平均BCE上限 | 完整Rule204粒子比例 |
|---|---:|
| 0.7 | 2.9% |
| 0.6 | 13.3% |
| 0.5 | 32.4% |
| 0.3 | 78.9% |
| 0.1 | 99.8% |
| 0.03及以下 | 100% |

这直接展示了在所采静态条件分布中，随着约束收紧，规则簇可以凝聚。
同时MLP、RNN等仍存在大量不同外推，说明充分拟合与函数凝聚不是同一件事。
RNN的Rule90末端收缩系数约677且副本SD仅约7，但未见位准确率约52%、函数熵接近128粒子能显示的7bit上限，目标完整映射未被采到。
所以稳定的宏观收缩曲线不等于函数分布已经凝聚，也不能据此推断每个候选函数的复杂度已经被逐一测定。

![体积与完整规则恢复](E:/code/clean_github/neural-sculpting-paradigm/tmp/cross_architecture_rule_volume_analysis_20260912/03_copy_volume_and_function_recovery.png)

跨架构的绝对`K_D`不是泛化分数：例如RNN的`K_D`比MLP高，但其未见位准确率也更高。
要解释泛化，仍需看该条件区域内部不同函数簇的相对质量。
这里比较的是架构、输入编码、读出设计和明确Gaussian参考测度的组合，不是对架构普遍能力的排名。

## 不是所有系数都已成为常数

在固定0.01至0.001区间，比较最前与最后的相邻窗口：

- MLP和ConvNeXt的系数仍增加约20%。
- RNN两规则增加约9%至18%，Transformer/Rule204约18%。
- 普通U-Net与Swin-Unet的Rule204系数下降约18%至21%。
- 改进U-Net的两条曲线变化较小，约下降2%和6%。

因此更准确的表述是：出现了可测、通常平滑、架构相关的收缩profile，其中部分窗口系数较稳定；不是所有架构都已经到达同一种恒定斜率深尾。
用平均BCE换算出的`a=-log(expm1(epsilon))`只是坐标重参数化，不代表所有格点真的等幅。
固定区间的直线拟合结果保存于`tail_fits.csv`，没有把高R²当成渐近定律或混合认证。

![局部收缩系数](E:/code/clean_github/neural-sculpting-paradigm/tmp/cross_architecture_rule_volume_analysis_20260912/04_local_contraction_slopes.png)

## 图像输出：格点全对不等于整图全对

为了不混淆二者，本轮用每个副本已保存的前两个权重，在全部256输入上重新前向。
以下每条件只有8个保存权重，指标只取240个未约束输入，**不是全部512粒子的整图统计**。

| 架构与采样约束 | 未见像素正确率 | 未见整张图像所有像素解码正确的比例 |
|---|---:|---:|
| Swin-Unet，8格点BCE≤0.001 | 64.78% | 0% |
| Swin-Unet，整图sigmoid-MSE≤0.0001 | 99.9994% | 99.43% |
| 普通U-Net，8格点BCE≤0.001 | 96.19% | 0.42% |
| 普通U-Net，整图raw-MSE≤0.002 | 99.98% | 82.92% |
| 改进U-Net，8格点BCE≤0.001 | 80.54% | 0% |
| 改进U-Net，整图sigmoid-MSE≤0.0001 | 98.71% | 72.14% |

Swin-Unet的格点BCE权重有明显网纹，而整图MSE权重几乎直接复制了整张黑白图：

![保存权重的实际输出](E:/code/clean_github/neural-sculpting-paradigm/tmp/cross_architecture_rule_volume_analysis_20260912/05_swin_unet_saved_outputs.png)

图中展示预设输入0/85/170/255，以及各条件r0保存的第一个粒子；没有选择最好图像。

这不是“MSE优于BCE”的公平损失比较：第三列只约束8个格点，第四列约束全部像素。
代码结构还给出一个明确原因：当前compact Swin-Unet最后是`2×2、stride=2`的转置卷积加逐像素1×1输出层；
8个读出格点的行列坐标均为偶数，只读取同一上采样相位，其它相位没有被格点BCE直接约束。
因而格点曲线对8bit读出任务有效，却不能代表整图生成质量；这不是原论文原尺寸Swin-Unet的普遍结论。

这里Swin-Unet/改进U-Net的MSE是sigmoid后的像素MSE，不能当作之前raw 0/1 MSE实验的重复。

## 哪些部分仍不宜定量使用

- 最大BCE两条记录已隔离，不能据其巨大K讨论换序或复杂度。
- Swin-Unet/Rule90格点BCE只有两副本到0.001；共同0.02处K的副本相对SD约34%，存在明显分叉。
- 普通U-Net/Rule204的raw-MSE在共同0.002处K的相对SD约59%，不宜把这条末端曲线当作可靠绝对值。
- Transformer/Rule90只能共同到0.6；其终点0.42至0.51不能被称为低loss极限。
- ViT、Swin、LSTM若干任务只是单次预算不足。时间限制本身不说明架构不能实现规则，也不单独说明采样失效。

四副本一致的部分有较好的初步复现性，但仍可能存在共同混合偏差；尚未进行更高粒子数/更强更新预算的独立确认。
本轮没有自动续跑50项或全体recheck。

## 数值审计

- 包内两份Jupyter源码的物理模型、目标、采样核心与当前已审计代码AST一致；未执行包内源码。
- 两规则的全真值表、训练输入、标签和图像均独立重建一致。
- 1608个已提交节点快照全部通过SHA256、loss、bit映射编码、heldout统计和函数熵复算。
- 约束违反数为0；训练格点与完整输入诊断中的对应logits差异为0；指标复算差异为0。
- 由保存的logits/图像重算loss，最大绝对误差约9.24e-8。
- 104个活跃副本各自末个固定节点，共208组保存权重，用CPU float64重新前向；最大原始输出差约1.87e-4，最大相对L2差约6.24e-6。逐副本第一个保存权重的约束格点无符号变化。
- 全部level的累计K算术和单调阈值复算通过，没有未提交的孤立快照。

这些检查支持实现与数据的一致性，不是稀有事件采样收敛证明。

## 资产

- [完整结构化分析](E:/code/clean_github/neural-sculpting-paradigm/tmp/cross_architecture_rule_volume_analysis_20260912/analysis.json)
- [每条曲线覆盖](E:/code/clean_github/neural-sculpting-paradigm/tmp/cross_architecture_rule_volume_analysis_20260912/coverage_by_curve.csv)
- [收缩系数](E:/code/clean_github/neural-sculpting-paradigm/tmp/cross_architecture_rule_volume_analysis_20260912/slopes_recomputed.csv)
- [整图前向检查](E:/code/clean_github/neural-sculpting-paradigm/tmp/cross_architecture_rule_volume_analysis_20260912/saved_image_checks.json)
- [快照审计](E:/code/clean_github/neural-sculpting-paradigm/tmp/cross_architecture_rule_volume_analysis_20260912/snapshot_audit.csv)
- [包分析脚本](E:/code/clean_github/neural-sculpting-paradigm/research/function_information_conservation/analyze_cross_architecture_rule_volume_package.py)
- [已保存图像权重前向脚本](E:/code/clean_github/neural-sculpting-paradigm/research/function_information_conservation/analyze_cross_architecture_saved_images.py)
