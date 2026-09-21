# 历史来源更正：有限宽 CGF 并非本轮首次引入

2026-09-15，在后续文献查重时发现并核对。

## 必须更正的表述

前一轮交流把有限宽 CGF 化简本身介绍得过于新。实际上，2026-08-31 已有相同的单隐藏层 CGF 推导与实现，并做过部分绝对体积核验。

- [旧结果记录](E:/code/clean_github/neural-sculpting-paradigm/research/function_information_conservation/RESULTS_3BIT_FINITE_WIDTH_VOLUME_CALCULATION_20260831.md)
- [旧 CGF 实现，第245行起](E:/code/clean_github/neural-sculpting-paradigm/research/function_information_conservation/calculate_3bit_finite_width_volumes.py:245)

旧代码 `FiniteWidthCGF.evaluate` 已经解析积分 Gaussian 读出，对单隐藏单元做 Gaussian 求积，并返回 CGF、均值、Hessian 和倾斜隐藏分布。其公式与本轮核心 CGF 相同。

## 两轮的真实区别

| 项目 | 2026-08-31 | 2026-09-15 |
|---|---|---|
| 有限宽联合输出 CGF | 已有 | 重新实现，不算新引入 |
| 概率计算 | KKT 倾斜后的重要性积分 | 快速鞍点与 BCE 边界曲率近似 |
| 单隐藏元求积 | 10阶、10000节点 | 3bit 32/48阶；4bit 24/32阶，部分40阶 |
| 主要核验覆盖 | 3bit 0.08处40轨道；0.01处7个规则 | 3bit全部40轨道、五档loss；固定64个4bit轨道、五档loss |
| 本轮重要增量 | 不适用 | 更完整回测、快速近似、收缩系数检验、4bit迁移及数值精度记录 |

旧方法的“无偏”是针对固定求积近似下的概率估计；不能把它表述为原连续 Gaussian 先验下无任何误差的精确概率。旧记录也已经标明部分困难规则重要性 ESS 很低。

## 不受此更正影响的结果

本轮 520 个预测节点、误差指标及无 SMC 标定的主方法没有改变。更正的是来源和增量归属，而不是撤回数值结果。

更准确的总结是：**重新接上并扩展了一条此前已走过的有限宽计算路线，而不是这次才第一次找到 CGF。**

本文件只补充历史来源，不修改已完成的预测核心、预测文件、方法快照或审计 hash。
