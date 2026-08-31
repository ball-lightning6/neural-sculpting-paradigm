# E29：Dead bit的loss-resolved静态预测

## 目的

检验训练中完全未激活的输入方向保持prior时，随loss下降增长的有效margin能否
恢复反事实不变性，并定量预测相同初始化分布下的无衰减Adam。

## 入口

- experiment_dead_bit_static_sgd_nngp.py
- analyze_dead_bit_temperature1_importance.py
- MOTIVATION_AND_PREREGISTRATION.md
- RESULTS_AND_CONCLUSION.md

## 冻结来源

| 文件 | 开发版来源 | SHA256 |
|---|---|---|
| experiment_dead_bit_static_sgd_nngp.py | research/function_information_conservation | a21e84847d45fd457edd7a1326d8c45bd898072eac55f7b3affa934ddcb845e7 |
| analyze_dead_bit_temperature1_importance.py | research/function_information_conservation | 3b850e1048e3f95dd47091d36b620fc677bb91cadb6df4866ed76e99aaae5fe2 |

原始主实验结果包SHA256为：
3dea5218d4628dc7273a1a0cad102952a75d09623242bdd07efb98f8c34fc008。

