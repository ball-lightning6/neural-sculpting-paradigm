# 8-bit 条件规则决斗结果

训练集只包含两条4-bit规则输出一致的128个组合；测试集只包含输出冲突的
另外128个组合。所有结果均已对调前后位置并按seed配对。

## 最深匹配训练loss：0.001

| 低Neural K规则 | 高Neural K规则 | 4-bit差距 | 配对seed | 低K匹配率 | 低K胜出seed | Wilson 95%区间 | 判决 |
|---|---|---:|---:|---:|---:|---:|---|
| parity1 | majority3 | 70.03 | 256 | 0.761 | 1.000 | [0.985, 1.000] | lower_neural_k_rule_preferred |
| parity1 | mux3 | 189.39 | 256 | 1.000 | 1.000 | [0.985, 1.000] | lower_neural_k_rule_preferred |
| parity1 | parity2 | 238.20 | 256 | 1.000 | 1.000 | [0.985, 1.000] | lower_neural_k_rule_preferred |
| parity1 | parity3 | 318.62 | 256 | 1.000 | 1.000 | [0.985, 1.000] | lower_neural_k_rule_preferred |
| parity1 | parity4 | 579.31 | 256 | 1.000 | 1.000 | [0.985, 1.000] | lower_neural_k_rule_preferred |
| majority3 | mux3 | 119.36 | 256 | 1.000 | 1.000 | [0.985, 1.000] | lower_neural_k_rule_preferred |
| majority3 | parity2 | 168.17 | 256 | 1.000 | 1.000 | [0.985, 1.000] | lower_neural_k_rule_preferred |
| majority3 | parity3 | 248.59 | 256 | 1.000 | 1.000 | [0.985, 1.000] | lower_neural_k_rule_preferred |
| majority3 | parity4 | 509.28 | 256 | 1.000 | 1.000 | [0.985, 1.000] | lower_neural_k_rule_preferred |
| mux3 | parity2 | 48.81 | 256 | 0.606 | 0.617 | [0.556, 0.675] | lower_neural_k_rule_preferred |
| mux3 | parity3 | 129.23 | 256 | 0.983 | 1.000 | [0.985, 1.000] | lower_neural_k_rule_preferred |
| mux3 | parity4 | 389.92 | 256 | 1.000 | 1.000 | [0.985, 1.000] | lower_neural_k_rule_preferred |
| parity2 | parity3 | 80.42 | 256 | 0.983 | 0.992 | [0.972, 0.998] | lower_neural_k_rule_preferred |
| parity2 | parity4 | 341.11 | 256 | 1.000 | 1.000 | [0.985, 1.000] | lower_neural_k_rule_preferred |
| parity3 | parity4 | 260.69 | 256 | 0.990 | 1.000 | [0.985, 1.000] | lower_neural_k_rule_preferred |

## 解释边界

冲突集没有外部真值标签；这里测量的是训练集无法区分两条规则时，优化器
更常沿用哪一个完整扩展。低K匹配率高于0.5支持4-bit绝对复杂度在直接
竞争中的预测作用；混合函数占优、位置效应或反向选择都是有效反结果。
