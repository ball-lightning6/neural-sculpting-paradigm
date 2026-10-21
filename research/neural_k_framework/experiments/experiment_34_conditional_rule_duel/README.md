# E34：8-bit 条件规则决斗（旗舰对决）

> 状态：高差距 pilot 与完整循环赛均完成。证据等级：预注册复杂度顺序的事前预测，单 `8→16→1 tanh` 协议。

## 1. 问题与设计

**独立测得的 4-bit 单规则 Neural K 顺序，能否事前预测真实 Adam 训练在直接竞争中的规则选择？**（本地另存48对规则决斗面板，与本实验同属"规则对决"家族；本实验是其中最先做、信号最强的一组。该历史面板未随本轮发布，也不是当前公开编号 E36。）

8-bit 输入分前后两个 4-bit 块，规则 f 读前块、规则 g 读后块：

- 训练集：`f(front) == g(back)` 的全部 128 个输入，标签取共同输出；
- 冲突集：`f(front) != g(back)` 的 128 个输入，不指定外部真值，只检验竞争偏好；
- 每对交换前后位置并按 seed 配对；`8→16→1 tanh`（161 参数）、full-batch Adam(lr=0.01, wd=0)；matched-loss 阈值 0.1/0.03/0.01/0.003/0.001。

规则面板与顺序预期在读取训练结果**之前**从独立测得的 4-bit Neural K 曲线冻结：parity1 < majority3 < MUX3 < parity2 < parity3 < parity4。

## 2. 结果

**高差距 pilot**：5 场 × 双向 × 256 seed，全部 100% 选低 K 规则。

**完整循环赛**（6 条规则的全部 15 个无序对，双向结果按 seed 配对合并，再比较 5 个 loss 截面，共 15×5=75 项判决）：**75/75 全部符合事先冻结的复杂度全序**。这75项共享规则并重复读取同组训练轨迹，不是75个独立任务或独立统计试验。主截面（BCE=0.001）逐对结果见 `FULL_TOURNAMENT_RESULTS.md`；代表性的非饱和读数：

- parity1 vs majority3：低 K 匹配率 0.761（其余为混合函数）；
- **MUX3 vs parity2：0.606**（配对 seed 胜率 0.617，Wilson [0.556, 0.675]）——这是 15 对中唯一非压倒性的判决，方向仍符合冻结全序，但偏好随 loss 加深从 0.684 单调降到 0.606（未反转，作为开放读数保留）。

## 3. 必须随结果保留的边界

1. 冲突集没有外部真值；测的是训练集无法区分两条规则时优化器更常沿用哪个完整延拓，不是"哪条规则正确"；
2. 多数判决中网络形成的是混合函数（如 parity1 vs majority3 全部 256 个配对 seed 都胜但平均匹配率只有 0.761）——"偏向低 K 规则"指函数人口偏向，不要求每个网络复制完整规则；
3. 单一网络与数据协议；不宣称胜率等于 K 差的 softmax，也不宣称逐对单调；
4. 本地未发布的48对补充面板在小 K 差档（20–40 bits）记录到平均偏好接近 0.5。这是保留的外推警告，不是本页已公开复现的证据；本实验的强信号不能直接外推到所有小差距规则对。

## 4. 文件

- `experiment_8bit_conditional_rule_duel_high_gap.py`：冻结实验脚本（无 argparse，配置在顶部；pilot 与完整循环赛同一脚本不同模式）；
- `README_8BIT_CONDITIONAL_RULE_DUEL_HIGH_GAP.md`：原始协议说明；
- `FULL_TOURNAMENT_RESULTS.md`：完整循环赛 15 对逐对结果与解释边界；
- 结果包：`E:/Downloads/results_8bit_conditional_rule_duel_high_gap_package.zip`（pilot）、`E:/Downloads/results_8bit_conditional_rule_duel_full_tournament_package.zip`（循环赛，含逐条件权重与图）。
