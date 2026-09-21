# 8-bit 条件规则决斗：完整循环赛

脚本：`experiment_8bit_conditional_rule_duel_high_gap.py`

## 要检验什么

此前的 4-bit 静态测量给出了若干规则的 Neural K 顺序。单独比较两个规则的
绝对 Neural K 并不能证明它们会在同一次训练中竞争。本实验故意构造一份同时
符合两条规则的训练集，再把两条规则互相矛盾的输入全部留作测试，直接观察
网络最终更常延续哪条规则。

对两个 4-bit 规则 `f` 和 `g`，8-bit 输入写成 `(front, back)`：

- 训练集保留 `f(front) == g(back)` 的全部输入，并用这个共同输出作标签。
- 冲突集保留 `f(front) != g(back)` 的全部输入，不人为指定正确标签。
- 所选规则均为 8/8 平衡，因此训练集与冲突集各有 128 个输入。
- 两个集合中，每一种 front 和 back 状态都恰好出现 8 次，训练标签为严格
  64/64 平衡。

所以网络在训练中完整见过两个 4-bit 规则的每一种局部输入，但训练数据不能
告诉它在冲突输入上应该服从前半还是后半。冲突集上的选择就是本实验要测的量。

## 规则与排序

4-bit Neural K 来自 `4 -> 16 -> 1 tanh`、Gaussian 参数测度在
`BCE = 0.01913` 深尾处的 `-log V`，单位为 nats：

| 低 Neural K 规则 | 高 Neural K 规则 | 差值 |
|---|---|---:|
| parity1 | parity4 | 579.31 |
| majority3 | parity4 | 509.28 |
| MUX3 | parity4 | 389.92 |
| parity1 | parity3 | 318.62 |
| majority3 | parity2 | 168.17 |

每对都交换一次前后位置，共 10 个条件。这样可以把“第一组输入更容易被网络
采用”之类的位置偏好与规则本身的偏好分开。

高差距 pilot 已经通过。正式脚本现在默认检验六条规则按上述 Neural K 顺序
形成的全部 15 个无序对。每对仍交换前后位置，因此完整循环赛共 30 个条件，
其中也包含 Neural K 差距较小、判别力更强的相邻规则。直接在 AutoDL
Notebook 运行代码即可，不需要传入命令行参数。

## 训练协议

- 网络：`8 -> 16 -> 1 tanh`，每个模型 161 个参数。
- 初始化：256 个独立标准 Gaussian 原始参数；forward 中继续使用原 4-bit
  测量的 fan-in 缩放。
- 优化器：full-batch Adam，学习率 `0.01`，不使用 weight decay。
- 比较截面：每个 seed 第一次达到训练 BCE
  `0.1, 0.03, 0.01, 0.003, 0.001` 时分别记录。
- 上限：30,000 step；长期没有新 seed 到达且 loss 基本不再下降时自动停机。
- 十个条件使用同一组初始化，正反位置结果按 seed 配对。

比较的是相同训练 loss 下的函数选择，不把相同步数误当成相同训练状态。

## 主要读数

`simple_match_fraction` 是冲突集 128 个输入中，网络输出与低 Neural K 规则
一致的比例。随机或无偏选择对应 `0.5`，高于 `0.5` 表示更偏向低 Neural K
规则。

脚本还会区分三种完整函数：

- `pure_simple_rule`：全部 256 个输入都等于低 Neural K 规则的扩展。
- `pure_hard_rule`：全部 256 个输入都等于高 Neural K 规则的扩展。
- `hybrid_function`：训练集全对，但冲突区形成了第三种混合函数。

主判决使用两个位置方向的配对均值及 seed bootstrap 95% 区间。只有至少 80%
的配对 seed 到达该 loss，且区间下界高于 `0.5`，才标记为低 Neural K 规则
占优。混合、无显著偏好、偏向高 Neural K 规则或强位置效应都保留为有效反结果。
报告同时给出“每个配对 seed 是否以低 K 规则占多数”的 Wilson 95% 区间，
避免 256/256 全成功时普通 bootstrap 显示成误导性的 `[1, 1]`。

## 运行

在服务器上运行：

```bash
python /root/autodl-tmp/experiment_8bit_conditional_rule_duel_high_gap.py
```

在 Jupyter 中也可以直接运行：

```python
%run /root/autodl-tmp/experiment_8bit_conditional_rule_duel_high_gap.py
```

脚本自动保存 checkpoint。中断后执行同一命令会从当前条件继续。若只想检查
线路，可运行：

```bash
python /root/autodl-tmp/experiment_8bit_conditional_rule_duel_high_gap.py --smoke
```

正式输出目录为：

```text
/root/autodl-tmp/results_8bit_conditional_rule_duel_full_tournament
```

完成后压缩包为：

```text
/root/autodl-tmp/results_8bit_conditional_rule_duel_full_tournament_package.zip
```

重点查看：

- `RESULTS.md`
- `conditional_rule_duel.png`
- `full_tournament_preference_matrix.png`（仅完整循环赛）
- `directed_rule_preference_matrix.csv`
- `paired_duel_verdicts.csv`
- `condition_threshold_summary.csv`
- `all_matched_loss_seed_records.csv`

## 结论边界

冲突集没有外部真值。本实验不判断哪条规则在人类意义上正确，而是判断此前
独立测得的 4-bit Neural K 顺序，能否预测两条完全符合训练数据的规则在直接
竞争时谁更常被网络延续。它首先测 Adam 的真实训练分布；若结果稳定，再用
静态 SMC 测量区分这种偏好来自低-loss 地形，还是主要来自优化器输运。
