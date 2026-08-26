# E23 深尾错序确认：预注册说明

## 1. 问题

E23 阶段 A 在 raw BCE `0.690 -> 0.670` 的浅区间冻结完整规则 volume score，阶段 B 再独立测量随机训练集恢复相变。Parity1--4 严格命中，但浅层分数相对于 `n50/n90` 有三组错序：

1. parity2 / MUX3；
2. parity3 / random-balanced；
3. parity4 / random-balanced。

这些错序可能表示浅层割线读得太早：规则的体积和局部收缩速度会随 loss 深入换序。但原 E23 的归一化参数被限制在 `Uniform[-1,1]`。在 width=16 的 tanh 网络中，bounded output logit 绝对值最多约4.25，故单样本 BCE 理论下界约为 `softplus(-4.25)=0.0142`；该协议不能进入任意低 loss。

新实验因此不是原 bounded profile 的无缝续接，而是一个明确的新协议：使用与 `Uniform[-1,1]` 同方差的无界 Gaussian 参考测度，把三对目标推入真正低-loss 深尾，检验排序方向是否跨参考测度保持稳健。

## 2. 固定协议

- 输入：8 bit，完整空间 256 状态；
- 网络：`8 -> 16 x 2 -> 1 tanh`，433 个参数；
- 参考测度：每个归一化参数独立 Gaussian，`sigma=1/sqrt(3)`；其方差与原 E23 的 `Uniform[-1,1]` 相同，但尾部无界；
- 采样器：保持该 Gaussian 测度不变的 pCN constrained SMC；
- 目标：parity2、parity3、parity4、MUX3、random-balanced；
- 共同父事件：五条规则的最小 full-target BCE 不高于 `0.700`；
- 8 个 replica，每副本 8,192 粒子；
- 五个分支从相同 parent replica 和 lineage 分出，并按相同阈值锁步推进。

该实验可以检验相变顺序是否得到无界深尾体积的跨协议支持，但不能把 Gaussian 体积数值与原 E23 uniform-cube volume score 当成同一条曲线上的连续点。

## 3. 操作性相变证据及分辨率

点值来自预注册样本数网格，不是无限精度相变位置：

| 规则 | `n50` 区间 | `n90` 区间 |
|---|---:|---:|
| parity2 | `(48,64]` | `(64,80]` |
| MUX3 | `(64,80]` | `(96,112]` |
| parity3 | `(80,96]` | `(96,112]` |
| parity4 | `(128,160]` | `(128,160]` |
| random-balanced | `>240`，右删失 | `>240`，右删失 |

相邻网格 tie、一个档位错序或 bootstrap 区间重叠不能单独判为理论不一致。本实验选择的三对方向由明确分离的区间或右删失关系支持。

原 E23 在每个相同 `n` 已有64份随机训练集。结果出现后增加的 agreement 亚网格诊断，以目标准确率0.90守门、在分叉最低点后的重新凝聚支插值 agreement=0.99，给出 parity2 59.98[58.77,61.07]、MUX3 69.47[65.02,72.40]、parity3 88.94[85.72,91.05]、parity4 151.93[149.65,153.81]，random >240。它不是原主判决，但为三对预注册方向提供额外分辨率；脚本把它明确标记为 post-hoc diagnostic。

## 4. 预注册方向

脚本记录以下对数体积比，正值支持操作性相变顺序：

```text
log V(parity2) - log V(MUX3)            > 0
log V(parity3) - log V(random-balanced) > 0
log V(parity4) - log V(random-balanced) > 0
```

运行开始后不得依据中间结果更改方向、目标函数、随机规则 seed 或停止标准。

## 5. 停止判据

只有在阈值低于 `ln(2)/256` 的 hard-exact 充分边界后，才允许触发确认性停止。三对目标必须同时满足：

1. 连续 5 个共同阈值中，8/8 replica 的预注册对数体积比均为正；
2. 同期体积比增长率的 replica 中位数持续为正。

若计划阈值全部用完仍未满足，报告当前比值、局部速度和未决状态；可以只在原阈值序列末尾追加更深阈值后 resume，不能更改既有前缀。

## 6. 判决边界

- 成功：说明 E23 操作性相变排序在同方差 Gaussian 深尾协议下仍得到支持，并表明排序具有一定跨参考测度稳健性；
- 未交叉但方向持续追赶：只能说明尚未采样到足够深；
- 深尾稳定反向：构成真正需要解释的反例，但仍须区分静态体积与 optimizer 可达性；
- 该实验不重新估计 `n50/n90`，不把两种量定义成同一个数，也不声称 Gaussian profile 就是原 uniform-cube profile 的续段。

## 7. 运行

```bash
python experiment_8bit_mismatch_joint_deep_bridge.py
```

脚本支持断点续跑。按 `Ctrl-C` 会保存 parent、五分支和所有比值表；保持 `RESUME=True` 重新运行即可继续。
