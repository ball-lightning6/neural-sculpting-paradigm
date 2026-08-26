# Parity2 / MUX3 fixed-D 静态 SMC：结果与裁决

## 1. 运行与数值状态

- 48个 fixed-$D$ 条件：2个目标 × 3种抽样协议 × 8份独立数据集；
- 每条件4个 replica，每副本2,048粒子；
- Gaussian 参考测度 `N(0,1/3)`；
- 10个 raw-BCE 阈值，从0.68推进到0.02；
- 共1,154个 SMC level，用时约1,798秒；
- 最后12层 proposal acceptance 均值约0.299，范围约0.289--0.308；
- 最后一层 survival 均值0.771，范围0.740--0.808；
- 所有条件在 `epsilon<=0.1` 时 train hard-exact mass 已约为1。

SMC 没有发生阈值停滞或 acceptance 崩溃。深尾 lineage 已明显收缩：`epsilon=0.02` 时多数 replica 只剩1--2条祖先，最大约6条。因此绝对 target-mass 小数不应当成高精度积分；主要使用跨阈值方向、数据集重复、replica 符号和数量级做裁决。

## 2. Hard-fit 后的主要结果

在 `epsilon=0.1`，全部条件已经 hard-fit：

| 协议 | 目标 | unseen accuracy | exact target mass | agreement |
|---|---|---:|---:|---:|
| uniform | parity2 | 0.917 | 0.038 | 0.941 |
| uniform | MUX3 | 0.821 | 0.000 | 0.948 |
| cell | parity2 | 0.944 | 0.209 | 0.954 |
| cell | MUX3 | 0.917 | 0.125 | 0.957 |
| conflict | parity2 | 0.969 | 0.260 | 0.967 |
| conflict | MUX3 | **0.988** | **0.465** | **0.986** |

跨目标排序与真实 AdamW 相变一致：uniform 和 cell 由 parity2 的目标质量占优，conflict 则由 MUX3 占优。Dataset bootstrap 对 MUX3-minus-parity2 的 target-mass 差在 `epsilon=0.1` 给出：uniform 为负、conflict 为正且95%区间均不跨零；cell 受8份数据集异质性影响，均值为负但区间较宽。

## 3. 深尾结果

到 `epsilon=0.02`：

| 协议 | parity2 target mass | MUX3 target mass | 占优目标 |
|---|---:|---:|---|
| uniform | 0.266 | 0.000214 | parity2 |
| cell | 0.469 | 0.284 | parity2 |
| conflict | 0.498 | **0.782** | **MUX3** |

对应目标相对全部竞争函数的 aggregate log-odds 为：

| 协议 | parity2 | MUX3 |
|---|---:|---:|
| uniform | -1.02 | -8.45 |
| cell | -0.12 | -0.92 |
| conflict | -0.008 | **+1.28** |

仅改变训练样本权重，MUX3 的 target-versus-competitor log-odds 从 uniform 到 conflict 提高约9.73 nat，赔率提高约1.7万倍。Parity2 从 cell 到 conflict 的 aggregate target mass 只变化约0.029，且8份数据集中正负各半；干预的静态选择性主要落在 MUX3 上。

## 4. 数据集与 replica 稳健性

在 `epsilon=0.02`：

- uniform：8/8数据集都是 parity2 target mass 高于 MUX3；
- cell：6/8数据集由 parity2 高于 MUX3；
- conflict：7/8数据集由 MUX3 高于 parity2；
- conflict-MUX3 相对 uniform-MUX3 在8/8数据集都提高 target mass，最小提升仍为0.219；
- conflict-MUX3 的32/32 replica 都有非零目标质量，replica 2.5%分位约0.190；
- uniform-MUX3 有25/32 replica 的目标质量为零；
- 目标函数作为 modal hard function 的数据集数：uniform-MUX3为0/8，cell-MUX3为4/8，conflict-MUX3为8/8。

Dataset-level paired bootstrap 在 `epsilon=0.02` 给出：

- MUX3，conflict minus uniform target mass：+0.782，95%区间约[0.598,0.921]；
- MUX3，conflict minus cell：+0.498，区间约[0.186,0.790]；
- uniform，MUX3 minus parity2：-0.265，区间约[-0.488,-0.075]；
- cell，MUX3 minus parity2：-0.185，区间约[-0.392,-0.016]；
- conflict，MUX3 minus parity2：+0.284，7/8数据集为正，但由于一份数据集反向较大，8-dataset bootstrap 区间约[-0.039,0.560]。

最后一项的均值、7/8方向和完整 loss 曲线均支持反转，但单独依靠8份数据集的最深点不能声称高精度总体差；真实训练干预提供了独立且样本量更大的外部确认。

## 5. 竞争函数机制被定位到 selector 冲突格

在 `epsilon=0.02`，将 MUX3 的224个未见输入按前三位格点分组。四个普通格 `x1=x2` 和四个 selector 冲突格 `x1!=x2` 的平均后验目标准确率为：

| 协议 | 普通格 | selector 冲突格 |
|---|---:|---:|
| uniform | 0.993 | **0.777** |
| cell | 0.999 | 0.909 |
| conflict | 0.999 | **0.995** |

Uniform-MUX3 的错误几乎全部集中在恰好能够区分 `copy x1`、`copy x2` 与真正 selector 规则的四个格点。Cell balancing 提供每格同样数量的证据，能够部分缓解；conflict enrichment 进一步把这些格点提升到75%训练权重，几乎完全消除错误。

Top-function 审计同样显示：

- uniform-MUX3 的目标函数在0/8数据集中为 modal；多个 modal function 与某个 copy 捷径高度相似；
- cell-MUX3 的目标在4/8数据集中为 modal；
- conflict-MUX3 的目标函数本身在8/8数据集中都是 modal。

因此“竞争函数分母”不再只是一个无法观察的解释项；其错误位置、函数形态和对定向样本权重的响应都被直接定位。

## 6. 静态与动态的最终分层

本实验回答了预注册主问题：**真实训练中由抽样方式引起的跨目标相变反转，已经存在于 fixed-$D$ 静态 loss-conditioned 函数分布；AdamW 不是制造这一定性排序所必需。**

但 optimizer 仍不是零贡献。静态深尾中 conflict 对 parity2 相对 cell 没有稳定负效应，aggregate target mass 甚至略高0.029；真实 AdamW 相变中 conflict 却让 parity2 的 `n50/n90` 各推迟8个样本。这一残差可能来自 optimizer 路径、暂态捷径、不同 loss/step 读取或跨 $n$ 的辨识过程。因此当前证据支持：

1. fixed-$D$ 竞争几何是一阶原因；
2. optimizer 运输调整定量幅度和部分单目标效应；
3. full-target Neural K-profile、fixed-$D$ 条件赔率和 optimizer 相变不能互相替代。

## 7. 理论裁决

被否定的强命题是：

> Grokking 相变点是完整目标 Neural K 的普遍单变量函数；不同目标面对统一且等强的噪声/捷径背景。

保留下来的更准确图景是：

$$
\text{完整目标 Neural K-profile}
\longrightarrow
\text{fixed-}D\text{目标/竞争函数赔率}
\longrightarrow
\text{数据分布下的辨识复杂度}
\longrightarrow
\text{optimizer 恢复相变}.
$$

Full-target volume 是强一阶预测量，特别是在竞争结构相近的 parity 家族；但跨函数族预测还必须给出抽样分布怎样削减目标特有的竞争分母。这个修正不是对原工作整体的否定，而是将一个过强的单变量对应改写成可测量、可干预的分层理论。

