# E10：实验动机与预注册判据

## 1. 为什么广泛扫描先失败

第一阶段在完整4-bit 空间中组合 projection、majority、parity 和随机平衡训练约束，扫描`k=2,4,6,8`及两种网络测度：

- `tanh 16x2`：67,108,864个 prior 样本；
- `GELU+LayerNorm 1024x3`：100,663,296个 prior 样本。

但“人类指定的生成规则”不一定是当前有限训练集下网络偏好的简单延拓；ANF、DNF、决策树和综合 rank 也不一定对应网络语言。严重欠约束条件下，低 loss 转向另一个阈值函数不能直接算作简单性失败。

## 2. 两个更干净的判别原则

1. 使用样本足以暴露的明显简单规则，如 projection、AND 和 OR；
2. 直接比较一对函数，其中 complex 由 simple 加入离散例外，并且：
   - 二者在训练集上逐点完全相同；
   - 完整 truth table Hamming weight 相同；
   - simple 是单层线性阈值函数；
   - complex 经线性规划证明不可线性分；
   - complex 的 ANF、normal form 与 decision tree 代理均严格更大。

训练样本只从两函数 agreement set 中抽取，使用`k=10,12,14`。

## 3. 主判决量

对 simple 函数`f_s`和 complex 函数`f_c`定义：

$$
R(q)=
\frac{P_q(f_s)/P_q(f_c)}{P_{hard}(f_s)/P_{hard}(f_c)},
$$

其中`q`是 raw BCE 最低分位，`P_hard`是只满足相同 hard constraints 的 baseline。`R(q)>1`表示 lower raw BCE 进一步提高简单规则赔率。

raw BCE 是唯一 primary endpoint，因为它是对应训练实际优化的 loss。normalized BCE 与 fixed-scale 只作机制诊断，不参与主命题真伪。

## 4. 最终可靠性标准

旧自动摘要错误地只要求 tail 中 simple+complex 实际总数足够。由于 baseline 可能已极端偏向 simple，complex 的0/1个命中会主导赔率。最终规则改为：

$$
E_0[N_s]=N_s^{base}q\ge20,
\qquad
E_0[N_c]=N_c^{base}q\ge20.
$$

每个条件选择满足该规则的最深分位，并用 Fisher exact test 比较“入选低 loss tail”的 simple/complex 概率。该标准在结果返回后、读取方向前由统计可靠性问题触发，必须与旧 ZIP 摘要分开记录。

## 5. 边界

- 这仍是静态参考测度，不是 SGD 终点分布；
- 12个可靠比较集中在 projection、AND、OR 的 moderate variants；
- high variants 和 parity 常因 prior 命中过少不可裁决；
- 12/12结果证明所测函数对中的强倾向，不是所有 loss 区间、函数对和架构的定理。
