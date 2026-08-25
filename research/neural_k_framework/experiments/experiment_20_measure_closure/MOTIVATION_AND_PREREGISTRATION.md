# E20：实验动机与严格集合关系

## 1. 两个不同对象

完整目标体积：

$$
V_g^{full}(\epsilon)=\mu\{L_{full,g}\le\epsilon\}.
$$

固定部分训练集下的候选函数质量：

$$
\Omega_D(g;\delta)=\mu\{L_D\le\delta,h_\theta=g\}.
$$

二者相关但不相等，不能未经桥接互证。

## 2. Leave-one-out 恒等式

15个常数0训练点留出状态0110，比较完整常数0与该点单例外规则。令：

$$
A_\delta=\{L_{15}\le\delta\},\quad
B_{y,\epsilon}=\{L_{16,y}\le\epsilon\},\quad
\delta=16\epsilon/15.
$$

因 BCE 非负，`B subset A`，严格有：

$$
\mu(B)=\mu(A)P(B\mid A).
$$

两个独立 full SMC 的体积比必须等于 subset 粒子的交叉事件比。

## 3. AND 四候选与 margin bridge

对平衡 AND n=10的 D440/F040/D040/F440 重复：`delta=16epsilon/10`。独立运行一个 subset SMC 和四个完整规则 SMC；随后从同一 subset parent 按 hard ID 分支，在保持`L10<=0.065`和 hard cell 不变时，仅收紧6个 heldout 点的最大 BCE margin 门槛 tau。

`tau=ln2`是 hard-cell 边界，四分支归一化质量必须复现父 hard posterior；之后的变化只来自同一 hard function 内部的连续 margin 核心收缩。

## 4. 边界

交叉事件为0只能给有限粒子上界；分支 tau 停止下降表示当前采样配置触及可达下界，不是数学零体积。Hard odds 只看 sign，不等于完整 soft-margin 事件体积。
