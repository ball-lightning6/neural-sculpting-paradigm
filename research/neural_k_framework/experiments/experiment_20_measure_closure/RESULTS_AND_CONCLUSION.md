# E20：完整结果与阶段裁决

## 1. Constant leave-one-out 三路闭合

在7个两个 full 事件均可测的阈值上，独立 full-rule 体积比与 subset 条件交叉事件比闭合，最大自然对数 ratio residual 约0.071。Hard sign odds 可与 full-rule 体积比显著不同，但 soft-margin 条件概率补上后恒等式成立。

## 2. AND 四候选闭合

四个候选的独立 full-rule SMC 与 subset 交叉事件在可共同测量区最大 closure residual 约0.049。由此确认 full-rule 与固定 D 不是同一量，却属于同一参数测度并满足严格条件分解；先前“体积差百个数量级而 hard 函数仍竞争”的表面矛盾不是 SMC 失真。

## 3. Hard-to-margin 连续桥

父 subset 推进到`L10<=0.065`后全部 hard-exact。四个 hard-cell 分支在`tau=ln2`起点以 TV 约0.0086复现父 hard posterior；只收紧 heldout margin，四个 cell 的绝对核心质量随后连续分离，D440 最终取得压倒性优势。分离发生在 hard ID 不变的 cell 内部，证明 hard function 并非复杂度流的最细粒度。

个别分支在更深 tau 附近停止推进，按预注册只报告可达下界或上界，不伪装成零体积。

## 4. 裁决

E20 建立了静态测量闭环：

1. `V_full`与`Omega_D`不是同一个量；
2. 它们通过集合包含和条件概率严格关联；
3. full-rule 体积可作为有桥接依据的测量，而非事后类比；
4. 固定 D 的 hard 函数竞争与完整规则极深体积差异可以同时成立；
5. 同一 hard cell 内部的 margin/连续实现仍可发生巨量体积收缩。

它没有证明独立定义的复杂函数在所有固定 D 下必然收缩更快；复杂度方向仍需 Neural K-profile 及独立样本相变预测提供外部判据。
