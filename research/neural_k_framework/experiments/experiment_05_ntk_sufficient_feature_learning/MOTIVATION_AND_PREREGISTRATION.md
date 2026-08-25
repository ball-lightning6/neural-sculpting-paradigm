# E05：实验动机与预注册判据

## 1. 当时的问题

阅读 [Göring 等（2025），*Feature Learning Is Decoupled from Generalization in High Capacity Neural Networks*](https://arxiv.org/abs/2507.19680)时，研究首先接受了一个概念区分：表示或 kernel 变化很大，不等于这些变化必然改善泛化。随后提出更直接的反向问题：

> 如果初始化固定 NTK 已经足以学习规则，真实有限宽网络是否仍会发生特征学习？

这能区分“特征学习只是固定 kernel 能力不足时的补丁”和“特征学习是降低训练目标时可以独立出现的表示运输”两种图景。

## 2. 竞争预测

### H1：kernel 充分性导致 lazy 停留

若初始化 kernel 已经给出零错误规则解，端到端训练没有必要显著改变归一化经验 NTK、隐藏特征 Gram 或 ReLU 激活门控。网络达到完整训练/测试零错误后，结构变化更应接近数值噪声。

### H2：特征学习与 kernel 充分性、泛化可分离

网络只直接优化训练目标。即使固定 kernel 已经能泛化，有限网络仍可能沿训练 loss 重组表示；这种变化可以发生在完整函数已经正确以后，也不必等价于额外泛化收益。

## 3. 三个必须分开的 NTK 对象

1. **解析无限宽 NTK**：采用 [Jacot、Gabriel 与 Hongler（2018）](https://arxiv.org/abs/1806.07572)提出的 NTK，按同一三隐藏层 He-ReLU 参数化递推，在8,000个 Rule 110状态上做 KRR，并用20,000个未见状态检查完整30-bit 输出。
2. **初始化有限宽经验 NTK**：对`30 -> 1024 x 3 -> 30`具体初始化计算精确 Jacobian Gram，并在输出位0、15、29上分别做 KRR。它只是无限宽核的有限宽采样，不要求恰好零错误。
3. **训练中多输出 block NTK**：在固定128状态 probe 上追踪三个输出位及交叉输出 block，用于测量端到端训练中的结构变化。

经验 NTK 的闭式快速算法必须先在微型网络上与逐参数 autograd Jacobian Gram 比较，误差通过审计后才允许正式运行。

## 4. 正式配置

- 任务：30-bit 一维元胞自动机 Rule 110单步更新，输出30 bit；
- 数据：8,000训练状态、20,000固定未见状态；
- 网络：无 bias 的`30 -> 1024 x 3 ReLU -> 30`；
- seeds：0、1、2；
- 优化器：Adam，学习率`1e-3`，weight decay 为0；
- batch：1,024；训练30,000步；
- loss：BCE 与 MSE 各三条轨迹；
- 快照：0、10、30、100、300、1,000、3,000、10,000、30,000步。

MSE 是必要对照：它避免把 BCE 在零分类错误后持续增大 logit/margin 所引起的纯尺度变化误判成特征学习。

## 5. 预注册主指标

- `block_ntk_cka_to_init`：使用 [Kornblith 等（2019）](https://proceedings.mlr.press/v97/kornblith19a.html)系统化的 centered CKA，比较经验 block NTK 与初始化；
- `ck_cka_to_init`：隐藏激活 Gram 相对初始化的 CKA；
- `gate_flip_fraction_from_init`：固定 probe 上的 ReLU 门控翻转率；
- `block_ntk_target_alignment`与隐藏 CK target alignment：变化是否具有任务方向；
- `*_to_first_perfect`：第一次训练集与20,000测试集同时零错误后是否继续变化。

只看 kernel 范数不构成证据。核心判据是归一化 CKA、门控和任务对齐共同变化；“全对以后继续变化”作为更强的补充判据单独报告。

## 6. 预先声明的边界

- 解析无限宽 NTK 零错误与某个宽度1,024经验 NTK 零错误不是同一命题；
- 特征学习不自动等于泛化改善、程序压缩或计算压力重分配；
- 该实验只检验有限宽、该参数化和该优化协议；
- 快照过疏可能把真实首次全对时刻登记得偏晚，使 post-fit 变化量被低估。
