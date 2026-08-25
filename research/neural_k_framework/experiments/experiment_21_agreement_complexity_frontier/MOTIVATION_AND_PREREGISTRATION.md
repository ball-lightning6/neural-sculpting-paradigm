# E21：实验动机与预注册判据

## 1. Agreement 能说明什么

对固定训练协议，多 seed 在未见输入上的高 agreement 表示经验函数分布集中；它不依赖研究者知道 teacher，因此可用于未知含义数据。低 agreement 只说明仍有延拓竞争，不能单独区分数据不足、标签噪声、优化不可达或多模态地形。

逐点 agreement 较高仍不保证完整函数收束，因此实验同时报告：

- unseen pairwise agreement；
- vote entropy；
- exact modal 完整函数质量；
- 0.5% Hamming ball mass；
- fresh-seed fit rate。

## 2. 三种贪心分支

从随机平衡 n=8训练集开始，每轮选择当前 committee 预测最接近50:50的未见输入；这与 [Seung、Opper 与 Sompolinsky（1992）的 Query by Committee](https://doi.org/10.1145/130385.130417)共享“用版本空间分歧选择查询”的基本思想。对标签0/1使用严格配对初始化分别从头训练：

- anti-consensus：选择最终全局 agreement 较低的分支；
- pro-consensus：选择 agreement 较高的分支；
- random-label：在可拟合分支中随机选。

标签选择不读取任何符号复杂度。正式 pilot 使用8-bit、16起点、64-seed 发现、512 fresh-seed 审计，补到 n=24。

## 3. 复杂度前沿

共享16条 anti 主干，在`n=8,16,24,32,48,64`冻结快照；每个快照切换 pro 补全，最多加入32个样本直至发现阶段窄后验，再用512 fresh seeds 审计。只有 fresh-narrow 终点进入符号复杂度分析。

预注册代理包括 essential variables、ANF、decision tree、ROBDD 与线性阈值审计。二次 polynomial threshold 是看到未解释候选后增加的 post-hoc 审计，必须单独标记。

## 4. 边界

- pro/anti 是每步贪心，不保证全局最优数据序列；
- high agreement 不等于外部正确；
- 宽函数分布的逐点多数函数不能拿来作稳定程序复杂度判决；
- “可读”指存在较短、可提取的符号程序，不等于人类能一眼发现；
- 当前结论限于8-bit tanh MLP。
