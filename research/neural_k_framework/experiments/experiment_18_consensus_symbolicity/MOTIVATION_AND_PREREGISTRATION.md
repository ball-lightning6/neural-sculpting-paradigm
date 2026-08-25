# E18：实验动机与预注册判据

## 1. 猜想

Agreement 不依赖研究者预设 teacher。若一个部分训练集使许多独立初始化收束到近乎同一个完整函数，它是否通常意味着 version space 中存在一个网络偏好、且人类可读的短规则？反例应是`high consensus + 无法由预注册符号语言压缩`。

## 2. Pilot

- 8-bit 完整输入空间256点；
- 540个数据集，其中512个随机平衡部分真值表、28个隐藏符号 teacher；
- 每集64 seeds，`8 -> 16 x 2 -> 1 tanh`，20,000步；
- 高共识要求 fit rate 不低于0.95、modal 完整函数质量不低于0.95、完整函数 collision 不低于0.90。

符号审计预注册 essential variables、ANF、decision tree、ROBDD 和 influence，不在看到候选后只凭自然语言判断。

## 3. 干预与大规模确认

- 对 teacher-free 高共识集加入高分歧一致点、同点冲突标签和低分歧一致点；
- 扫宽度16--1024、深度和 tanh/GELU；
- 大规模扫描8,192个新随机 n=12训练集，64 seeds 筛选，4,096 fresh seeds 确认；
- 预注册 Tier 审计与事后线性阈值 LP 审计严格分栏。

## 4. 边界

高共识只定义函数凝聚，不保证外部正确；符号复杂度必须在训练集相容候选中比较。实验检验单向命题`high consensus -> low readable complexity`，不检验其逆命题。
