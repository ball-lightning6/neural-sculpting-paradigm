# E18：高函数共识与符号可读性

## 目的

E18 在不知道数据生成规则的 teacher-free 随机部分真值表上，寻找跨初始化高度收束的完整函数，并审计这些函数是否属于人类可读的低复杂度符号族；随后进行跨容量、样本干预和8,192数据集大规模复核。

## 文件与运行顺序

```bash
python experiment_8bit_consensus_symbolicity.py
python experiment_8bit_consensus_width_intervention.py
python experiment_8bit_consensus_large_scale.py
```

- [实验动机与判据](MOTIVATION_AND_PREREGISTRATION.md)
- [结果与边界](RESULTS_AND_CONCLUSION.md)

## 脚本 SHA256

```text
ed04dfd5c7c78407124776b6e829abfa73e0832f025964d0afb30c9606e8d561  pilot
5073de1c3d9d1ad870cce02a462ad3fa2814406a2e6a7c793b684907a13e6905  intervention
e7f08889e7e56329bfe207a8ce7168392cbe594316863d878c07aeda72710e9b  large-scale
```

关键 ZIP SHA256：

```text
cc7da4ce381295672ac18b54ef9a39b31b7eab946f754cf218684c8890959375  pilot
4720d2d073119cca52d8654ec6f316514ef99b43f40da229d95b6640884a2fa2  intervention
958e51fc7339810f2783fea46e8f07486f018560aea15ae40660affebfc03845  large-scale
```

原始 ZIP 不发布。
