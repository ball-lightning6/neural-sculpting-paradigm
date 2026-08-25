# E23：完整目标体积前瞻预测数据相变

## 目的

E23 在读取任何随机子集训练结果前，先用完整真值表 SMC 冻结八条8-bit 规则的局部体积收缩分数；随后训练9,736个随机数据集条件，检验静态体积能否前瞻预测`n50/n90`样本相变，并用跨函数族反例压力测试单标量复杂度假说。

## 运行顺序

```bash
python experiment_8bit_rule_volume_preregister.py
python experiment_8bit_volume_to_data_transition.py
```

- [实验动机与预注册](MOTIVATION_AND_PREREGISTRATION.md)
- [结果与边界](RESULTS_AND_CONCLUSION.md)

## SHA256

```text
466a59d8e0a2243a9040d328dfef5e6f11c4ecb9893452b72024c6ecbe02fbb7  volume script
7b14e28ffd97a50da0af424e1bf755e72e82ad894e613cd129bcf00e5f742331  transition script
ebf266355364f25a0e694ea41863190cbf8a546025c1c49fda309fcb5c1172c5  volume ZIP
6e4075cae4fbd7072ab53f1e2764e65bbbf1934ced35f0301da965e8e0cd834d  transition ZIP
```

原始 ZIP 位于`E:\Downloads`，不进入发布包。
