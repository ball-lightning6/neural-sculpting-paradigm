# E22：逐样本自由能与信息不变量

## 目的

E22 把“逐个加入样本时的惊讶度能否累积为规则难度”改写为同一静态参数测度下的配分函数问题。它穷举3-bit 输入的全部部分标注状态、256条完整规则和全部样本加入顺序，检验逐步自由能增量是否为路径无关的端点势差。

## 运行

```bash
python experiment_static_free_energy_information_invariant.py
```

- [实验动机与判据](MOTIVATION_AND_PREREGISTRATION.md)
- [结果与边界](RESULTS_AND_CONCLUSION.md)

## SHA256

```text
8042df5e2818145ac3a8323c6343279efa5e2a11d01269c63071cdc125980514  script
8a7d2eb7b02a133cf1fc1a77fa807044dee7c6cdabe9cc8179d51813d55e7a12  result ZIP
```

原始 ZIP 位于`E:\Downloads`，不进入发布包。
