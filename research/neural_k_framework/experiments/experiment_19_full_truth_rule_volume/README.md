# E19：完整真值表规则的 loss-resolved 体积

## 目的

E19 用完整16点训练集直接指定每条4-bit 规则，避免部分训练集究竟诱导哪个函数的歧义，并用 constrained SMC 比较不同完整目标达到相同 loss 的参数质量。

运行：

```bash
python experiment_full_truth_rule_volume_smc.py
```

- [实验动机与判据](MOTIVATION_AND_PREREGISTRATION.md)
- [结果与边界](RESULTS_AND_CONCLUSION.md)

脚本 SHA256：`3c15b71d1ac7bc96b2770100adf13642ec82f058ae54c595496f164d29ca813c`。结果 ZIP SHA256：`5dfcdcd7c680f5eb2eef1d2c604ec2e81b72655b4355de969e9627c03781c573`。
