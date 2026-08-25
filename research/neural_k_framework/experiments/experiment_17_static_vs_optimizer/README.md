# E17：静态低-loss 质量与 optimizer 运输

## 目的

E17 用暴力 prior、rare-event constrained SMC、三种 optimizer crossing、32,768-seed AdamW、SMC 深尾起点续训和 seed/time 混合实验，系统检验：静态 loss 体积对真实训练有多强预测力，以及 optimizer 是否只是该静态分布的无偏或二阶采样修正。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- `experiment_static_loss_vs_optimizer_distribution.py`
- `experiment_static_low_loss_constrained_smc.py`
- `experiment_balanced_n10_tanh16_many_seed_retrain.py`
- `experiment_train_from_smc_deep_tail.py`
- `experiment_8bit_seed_time_mixing.py`

推荐运行顺序：

```bash
python experiment_static_loss_vs_optimizer_distribution.py
python experiment_static_low_loss_constrained_smc.py
python experiment_balanced_n10_tanh16_many_seed_retrain.py
python experiment_train_from_smc_deep_tail.py
python experiment_8bit_seed_time_mixing.py
```

`train_from_smc_deep_tail`读取 constrained SMC 的 checkpoint；其余脚本可以独立运行。SMC 达到层数上限后可保持同一结果目录重跑，自动从 checkpoint 续接。

## 冻结脚本 SHA256

```text
f11398741e099168de115cc26795174f4079589f729749f742e710b42a7aa6a2  static/optimizer
22e1dfefe45848de3c5aff5c25163d53f164e18b523b4d48949ab538a72d018c  constrained SMC
4a39fe18290803c4de3a0065be05ccd2b307fbd93b6d09bcd2ace7770079b50f  many-seed retrain
8fca65f0ca1a893276ac5163906d3310e2c365b7a6ff0f6290aefb9634221f74  train from SMC
bfcf7625c6cd8025034b81f4a1320084350348052ab1eb05d2d9996ff796cde7  seed/time mixing
```

## 关键结果包 SHA256

```text
368a94da2e539d96af907cd38bab368e4df4c25a6c9f51dda211a16c5ee01228  static vs optimizer
e68026823ee36de37ec8ac7e0eaa113a439b8c86edb49acca3971e2693f57af8  SMC final continuation
77da455af559a0e11296f0e923be140d2370a8e494df80637353b48c2e1e2f11  32768-seed AdamW
2c6f21f4b6479b75d89eae6b516bb83c9166ccef1e2c4781e17c3f62916f0e0e  train from SMC
a1b344a1094a1cfb8ec2d0bb2b6d6f0c121abf3fb149dfa788ac8ef56e92a6b1  seed/time mixing
```

SMC 多次 ZIP 只是同一 checkpoint 续跑，不分别计为独立实验。原始包位于`E:\Downloads`，不进入发布包。
