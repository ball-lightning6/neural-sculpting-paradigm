# E24：深尾 Neural K-profile 与排序交叉

## 目的

E24 先在4-bit、54条完整规则上把 Gaussian 参考体积推进到 hard-exact 区域，再选择唯一仍在追赶 parity4 的单点例外规则。随后从共享父系综并行推进两条分支，直接测量体积比是否在更深 loss 发生反转。

## 运行顺序

```bash
python experiment_4bit_deep_volume_flow.py
python experiment_parity4_flip0000_joint_deep_bridge_lockstep.py
```

`experiment_parity4_flip0000_joint_deep_bridge.py`保留同一实验的完整入口；lockstep 版本是用于同步观察交叉并按共享阈值停止的冻结实现。

- [实验动机与预注册](MOTIVATION_AND_PREREGISTRATION.md)
- [结果与边界](RESULTS_AND_CONCLUSION.md)

## SHA256

```text
8a0f603cd83220e58faccba9f16451611d30db52d159bb48b11f2edeffba954a  deep panel script
5e97d88619f6faa2d45b3eba4da46542e911c96941821a849b53912c3a1b5c91  joint script
45de3e1fdfb03163e73db7ba5b00a83010e7bdb2c86cebf1f1d7e2b6f73f13b2  lockstep script
935971c32e35d3cf0a8280629806fe2a1d12ccef48d507949badf99d87443bbe  deep panel ZIP
db3b99d03e3369888d4e973f177d944ea1f5925b0a817aaa52dea35b727b3ad7  joint bridge ZIP
```

原始 ZIP 位于`E:\Downloads`，不进入发布包。
