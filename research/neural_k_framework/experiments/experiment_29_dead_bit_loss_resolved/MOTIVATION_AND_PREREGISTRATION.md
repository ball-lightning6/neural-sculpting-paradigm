# E29：动机与判决协议

## 1. 背景

Izmailov等人证明，训练中恒为0的输入特征对应权重在Bayesian posterior中保持
prior；MAP/带正则化SGD把这些权重压向0，因此可能在协变量偏移下更稳。

E29不预设静态方法必须自动删除dead权重，而是问：即使dead方向完全保持prior，
更深loss带来的有效margin是否足以抵御binary反事实翻转。

## 2. 协议

训练完整覆盖三个有效bit的8个状态，但dead bit z恒为0。测试把z改为0.25、
0.5、1或2，目标保持不变。函数panel包含projection、XOR、majority、parity、
MUX。

比较Gaussian constrained SMC、512-seed无衰减Adam、L2/MAP Adam、
MC-NNGP-like kernel和标准温度1posterior直接积分。

## 3. 预注册判决

主比较是z=1时无衰减Adam与matched-loss静态SMC的严格正确质量：

- 五函数平均绝对差不高于0.10；
- 最坏函数绝对差不高于0.20。

L2是机制对照，不要求与静态质量相等。z=2是更强幅度外推，只用于测量适用
边界，不参与主通过判据。

