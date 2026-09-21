# E32：随机记忆缩放

公开实验 E32。固定 8→16→16→1 tanh 协议，在多组随机标签数据上测量有限 loss 窗口的体积收缩系数随记忆样本数的变化。

主要报告为 `RANDOM_MEMORY_SCALING_REPORT.md`。结果支持近似线性响应，并作为 Neural K/实现复杂度的外部构造效度校准；不等同于机器无关 Kolmogorov complexity，也不是架构无关容量定理。
