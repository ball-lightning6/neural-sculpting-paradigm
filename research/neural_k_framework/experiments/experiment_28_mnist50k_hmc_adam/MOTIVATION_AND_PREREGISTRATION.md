# E28：动机与判决协议

## 1. 问题

小型Boolean任务中的静态体积可能只是玩具现象。E28把完整网络参数全部纳入
HMC，在50,000张真实图像上构造静态函数系综，并与相同架构、切分和初始函数
分布的Adam比较。

**解释边界（2026-10-21补充，不改变冻结协议）：** 这里改用了Gaussian先验乘以
likelihood的后验目标，不是把Boolean任务的硬loss阈值体积实验直接放大。
本实验只能作为加权后验采样与Adam的对照，不能填补完整MNIST上的硬阈值体积验证。

## 2. 协议

- 网络：Conv(1,4)->Conv(4,8)->Linear(392,10)，4,266参数；
- HMC：16条chain，beta=n=50,000，30个冻结步长后快照，共480个样本；
- 采样目标：Gaussian先验按整训练集likelihood加权；HMC使用全数据梯度，不做硬loss阈值条件化或体积归一化常数估计；
- Adam：32 seed、batch 256、50 epochs；
- plain与对应Gaussian prior的MAP两种Adam；
- 无数据增强，validation/test标签不参与训练和停止。

## 3. 判决

主问题是likelihood加权后验HMC能否达到与同协议Adam相近的预测量级，而不是预设HMC必须
显著胜出。还必须报告成员数配平、逐样本配对误差、完整函数多样性和链内/链间
分歧。高accuracy不能自动证明全局MCMC混合，也不能验证固定loss阈值体积的预测。

