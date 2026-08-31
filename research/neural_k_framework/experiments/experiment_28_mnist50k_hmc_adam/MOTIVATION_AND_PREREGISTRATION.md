# E28：动机与判决协议

## 1. 问题

小型Boolean任务中的静态体积可能只是玩具现象。E28把完整网络参数全部纳入
HMC，在50,000张真实图像上构造静态函数系综，并与相同架构、切分和初始函数
分布的Adam比较。

## 2. 协议

- 网络：Conv(1,4)->Conv(4,8)->Linear(392,10)，4,266参数；
- HMC：16条chain，beta=n=50,000，30个冻结步长后快照，共480个样本；
- Adam：32 seed、batch 256、50 epochs；
- plain与对应Gaussian prior的MAP两种Adam；
- 无数据增强，validation/test标签不参与训练和停止。

## 3. 判决

主问题是静态HMC能否达到与真实optimizer相同的预测量级，而不是预设HMC必须
显著胜出。还必须报告成员数配平、逐样本配对误差、完整函数多样性和链内/链间
分歧。高accuracy不能自动证明全局MCMC混合。

