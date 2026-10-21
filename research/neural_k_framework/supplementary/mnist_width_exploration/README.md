# MNIST宽度探索：文献启发的现象复现尝试

本补充保留已有MNIST/MLP训练脚本，供核查双下降说明中的有限训练对照。
它不是新的Neural K测量，也不是已经完成的双下降机制验证，不另立正式实验编号。
本次发布没有运行新的训练，缺失条件没有补算。

## 1. 与文献的关系

脚本注释参考[Nakkiran等的Deep Double Descent](https://arxiv.org/abs/1912.02292)
设置族，但使用MNIST、双隐藏层ReLU MLP和300 epochs，属于改换任务及架构后的
现象复现尝试，不是逐配置的原论文复现。

[Stephenson与Lee（2021）](https://arxiv.org/abs/2108.12006)也研究时间轴双下降。
其原文第5节采用CIFAR10、ResNet18、0至10%的标签噪声和10个随机种子，明确未做
CIFAR100。本地脚本的任务和架构不同，不将其标为该论文的严格复现结果。

## 2. 保存结果的状态

现存记录包含13个完成300 epochs的条件，计划为7个宽度乘2个seed，缺少W4096
的seed1。每个已保存终点仍有非零训练错误。宽度64至2048的双seed平均测试错误率
从16.140%降至8.145%；W4096只有一个seed，测试错误率为8.420%。
这些数据不能单独确认完整容量双下降或严格插值后的第二次下降。

当前脚本默认对15%的样本尝试均匀重标注，允许抽回真类；该比例不等于实际错标率。
原JSONL没有完整运行配置快照，因此不能仅凭当前默认值断言历史环境变量没有改动。
agreement、gzip长度、logit敏感度和权重范数均为代理量，不是Neural K。

## 3. 复现入口

脚本配置在顶部，不使用argparse。需要PyTorch GPU版、NumPy和Matplotlib。
在本目录运行前，设置NSP_MNIST_ROOT为数据根目录；该目录下应有MNIST/raw中的
四个未压缩IDX文件：train-images-idx3-ubyte、train-labels-idx1-ubyte、
t10k-images-idx3-ubyte、t10k-labels-idx1-ubyte。默认根目录为本目录的data。
NSP_NOISE_RATE控制重标注比例；NSP_DD_OUT用于区分输出目录后缀。

运行[现有训练脚本](experiment_mnist_double_descent.py)将执行长训练，而不是只显示
历史结果。应先确认设备空闲、数据来源和预算。本发布版仅调整数据路径配置，并在
CUDA不可用时明确拒绝长时间CPU回退；不改变网络、优化器、随机种子或训练数值核心。
原始脚本、权重和预测记录保留在本地，不随本补充上传可由脚本重建的结果。
