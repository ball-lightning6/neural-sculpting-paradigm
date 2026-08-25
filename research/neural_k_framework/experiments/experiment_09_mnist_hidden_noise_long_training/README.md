# E09：MNIST 80%隐藏标签噪声长训

## 目的

该实验区分两个经常被混写的命题：

1. 含80%错误标签的 MNIST 训练集仍能在早期学到真实数字规则；
2. 网络已经充分记忆80%错误标签后，是否仍保持约91%的干净测试准确率。

实验固定一次标签污染，不向网络提供噪声身份，并把最佳验证点、首次近完全插值和长训终点严格分开。

## 文件

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- [`experiment_mnist_hidden_noise_long_training.py`](experiment_mnist_hidden_noise_long_training.py)

运行：

```bash
python experiment_mnist_hidden_noise_long_training.py
```

## 冻结来源

| 文件 | 开发版来源 | SHA256 |
|---|---|---|
| `experiment_mnist_hidden_noise_long_training.py` | `research/overfitting_related_research/scripts/experiment_mnist_hidden_noise_long_training.py` | `1985627d1c8f81d288aba30ad1337c6f852eff096f882aa4731598cc2ef712ac` |

本地结果包：`E:\Downloads\results_mnist_hidden_noise_long_training.zip`SHA256：`1f54d164256b4df403b5306dd559078f08e372c7f8f87c9f3fca4a7274a9988f`。

ZIP 可快速重建，不进入最终发布包。
