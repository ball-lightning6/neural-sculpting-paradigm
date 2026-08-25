# E25：MNIST 样本复杂度与静态体积分支预测

## 目的

E25 把 Boolean 真值表方法迁移到真实图像。Stage 0先在两个二分类任务上校准最小训练集和过拟合 loss 区间；Stage 1冻结条件后，从同一训练 loss parent 比较每张未见图像两个标签分支的静态质量，检验其预测准确率、NLL U 形和 SGD 转折位置。

## 运行顺序

```bash
python experiment_mnist_loss_calibration.py
python experiment_mnist_static_branch_prediction.py
```

- [实验动机与冻结协议](MOTIVATION_AND_PREREGISTRATION.md)
- [结果与边界](RESULTS_AND_CONCLUSION.md)

## SHA256

```text
0cbd6099986c973fd77e38397d8f14a878ca7054f4e308e9675dfc819e1c5df5  calibration script
20b19dffdcf254bd0656e83b9b02393e9ee887c0f00f62dc7ed2019d6c766eeb  static prediction script
11e786b30e3e76739e3666f805fbbaaf9060475c021e54c7761b862ace85dae5  calibration ZIP
f104fc963fa9e7afe1531b32d63a2da80796899f84179cd5c68df24de3b05ae9  prediction ZIP
```

MNIST IDX 文件需置于脚本配置的数据目录。原始结果 ZIP 位于`E:\Downloads`，不进入发布包。
