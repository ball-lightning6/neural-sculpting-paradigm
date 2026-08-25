# E25：完整结果与阶段裁决

## 1. 四个样本的任务差异

Stage 0中，按每条轨迹最低 validation BCE 选 checkpoint：`0/1,n=4`的 median validation accuracy 为96.74%、test 为98.31%；`3/8,n=4`仅为70.44%/69.66%。按 median best-validation 至少95%的操作阈值，0/1在 n=4已跨过，3/8直到 n=512才跨过。

相同输入维度、网络、样本数和训练预算因此对应约128倍的样本辨识尺度差异。

## 2. 静态体积预测真实图像

Stage 1九个 loss 截面全部完成。Test static hard accuracy 从浅到深大致为：

```text
0 vs 1: 92.97% -> peak 96.35% -> 96.09%
3 vs 8: 76.69% -> peak 78.13% -> 78.13%
```

四个训练样本下，两任务在`epsilon=0.6`即明显高于随机；深入 loss 总体改善 hard 预测，但不是全程单调。静态分布也明显不同于 matched-loss AdamW，尤其3/8浅层静态准确率约76.7%而 SGD 约60%。

![MNIST 静态分支预测与集中度](../../assets/figures/e25_mnist_static_prediction.png)

## 3. 静态几何重建 NLL U 形

0/1的 test 和 validation soft NLL 在`epsilon=0.0006`附近最低；同一固定训练集的 SGD validation 最低点对应 train BCE 约`1.46e-4`，相差一个预注册网格区间。3/8的 test 最低在0.01、validation 最低在0.03，SGD 最低对应 train BCE 约0.0455，直接落在相邻区间。

因此经典 validation-loss U 形至少存在一个 optimizer 之前的静态几何来源。由于 Stage 0参与选择任务和 loss 覆盖范围，这还不是对全新数字对转折位置的完全盲预测。

## 4. Agreement、accuracy 和校准发生分离

Hard point collision 随 loss 深入：0/1从0.595升到0.979，3/8从0.542升到0.922。但3/8准确率在78%左右停住后仍继续收束。持续预测错误的样本上，真实标签 soft 分支质量在 NLL 最优点后继续下降，解释了 hard accuracy 近乎不变而 NLL 上升：少量边界修正与错误置信度膨胀可以同时发生。

## 5. 体积差与样本复杂度同向

3/8相对0/1少的 median 参数质量从`epsilon=0.6`时0.56个十进制数量级，扩大到`epsilon=4e-5`时55.43个数量级；这与95%辨识阈值`n=512 vs n=4`同向。它构成 Neural K-profile、真实图像预测和样本相变之间的三角测量。

## 6. 边界

- 只覆盖两个二分类任务、tiny MLP 和各一个固定 n=4数据集；
- static soft branch 不是完整 Bayesian posterior predictive；
- 深尾 lineage 有退化，单样本高精度赔率仍需更多粒子和重复；
- 结果证明静态体积有一阶预测力，同时再次否定“SGD 无偏采样静态后验”；
- 本实验不解析为什么该网络语言给3/8分配更小低-loss 体积。
