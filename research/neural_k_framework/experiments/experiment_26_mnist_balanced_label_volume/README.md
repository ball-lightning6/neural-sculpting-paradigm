# E26：MNIST平衡无标签划分的静态体积

## 目的

检验在只知道二分类、5:5类别比例和一个标签anchor时，低-loss参数体积能否
在126种候选标签组合中盲选出自然的数字0/1划分。

## 入口

- experiment_mnist_unlabeled_label_volume.py
- MOTIVATION_AND_PREREGISTRATION.md
- RESULTS_AND_CONCLUSION.md

脚本需要本地MNIST IDX文件，默认目录为/root/mnist_dataset。输出目录可用
NSP_MNIST_LABEL_VOLUME_RESULT_DIR覆盖。

## 冻结来源

| 文件 | 开发版来源 | SHA256 |
|---|---|---|
| experiment_mnist_unlabeled_label_volume.py | research/consensus_symbolicity/experiment_mnist_unlabeled_label_volume.py | 1b1d34211b669d0c70c21fdb0345e8df75f048bc2ec94d9ae9664da5e092bee6 |

本地原始结果包为results_mnist_unlabeled_label_volume_package.zip，SHA256：
86b8f81f375509d4bcc44086aefa708fdd5639e0c3c8e175a01f6b1413ee72ec。
原始ZIP不进入发布包。

## 上传边界

本单元只收录5个0、5个1的平衡候选实验。后续取消类别比例约束的512候选实验
及其深尾跟进不进入本次上传。

