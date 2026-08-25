# Experiment 1：静态硬条件化不足以描述训练函数分布

## 科学问题

检验强命题：

$$
P_{\mathrm{trained}}(f\mid D)
\stackrel{?}{=}
P_{\mathrm{init}}(f\mid f\models D).
$$

本实验不否定初始化函数先验的简单性偏置，而是判断真实优化是否只是删除不符合训练标签的初始化函数，再从剩余函数中按原比例采样。

## 文件与顺序

### E01a：逐函数分布、排名反转与路径依赖

```bash
python experiment_static_prior_vs_sgd_posterior.py
```

核心设置：

- `3-bit -> 1-bit`，完整256个 hard function 可逐一编号；
- `3 -> 64 x 10 -> 1 tanh`；
- 1,048,576个初始化 prior 样本；
- 每个训练条件8,192个初始化；
- 两个单样本条件和 Rule30 的`k=2/3/4`嵌套条件；
- 首次拟合、post-fit 100/1,000步；
- direct/forward/reverse 三条路径，各4,096个初始化。

**实现边界：脚本历史名称写作`SGD posterior`，实际优化器是 full-batch Adam。**正文和结果不得把它误写成所有 SGD 变体的无条件结论。

默认输出目录：

```text
results_static_prior_vs_sgd_posterior/
```

### E01b：单样本极简函数质量运输

```bash
python experiment_single_sample_prior_dynamics_1024.py
```

核心设置：

- `3 -> 1024 -> 1024 -> 1024 -> 1`；
- GELU + LayerNorm，Adam，`weight_decay=0`；
- 4,096个初始化 prior 样本；
- `000/111 -> 0/1`四个条件，每个128个配对初始化；
- 保存首次拟合和其后1、2步的完整8点 hard function 与 logits。

默认输出目录：

```text
results_single_sample_prior_dynamics_1024/
```

## 预期复现结果

Experiment 1a：

- 首次拟合时，训练分布与 hard-conditioned prior 的 TV 约0.10--0.31；
- post-fit 100步后约0.54--0.64，抽样噪声基线约0.01--0.04；
- `rule30_k3`中主导函数发生排名反转；
- 相同初始化与最终训练集、不同训练历史，post-final 2,000步后仍有约19.5%--28.2%的完整函数不同。

Experiment 1b：

- 初始化 prior 中常数0/1分别约14.01%/13.72%，非常数函数约72.27%；
- 单样本 hard-conditioned prior 给目标常数约27.5%--28.0%；
- 真实训练四组均为128/128在一步后成为目标常数；
- 初始化时已经满足训练点的兼容非常数 seed 也以45/45、47/47全部迁移；
- 训练分布相对静态条件先验 TV 约0.72、JS 约0.52 bit，hard-function 熵约`5 bit -> 0 bit`。

## 理论裁决边界

可以声称：

> 在这些明确协议中，优化器会系统运输已经满足训练集的函数质量；训练后函数分布不是初始化 hard-function prior 的一次被动条件化。

不能据此声称：

- 初始化先验不重要；
- 一切 Bayesian 描述都错误；
- 所有优化器、网络与任务都具有相同坍缩；
- 无限训练时间后不同路径必然永不汇合；
- hard function 稳定意味着 logits 或内部表示停止变化。

## 冻结来源与 SHA256

| 冻结文件 | 开发版来源 | SHA256 |
|---|---|---|
| `experiment_static_prior_vs_sgd_posterior.py` | `research/function_information_conservation/experiment_static_prior_vs_sgd_posterior.py` | `c24388482df5dfb53a6fdcff27a566d9b9b03827b62d7b718b021d9196d44edf` |
| `experiment_single_sample_prior_dynamics_1024.py` | `research/function_information_conservation/experiment_single_sample_prior_dynamics_1024.py` | `7e339772fd3263bfed4e0eacfdfef28ee05d23ef3aad9af3225b4134c53d99ca` |

复制时间：2026-08-25。两份冻结副本与上述开发版源文件逐字一致。

## 相关文档与本地结果缓存

- [实验动机与预注册判据](MOTIVATION_AND_PREREGISTRATION.md)
- [完整结果与阶段裁决](RESULTS_AND_CONCLUSION.md)
- `E01a`原始结果包本地留存在`results/results_static_prior_vs_sgd_posterior.zip`，SHA256：`ddfd912adb0581d174a88401dc7391267c89650752d0fcf30821fa3000b2e2eb`
- `E01b`原始结果包本地留存在`results/results_single_sample_prior_dynamics_1024.zip`，SHA256：`e75fa3084824ab7fc553d4d20861af3135512838bd4c22ead6708e021f28a6b5`

两份 ZIP 只用于本地复核本文数值，不进入最终上传包；实验运行很快，发布内容以脚本、配置说明和关键结果表为准。
