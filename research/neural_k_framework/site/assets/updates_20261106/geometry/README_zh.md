# 2026-11-06 增补：体积收缩、径向混合与局部几何

本增补独立核对提交 `3834fb3f6555c7c58218462502635b16b9b377f9`（2026-10-27）之后的三组 2026-11-01 实验，并与既有 E33/E37 的定义对齐。本文于 2026-11-06 整理；实验日期不是本次审计日期。[English](README_en.md) 与本文逐段对应，共享公式、数字和来源。本次只做短 CPU 数据核验，没有训练、新 SMC 或 GPU 运算。

2026-11-07 补正：本轮加权问题的直接证据是第 1.3 节。历史对话中找回了 2026-11-01 的径向重加权命令及输出；本次用同一批 Gaussian 粒子的半径和逐池成本复得其结果。第 1.1–1.2 节保留旧跨测度表和独立均匀球复测作辅助，第 4 节另列固定球面漂移。没有修改原始实验。

## 1. 问题与结论边界

本次回答三个相邻但不同的问题：有限 loss 窗口中的收缩是否近似幂律；Gaussian 下较平直的曲线是否部分来自径向贡献重分配；局部输出几何和末层实现成本能否解释全局收缩斜率的差异。这里的任务都是小型 Boolean 网络上的完整目标约束，不是部分训练集的候选竞争，也不是新的泛化或优化器实验。

**综合结论：有限窗口内可复现的近似收缩规律、以及据此定义的有效复杂度，并非 Gaussian 测度独有。** 均匀球的独立更新核复测、均匀立方体面板和若干固定球面对照，也给出近似线性的 K 曲线及规则相关系数。因此，不能把整个规律解释为 Gaussian 加权制造的假象。这里“复杂度近似稳定”指指定精度坐标下的收缩系数在所测窗口相对平缓，不是 K 本身不再增长，也不是只要质量下降就算斜率稳定。各测度的平直程度并不相同，尤其 t8 后段明显变平。

这是**依赖测度、参数化、规则和窗口的经验律**：加权既改变质量与曲线位置，也能改变斜率和曲率，并非只改一个截距。Gaussian 径向混合可解释部分平直化，但不是非 Gaussian 规律的替代解释。最细局部秩不能直接替代全局斜率；末层积分提供可计算的几何联系，却不证明所有网络复杂度等价于 margin、白化范数或 SLT 学习系数。最低范数也不等于实现集合的总概率质量。

### 1.1 已测测度、窗口与形状

旧公开证据见 E37 原报告（[中文](https://github.com/ball-lightning6/neural-sculpting-paradigm/blob/3834fb3f6555c7c58218462502635b16b9b377f9/research/neural_k_framework/experiments/experiment_37_measure_robustness/README.md) / [English](https://github.com/ball-lightning6/neural-sculpting-paradigm/blob/3834fb3f6555c7c58218462502635b16b9b377f9/research/neural_k_framework/translations_en/post_upload/e37.md)）与固定半径及自然测度增补（[中文](https://github.com/ball-lightning6/neural-sculpting-paradigm/blob/3834fb3f6555c7c58218462502635b16b9b377f9/research/neural_k_framework/experiments/experiment_37_measure_robustness/FIXED_RADIUS_20261027.md) / [English](https://github.com/ball-lightning6/neural-sculpting-paradigm/blob/3834fb3f6555c7c58218462502635b16b9b377f9/research/neural_k_framework/translations_en/post_upload/e37_update_20261027.md)）。链接固定到已提交版本，不依赖未公开原始目录。

下面的 20 规则比较使用 `4→16→1 tanh`、97 维标准化参数、全部 16 个 0/1 输入；均匀球取 R72，宽 Gaussian 取 `sigma=72/sqrt(99)`，球对称 t8 为 `theta=sigma*sqrt(6/G)*Z`、`G~chi²_8` 且与 `Z~N(0,I)` 独立，均匀立方体各坐标独立取 `[-sqrt(3)*sigma,+sqrt(3)*sigma]`。四者匹配先验平均平方范数；t8 的全向量共享一个 G，立方体同时改变方向分布，不能都称作纯径向重加权。旧单位 Gaussian 与球内截断单位 Gaussian 不属于此尺度匹配组。

共同窗口为 `BCE 0.01→0.001`，节点为 `[0.01,0.0063,0.004,0.0025,0.0016,0.001]`，精度坐标为 `a=-ln(expm1(epsilon))`。表中 `c_a=ΔK/Δa` 是整窗口割线，单位为 bit/a；“漂移”是最后相邻段斜率除以第一相邻段斜率再减 1，不是点导数或跨测度系数差。先在概率空间合并各池，再计算 K。R² 只描述整条曲线的近线性，必须与逐段斜率一起看。

| 测度 | 常数 / copy 取反 / parity2 / parity4 的 `c_a` | 20 规则绝对漂移中位数 | 20 规则线性拟合 R² 范围 |
|---|---|---:|---:|
| R72 均匀球 | 0.600 / 1.962 / 5.716 / 12.488 | 13.71% | 0.998712–0.999949 |
| 宽 Gaussian | 0.598 / 1.907 / 5.046 / 8.828 | 2.74% | 0.999715–0.999993 |
| 球对称 t8 | 0.609 / 1.435 / 1.975 / 2.163 | 25.40% | 0.996368–0.999881 |
| 均匀立方体 | 0.628 / 2.668 / 6.948 / 9.938 | 12.84% | 0.999338–0.999902 |

这张表给出正面证据，也标出其强弱：均匀球和立方体可由窗口平均系数作近似描述，但仍有可见漂移；t8 的 parity2 相邻斜率为 `2.248→1.753`（-22.00%），parity4 为 `2.520→1.857`（-26.30%），不能称作与宽 Gaussian 同样稳定。相对于均匀球，宽 Gaussian / t8 / 立方体在终点 K 上的 Spearman 为 `0.99699 / 0.98947 / 0.94887`，在窗口系数上为 `0.99549 / 0.90526 / 0.94135`。主要排序稳健与每条曲线的斜率稳定是不同判断；190 个规则对不是独立重复。三种当时新增测度共 60 作业、每作业 8 池和每池 2048 粒子，均到达 0.001；均匀球取自已有 36 规则面板中的同一 20 条，不与下述独立复测混为一批。

### 1.2 非 Gaussian 近似规律的独立复测与其他对照

更直接的近线性复测来自旧 R72 均匀球：常数 `0x0000`、copy 取反 `0x00FF`、parity2 `0x0FF0`、parity4 `0x6996`，独立新 seed，每规则 8 池、每池 4096 粒子、每层 48 次更新，并由原旋转及径向刷新核改为对称随机游走加硬拒绝。四条曲线全部到达 0.001；它们与主扫描的 64 个共同节点 K 最大差为 0.4468 bit。下表仍用同一 6 节点窗口，未用高 R² 代替漂移检查。

| 规则 | `c_a` / bit/a | 线性拟合 R² | 首段→末段斜率 / bit/a | 相对漂移 |
|---|---:|---:|---:|---:|
| 常数 | 0.599 | 0.999621 | 0.584→0.637 | +9.01% |
| copy 取反 | 1.980 | 0.999887 | 1.931→2.041 | +5.71% |
| parity2 | 5.794 | 0.999795 | 5.580→6.019 | +7.88% |
| parity4 | 12.505 | 0.999350 | 11.703→13.440 | +14.84% |

这支持“非 Gaussian 下也有经独立核复核的有限窗口近似规律”，不支持严格常数斜率。来源报告为 `research/notes/20260913_uniform_ball_euclidean_volume_zh.md`；本次从 `tmp/4bit_uniform_ball_recheck_20260913/*/volume_curve.csv` 复算，并与同目录 `analysis/curve_summary.csv`、`analysis/local_slopes.csv` 交叉检查。这里的路径仅作仓库内来源标识，不是公开附件链接。历史 R24 的 parity2/parity4 未完成窗口，R48 的 parity4 后段明显变陡且有告警，均不算斜率稳定的正例。

| 其他已做对照 | 实际窗口与结果 | 可以支持什么 |
|---|---|---|
| R72 球内截断单位 Gaussian 与无界单位 Gaussian | BCE 0.01→0.001；常数 `c_a` 为 16.864 对 16.893，copy 取反为 36.882 对 36.880；parity4 为 239.062 对 235.376，均保留 Gaussian 深尾告警 | 硬边界不能独自解释均匀球与单位 Gaussian 的数值差；不是新增的一整类稳定重加权扫描 |
| E37 固定 R20/R40 均匀球面 | BCE 0.03→0.001；R40 的 projection / majority3 / parity2 的 `ΔK/log2(30)` 为 2.99 / 5.67 / 14.07；R20 仅 projection 完成，另两规则共 8 池停滞 | 去掉径向迁移后仍有规则相关角向收缩；完整窗口平均值本身不能证明局部斜率稳定，未达不能补零 |
| 本文的四个条件 Gaussian 半径层、R72 球内及 R72/R80/R88/R96 球面 | BCE 0.01→0.001；3 规则、10 条件；增强轮仅 2 规则、6 条件，详见第 4 节 | 独立径向重构、半径依赖和球面形状；不能把全部条件都算作不同的无条件先验 |
| 旧 E37 / supplementary 的 3-bit 均匀球 MSE 目标移动 | R72，MSE=0.01，目标间距 18→20 的 K 增量为常数 / copy / 多数 / parity 的 0.25 / 1.77 / 2.60 / 5.51 bit；R72/R96 在 MSE=0.001 另查端点交互项 | 非 Gaussian 的规则相关目标响应及有限窗口近似可加性；移动目标不是嵌套事件，不能并入 BCE 斜率稳定证据 |

同一 `research/neural_k_framework/supplementary/measure_robustness/RESULTS.md` 还记录输出参考中心与目标同步平移、只移目标、去掉末层 bias、目标半间距 `0.01/0.1/0.5/1/3` 的 MSE 对照。40 条曲线全部达到 `1e-5`；末 4 节点为 `0.001/0.0003/0.0001/1e-5`，报告的深尾斜率约为 `12.2–13.5 bit/decade`。这些是输出事件、坐标或架构的干预，不是另外几种径向先验；其 MSE 精度斜率也不能和上述 bit/a 混算。

### 1.3 同一批 Gaussian 粒子的径向重加权（2026-11-01）

**改变径向密度形状会改变收缩斜率，但在几个非 Gaussian 权重下，parity4 的首末段漂移仍然很小。** 这是对已保留的直接 Gaussian 条件粒子换密度的诊断，不是为每个目标先验重新运行 SMC。copy 取反使用 `results_v1/copy__gaussian`，parity2、parity4 使用 `results_v2_enhanced/parity2__gaussian`、`results_v2_enhanced/parity4__gaussian`。每个节点均为 4 池，首轮每池 768 粒子，增强轮每池 1536 粒子。

主对照改变的是相对于参数空间体积的密度，不是半径的一维密度。取 `d=97`、`sigma0=72/sqrt(99)`；改变 p 时保持平均平方半径不变，`p=2` 正好是原 Gaussian。方向分布不变，径向 Jacobian 在密度比中相消：

$$
\rho_p(\theta)=C_p e^{-(r/a_p)^p},\quad r=\|\theta\|,\quad
a_p^2=d\sigma_0^2\frac{\Gamma(d/p)}{\Gamma((d+2)/p)},\quad
C_p=\frac{p}{S_{d-1}a_p^d\Gamma(d/p)},\quad S_{d-1}=\frac{2\pi^{d/2}}{\Gamma(d/2)}.
$$

在 BCE 节点 `[0.01,0.0063,0.004,0.0025,0.0016,0.001]`，以目标密度与原 Gaussian 密度之比 w 修正各池成本，再平均池概率。表中规则列为 `a=-ln(expm1(epsilon))` 上的 6 点最小二乘直线斜率（bit/a），不是第 1.1 节的割线。漂移为 `100*(末段斜率 / 首段斜率 - 1)`，单位为 %；权重 ESS 占比取各节点、各池的最小值，N 为每池粒子数：

$$
K'_b=K_{0,b}-\log_2\!\left(\frac1N\sum_i w_{bi}\right),\qquad
K'=-\log_2\!\left(\frac1B\sum_b2^{-K'_b}\right),\qquad
q_b=\frac{(\sum_i w_{bi})^2}{N\sum_i w_{bi}^2}.
$$

| `p` | copy 取反 | parity2 | parity4 | parity4 漂移 | parity4 最低权重 ESS 占比 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.8816 | 4.6017 | 7.5530 | -8.93% | 0.00824 |
| 1.5 | 1.9138 | 4.8931 | 8.2793 | -2.88% | 0.21564 |
| 2 | 1.9348 | 5.0735 | 8.8511 | -2.25% | 1.00000 |
| 2.5 | 1.9479 | 5.1934 | 9.2837 | -2.37% | 0.67828 |
| 3 | 1.9561 | 5.2790 | 9.6239 | -1.98% | 0.35339 |

在 `p=1.5/2/2.5/3` 下，三条规则的斜率都随 p 上升，parity4 的末段只比首段低约 2%–3%。**`p=1` 是压力对照，不是可靠正证**：parity4 的最低权重 ESS 占比仅 0.00824。旧 E37 的正式扫描用球对称 t8 替代了原计划的 Laplace；这里的重加权并不是一次独立 Laplace 扫描。

另一对照只改 Gaussian **方差**。令 `h=v/v0`、`v0=sigma0²`，沿用相同节点和合池方法。此处的 `0.9/1/1.1` 不是标准差倍率，不与另一次更宽的标准差扫描合并：

$$
\log w=-\frac d2\log h+(1-1/h)\frac{r^2}{2v_0}.
$$

| `v/v0` | copy 取反 | parity2 | parity4 | 最低权重 ESS 占比：copy 取反 / parity2 / parity4 |
|---:|---:|---:|---:|---:|
| 0.9 | 2.0623 | 5.4174 | 9.4643 | 0.58401 / 0.54808 / 0.53604 |
| 1 | 1.9348 | 5.0735 | 8.8511 | 1.00000 / 1.00000 / 1.00000 |
| 1.1 | 1.8162 | 4.7837 | 8.3510 | 0.56448 / 0.54852 / 0.54935 |

两表的 ESS 只诊断重要性权重是否集中，不是 MCMC ESS；重加权保留原 SMC 的谱系相关，高 ESS 也不能保证未采到的尾部有覆盖。第 4 节的层贡献 `w_j=p_j*A_j/V` 随 loss 改变是固定先验下的条件化，`K_frozen` 是冻结层贡献的平均成本；两者都不是这里的换密度操作。

原命令与输出从历史对话找回，当时没有另存诊断输出文件。[本次复核记录](radial_reweighting.json) 是 2026-11-07 生成的，只含配置、合池节点 K、曲线汇总及来源 SHA256，不含粒子。短 CPU 复算与找回的 21 个斜率、4 个漂移百分数的最大绝对差均为 0；5 个 parity4 最低权重 ESS 占比的最大差为 4.44e-16。这是算术核对，不是统计误差。

在本目录运行[独立脚本](radial_reweighting.py)：第一条是默认的标准库只读检查，从发布的节点 K 复算斜率与漂移并核对配置和两份报告表；不能从 K 重建权重 ESS，也不验证源文件的实际 hash。第二条需原始文件与 NumPy，只读复算 K、权重 ESS 和来源 hash，对照发布记录；只有显式附加 `--write` 才生成 JSON。

```bash
python -I -S -B radial_reweighting.py --public-only
python -B radial_reweighting.py --from-sources
```

## 2. 测量对象与不可混用的协议

对完整输入表的标签符号、logit、平均 BCE、参考测度下的质量和 bit 成本，统一记为：

$$
s_i=2y_i-1,\quad m_i=s_i z_\theta(x_i),\quad
L_f(\theta)=\frac1{16}\sum_{i=1}^{16}\log(1+e^{-m_i}),\quad
V_{\mu,f}(\epsilon)=\Pr_\mu[L_f\le\epsilon],\quad
K_{\mu,f}(\epsilon)=-\log_2 V_{\mu,f}(\epsilon).
$$

当 `epsilon < ln(2)/16 = 0.0433217` 时，任一错误符号或零 margin 都与平均 BCE 上限矛盾，因而全部输入的 hard 标签必然正确。高于此阈值的完整目标事件仍只是 BCE 事件，不能自动视作互斥的 hard-function cell。平均 BCE、逐点 margin 下界与等幅 logit 密度是不同对象。

| 实验 | 实际输入与输出 | 参数测度或局部目标 | 主分析窗口 |
|---|---|---|---|
| 跨架构 | 全部 16 个 4-bit 输入；MLP/CNN 编码为 ±1，GRU/Transformer 使用 bit token；单 logit | 全部可变参数的独立标准白化 Gaussian；fan-in 缩放 | BCE 0.03 到 0.0001，11 节点 |
| 径向混合 | 全部 16 个 0/1 输入；单 logit | 97 维标准化坐标的宽 Gaussian、条件 Gaussian 半径层、均匀球内或均匀球面 | BCE 0.01 到 0.001，6 节点 |
| SLT 局部探针 | 与所选跨架构 MLP 相同；保持连续 logits 的辅助误差 | 锚点附近的定位 Gibbs 目标，不是原全局先验 | 受控 BCE 锚点 0.03、0.003、0.0001 |
| 末层积分 | 原跨架构全局 SMC 的完整末态粒子 | 原标准白化 Gaussian 条件于 BCE 事件 | 点导数 0.0001；割线 0.0002 到 0.0001 |

跨架构主表先平均各池 K；径向主表先平均概率，再取负对数。两种汇总不相等，不能把其差异当作科学效应。三个精度坐标也必须区分：

$$
\overline K=\frac1M\sum_{r=1}^M K_r,\qquad
K_{\rm pooled}=-\log_2\!\left(\frac1M\sum_{r=1}^M2^{-K_r}\right),\qquad
t=\ln(1/\epsilon),\quad u=\log_2(1/\epsilon),\quad
a(\epsilon)=-\ln(\operatorname{expm1}(\epsilon)).
$$

若 `V ≈ C epsilon^kappa`，则 `dK/dt = kappa/ln(2)`，而 `dK/du = kappa`。径向表使用 `dK/da`；有限 epsilon 下 `a` 与 `t` 不是完全相同的坐标。拟合截距不作为 hard-function 先验质量估计。

## 3. 跨架构有限窗口收缩

### 3.1 实际模型与采样协议

源码、结果 identity 中的参数形状以及参数数逐项一致。下表的参数数是参与测度和采样的维数，包含可变 bias；不包含固定的位置编码或固定归一化 affine 常数。

| 配置 | 实际宽度、深度与读出 | 采样参数数 |
|---|---|---:|
| tanh16 | 4→16→1；1 个 tanh 隐层 | 97 |
| tanh32x3 | 4→32→32→32→1；3 个 tanh 隐层 | 2305 |
| relu32x3 | 4→32→32→32→1；3 个 ReLU 隐层 | 2305 |
| cnn8x2 | 1D 卷积通道 1→8→8，kernel 3、padding 1、ReLU；长度 4 展平为 32 后线性读出 | 265 |
| gru16 | 2×4 token embedding；1 层 hidden 16 GRU，展开 4 步，最终 hidden 线性读出 | 1081 |
| transformer8 | 3×8 embedding（含 CLS）；4 个输入 token 加 CLS；1 个宽度 8、2 heads、FF 16 的 ReLU encoder block，CLS 线性读出 | 601 |

Transformer 的两处 LayerNorm 都固定 gain=1、bias=0，epsilon=1e-5；对应 32 个固定 affine 数值不计入 601 维。它不是既有 E33 条件推导中随机 LayerNorm affine 与 Gaussian 读出共同放大的同一协议。MLP/CNN 的权重和 bias 都按所属权重的 fan-in 开方缩放；embedding 标准差为 1。GRU 的时间展开步数不等于堆叠层数。

规则为第一位 copy 与前两位 XOR（parity2），每条规则约束全部 16 点，无幅度正则、等幅限制或 SGD。预定 BCE 节点为 `[0.3,0.1,0.03,0.02,0.01,0.006,0.003,0.002,0.001,0.0006,0.0003,0.0002,0.0001]`。独立 384 粒子 pilot 用存活分位数 0.6 选择中间阈值，并调节 block pCN 步长；正式池冻结该日程，累计实际幸存比例，不用固定层保留率代替实测质量。

主预算为每条件 3 池、每池 768 粒子、12 block sweeps；增强预算只复核各配置 parity2，为每池 1536 粒子、24 sweeps，使用独立正式随机流但共享该条件的 pilot。原记录为 Python 3.13.5、PyTorch 2.11.0+cu130、CUDA 13.0、RTX 3080、cuda:1、float64，TF32 关闭。报告的采样耗时不作为本次审计耗时。

### 3.2 主要结果

主预算的最深 K 及窗口斜率如下。前窗口为 `0.03→0.006`，后窗口为 `0.0006→0.0001`；斜率单位均为 bit/nat。未将增强曲线混入主表。

| 配置 | copy 最深 K | parity2 最深 K | copy 前→后斜率 | parity2 前→后斜率 |
|---|---:|---:|---:|---:|
| tanh16 | 190.00 | 709.63 | 21.24→24.98 | 75.88→85.32 |
| tanh32x3 | 226.15 | 452.54 | 24.81→26.82 | 43.74→39.50 |
| relu32x3 | 172.90 | 281.48 | 17.33→13.17 | 26.14→19.38 |
| cnn8x2 | 152.08 | 274.10 | 15.71→12.89 | 27.56→21.66 |
| gru16 | 307.24 | 441.80 | 28.61→24.81 | 39.53→38.69 |
| transformer8 | 102.57 | 135.40 | 8.70→13.50 | 10.28→14.27 |

12 条主曲线在 11 个低-loss 节点上的直线拟合 R² 为 0.99246–0.99995，RMSE 约占 K 跨度的 0.22%–2.76%。但仅用较浅 8 节点拟合后，对最深 3 节点的预测 RMSE 为 0.30–10.95 bit；1 bit 已对应 2 倍质量差。因此“图上近似直线”不等于精确体积预测或恒定斜率。

同为 2305 参数的 tanh32x3 与 relu32x3 仅改变激活函数，却具有不同曲率方向。六配置中 parity2 相对 copy 的 K 劣势随阈值收紧扩大，但这只有两条规则，不能推出所有函数的复杂度定理。`K=a+b*t^p` 的自由 p 仅为描述性拟合；主预算 copy/parity2 的 p 分别为 tanh16 1.27/1.14、tanh32x3 1.08/0.81、ReLU 0.59/0.55、CNN 0.69/0.61、GRU 0.69/0.90、Transformer 1.64/1.43，不是已识别的最终渐近指数。

### 3.3 完成状态、预算敏感性与负结果

最终原始结果为 18/18 曲线、54/54 池到达 0.0001；最终失败曲线为 0。历史 `results_v1/progress.json` 保留 3 次文件访问失败，不计作额外重复；`results_v2/progress.json` 还保留 3 次增强入口 identity 元组/列表比较错误。后者在正式增强采样前发生，最终状态须结合 `additional_budget_checks.json` 的 6 项完成记录和实际 result 文件判断，不能只读旧 progress。

| parity2 配置 | 主预算最深 K | 增强最深 K | 增强减主预算 K | 窗口 ΔK 的变化 |
|---|---:|---:|---:|---:|
| tanh16 | 709.63 | 708.56 | -1.07 | -0.70 |
| tanh32x3 | 452.54 | 451.54 | -0.99 | -0.38 |
| relu32x3 | 281.48 | 280.19 | -1.29 | -0.02 |
| cnn8x2 | 274.10 | 274.78 | +0.68 | +0.69 |
| gru16 | 441.80 | 429.21 | -12.59 | -10.48 |
| transformer8 | 135.40 | 134.62 | -0.77 | -0.71 |

差值由未舍入值计算。除 GRU 外，低-loss 窗口收缩增量的变化不超过约 0.71 bit，支持这批 parity2 曲线形状的预算稳定性，而非全部绝对质量已收敛。GRU 最深差 12.59 bit，对应约 6000 倍质量差；没有第三档预算或独立全局估计器，不能称其精度已站稳，也不能仅凭此把它认定为体积律的反例。copy 没有增强预算。池间 SD、接受率、曲线平滑和祖先数均不能排除共同模式遗漏；祖先数不是 ESS。

## 4. 半径条件体积与 Gaussian 径向混合

### 4.1 模型、测度与实际范围

径向实验为 `4→16→1 tanh`、97 参数、0/1 输入；第一层 80 个标准化参数除以 2，末层 17 个除以 4。规则编码依次为 copy 的取反 `0x00FF`、parity2 `0x0FF0`、parity4 `0x6996`，不可将第一条的标签直接写成跨架构实验的 copy。半径在这套标准化坐标 theta 中定义，不是实际缩放后权重总范数，也不是内部单位 Gaussian 坐标 u 的范数。

$$
\theta=\sigma u,\quad u\sim N(0,I_{97}),\quad
\sigma=72/\sqrt{99}\approx7.23627227,\qquad
\mathbb E\|\theta\|^2=97\sigma^2\approx5079.272727.
$$

这与 R72 均匀实心球匹配平均平方半径，而不是与 R72 球面匹配。十种测度条件为直接 Gaussian、四个条件 Gaussian 半径层 `[0,72)`、`[72,80)`、`[80,88)`、`[88,+inf)`、R72 均匀实心球，以及 R72/R80/R88/R96 均匀球面。层先验质量为 0.575510896、0.381866476、0.041947433、0.000675196。条件半径层仍有层内径向混合；只有球面对照完全固定半径。

首轮为 3 规则×10 测度条件，每条件 4 池、768 粒子、普通层 16 次更新、预定节点 48 次更新；256 粒子独立 pilot 冻结阈值与提案尺度。节点为 `[0.7,0.3,0.1,0.03,0.016,0.01,0.0063,0.004,0.0025,0.0016,0.001]`。增强轮仅测 parity2/parity4 的直接 Gaussian、四半径层和 R72 球面，独立新 seed，1536 粒子、32/96 次更新。两轮均为 float64、cuda:1、RTX 3080，TF32 关闭并开启确定性运算，无梯度预热。

首轮 30/30 条件、120/120 池、330 条件-节点完成；增强轮 12/12 条件、48/48 池、132 条件-节点完成；两轮最终失败条件均为 0。两轮不合并为同预算重复，也不把 22 个增强规则-节点说成独立任务。设计是在查看旧面板之后提出的机制检验，不声称盲预测或正式预注册。

### 4.2 重构与曲率分解

对球对称 Gaussian，完整径向密度必须包含 Jacobian；固定半径角向概率与 Gaussian 总质量有以下关系。四层重构使用各条件先验的独立样本，而非把直接 Gaussian 终点样本拆层。

$$
A_f(r,\epsilon)=\Pr_U[L_f(rU)\le\epsilon],\quad U\sim\operatorname{Unif}(S^{96}),\qquad
V_f(\epsilon)=\int_0^\infty A_f(r,\epsilon)\rho_\sigma(r)\,dr,\quad
\rho_\sigma(r)\propto r^{96}e^{-r^2/(2\sigma^2)}.
$$

$$
V(a)=\sum_jp_jA_j(a),\quad K_j=-\log_2 A_j,\quad
w_j(a)=\frac{p_jA_j(a)}{V(a)},\qquad
K''(a)=\mathbb E_w[K_j''(a)]-\ln(2)\operatorname{Var}_w[K_j'(a)].
$$

可微且允许交换求导与积分时，层间重新分配可抵消层内向上曲率，但不保证总曲线为直线。这里的 `A_j` 是条件 Gaussian 半径层概率，不是某个单一球面的 `A_f(r,epsilon)`。四层覆盖全部半径空间，没有用稀疏球面插值代替连续积分。

冻结起点 `epsilon=0.01` 权重的平均成本诊断满足精确的有限分层恒等式；它不是另一个实际混合测度的概率成本。

$$
K_{\rm frozen}(a)=K(a_0)+\sum_jw_j(a_0)[K_j(a)-K_j(a_0)],\qquad
K_{\rm frozen}(a)-K(a)=D_{\rm KL,bits}(w(a_0)\|w(a)).
$$

### 4.3 主要结果与固定球面对照

全部节点上，分层重构与独立直接 Gaussian 的最大 K 点估计差，首轮为 0.414 bit，增强轮为 0.085754 bit。下表曲率单位为 bit/a²，末列为 bit；混合与冻结列来自整窗口二次拟合，中心两列来自各层二次拟合的中心导数，故不要求逐列精确相减。

| 预算/规则 | 混合曲率 | 冻结曲率 | 中心平均层内曲率 | 中心层间方差项 | 终点冻结成本差 |
|---|---:|---:|---:|---:|---:|
| 首轮/copy 取反 | 0.024 | 0.036 | 0.036 | 0.011 | 0.029 |
| 首轮/parity2 | 0.045 | 0.138 | 0.137 | 0.090 | 0.257 |
| 首轮/parity4 | -0.098 | 0.244 | 0.206 | 0.296 | 0.837 |
| 增强/parity2 | 0.091 | 0.213 | 0.207 | 0.114 | 0.316 |
| 增强/parity4 | -0.114 | 0.278 | 0.230 | 0.353 | 1.014 |

增强 parity4 的 R≥88 条件贡献从 15.54% 升至 58.69%，而该层原先验质量仅为 0.06752%；这是条件筛选的重分配，不是采样时修改先验。增强 parity2 的混合曲率 95% 池 bootstrap 区间为 [0.042,0.139]，parity4 为 [-0.224,-0.015]，故不能声称严格恒定指数。负曲率表示仍为正的收缩速度下降，并非质量增加。

首轮各固定球面的窗口线性斜率如下，单位为 bit/a。固定半径仍有规则相关收缩，且不同半径不共享相同系数。

| 规则 | R72 | R80 | R88 | R96 |
|---|---:|---:|---:|---:|
| copy 取反 | 1.930 | 1.709 | 1.536 | 1.343 |
| parity2 | 5.628 | 4.751 | 4.148 | 3.730 |
| parity4 | 12.503 | 10.309 | 8.694 | 7.762 |

同一 6 节点窗口中，首轮 parity4 球面曲线的首尾相邻段斜率漂移在 R72/R80/R88/R96 分别为 `+19.92% / +8.84% / +4.97% / +5.09%`；增强 R72 仍为 `+16.87%`。因此部分非 Gaussian 球面的斜率相当平缓，但不是全部球面同样稳定；R72 的较高直线拟合 R² 也没有抹去残余曲率。这些数字由 [radial_curves.csv](radial_curves.csv) 的逐池 K 在概率空间合并后短 CPU 复算，不是新采样。

### 4.4 不确定度与负结果

总质量接近不意味着参数分布相同。增强轮直接 Gaussian 与重构四层分布的平均/最大 TV，parity2 为 0.0269/0.0423，parity4 为 0.0351/0.0598。parity4 终点重构 K 差的 95% bootstrap 区间仍为 [-0.606,0.437] bit，窗口增量差为 [-0.268,0.291] bit。区间来自每条件 4 池整曲线重抽样，条件于冻结 pilot，不包含共同混合偏差；本次核对保存区间，没有重新执行 bootstrap 或读取粒子复算 TV。

旧 2026-09-13 面板的事后复算有 96 条曲线，20 条尺度匹配规则中 20/20 的 Gaussian 首尾斜率相对漂移小于均匀球；绝对漂移中位数为 2.74% 对 13.71%。这只是指定面板与指定指标，不是独立随机任务上的普适性检验。旧样本粗分层是机制线索，不与新增独立分层重构混算为重复证据。

原数值审计对两轮合计 682368 组参数、462 个条件-节点复算前向；本次只核对其保存摘要与来源，不冒称重新检查了这些粒子。首轮 parity4 某些球面池只剩 1–4 个初始祖先，增强轮最少为 8，均不能据此计算 ESS。独立 chi CDF 核验曾触发初始机器精度容差，实际最大绝对差为 2.62e-10，后采用 1e-8 绝对/1e-7 相对容差并保留误差；不是采样失败，也未修改采样值。

## 5. SLT 局部探针的正面证据与失败边界

### 5.1 锚点和局部目标

选取 tanh16（97 参数）与 relu32x3（2305 参数）的 copy/parity2 四条件。从每条件原主预算的 3 池各取 2 个末态参数，共 24 个原型。固定隐藏参数，只同步缩放末层权重和 bias，使 BCE 达到 0.03、0.003、0.0001，产生 72 个受控锚点。这不是在三个阈值重新采得的全局条件分布，缩放过程也不是训练。训练点 margin 全正后，仅放大末层即可继续降低 BCE，因此更低 BCE 不必然要求新特征；这条路径的存在也不决定其原测度质量。

局部采样在全部参数方向上进行，使用连续 logit 保持误差而非把非驻点 BCE 当作局部极小值。固定线性预条件变换不改变目标；定位 Gaussian 与原全局 Gaussian 不同。

$$
R(\theta;\theta_0)=\frac1{32}\sum_{i=1}^{16}[z_\theta(x_i)-z_{\theta_0}(x_i)]^2,\qquad
q_{\beta,\gamma}(\theta\mid\theta_0)\propto
\exp[-\beta R-\gamma\|\theta-\theta_0\|^2/2].
$$

测量量为有限尺度成本 `beta E[R]`、logit Jacobian 的 Gram 谱和 BCE 广义 Gauss-Newton 谱。若 lambda 是对应 Gram 特征值，谱参与维数为：

$$
d_{\rm eff}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2},\qquad
\beta\mathbb E[R]_{\rm linear}=\frac12\sum_i\frac{\beta\lambda_i}{\gamma+\beta\lambda_i}.
$$

### 5.2 局部结果

全部 72 个锚点在相对奇异值阈值 1e-6、1e-8、1e-10、1e-12 下均得到秩 16；局部二次输出保持模型的极细尺度基准为 r/2=8。这是数值秩而非精确秩证明。下表使用 BCE=0.0001 的 6 锚点均值；成本为 gamma=1e7 的 pCN-MH 均值。

| 配置/规则 | 秩 | logit 谱参与维数 | BCE GGN 谱参与维数 | 成本 beta/gamma=100 | 成本 beta/gamma=10000 |
|---|---:|---:|---:|---:|---:|
| tanh16/copy | 16 | 5.36 | 4.78 | 5.76 | 7.93 |
| tanh16/parity2 | 16 | 7.08 | 6.71 | 6.42 | 7.99 |
| relu32x3/copy | 16 | 4.18 | 3.71 | 7.57 | 8.02 |
| relu32x3/parity2 | 16 | 5.70 | 4.83 | 7.83 | 7.92 |

有限分辨率谱与较低探测强度的均值在这两个架构内都把 parity2 排在 copy 之上，但高探测强度趋近 8；ReLU 的细微倒序不能作为稳定复杂度翻转。相同局部秩不能解释差异很大的全局 BCE 斜率，连续输出保持成本也不能直接命名为 hard rule 的复杂度或新的 Neural K。

### 5.3 首轮失败与强定位复核

首轮 gamma=1000、10000，beta/gamma=100、10000，SGLD 步长 0.04/0.02；tanh 两条件完成，ReLU/copy 在第 5 步发生非有限能量，已完成的 pCN 也有零接受率链，ReLU/parity2 未启动。保留 2 完成、1 失败、1 未启动，不将首轮作为可靠平衡态重复。失败说明在 `results_v1/FAILURE_NOTE.md`，该失败条件没有最终 `status.json`，不能因缺文件而从失败计数中消失。

正式复核改为 gamma=1e6、1e7，保持 beta/gamma=100、10000。pCN-MH 使用 rho=0.6、burn=250、draws=1000；固定 SHA256 的 vendored devinterp 1.3.2 SGLD 核使用步长 0.02、burn=600、draws=2400，半步长 0.01 时为 1200/4800；thin=1。四条件全部完成，最终失败为 0。共 72×4×3×3=2592 条局部链运行，不是 2592 个独立数据集。

pCN 各链最低接受率为 0.293；两档 SGLD 与 pCN 的组均值最大相对差为 4.87%/5.80%，pCN 与局部线性解析基准的最大组均值差为 0.1316。仍有链内漂移，不把几百分之一的差异当作稳定排序。原官方单步、二次/四次目标校准及预条件检查通过，不能由此证明任意非线性局部目标都充分混合。

## 6. 末层几何：条件模型与全分布积分

### 6.1 固定隐藏表示不等于所有网络

给定隐藏参数 eta，将末层缩放与 bias 吸收进特征矩阵 B_eta，独立标准白化读出 w 满足下式。单位共同 margin 的最低读出代价与单位等幅输出的最低范数是不同约束；后一等式要求 Sigma_eta 可逆，否则必须检查可达性，不能任意添加 jitter 后声称原密度成立。

$$
z=B_\eta w,\quad w\sim N(0,I_k),\quad \Sigma_\eta=B_\eta B_\eta^T,\qquad
C_\eta(s)=\frac12\inf_{\operatorname{diag}(s)B_\eta w\ge\mathbf1}\|w\|^2,\qquad
\inf_{B_\eta w=as}\|w\|^2=a^2s^T\Sigma_\eta^{-1}s.
$$

这些是固定表示、指定白化坐标下的条件几何，不是“所有网络复杂度等价”的定理。对固定可行表示，Gaussian 大幅度尾部的主导代价可以联系到最低范数；有限幅度总质量仍包括可行方向、邻域体积、隐藏表示的先验稀缺性和不同实现的积分。改变表示、参数化、参考测度或输出事件会改变答案。完整网络的事件质量仍必须积分：

$$
V_f(\epsilon)=\int \Pr_w[L_f(\eta,w)\le\epsilon\mid\eta]\,\mu_\eta(d\eta).
$$

本轮没有重新求全网络最低范数，没有验证无限尾部，也没有把一个最佳实现的代价当作上式积分。既有 E33 的条件放大阶数推导仍是条件解释稿，不升级为跨架构普适证明。

### 6.2 原 Gaussian 条件分布下的末层恒等式

与固定隐藏表示的示意不同，这项诊断对原 SMC 全部末态粒子取条件期望，包含隐藏表示的分布。设 k 包括末层 bias，L 为平均 BCE，m 为正 margin；在末层线性、标准白化 Gaussian、epsilon<ln(2)/16、边界正则及无穷远通量消失等条件下，带 Gaussian 密度的散度公式给出：

$$
g=w\cdot\nabla_wL=-\operatorname{mean}_i[m_i\sigma(-m_i)],\qquad
h=w\cdot\nabla_wg=g+\operatorname{mean}_i[m_i^2\sigma(-m_i)\sigma(m_i)],
$$

$$
\kappa(\epsilon)=\frac{\epsilon V'(\epsilon)}{V(\epsilon)}
=\mathbb E\!\left[1+\frac{L(k-\|w\|^2)}g-\frac{Lh}{g^2}\ \middle|\ L\le\epsilon\right].
$$

这里 sigma 是 logistic 函数，不是前文 Gaussian 尺度。恒等式右侧不使用累计幸存比例或拟合斜率，但仍使用 SMC 条件粒子，因此不是独立采样验证。固定前层逐个积分可避免把 ReLU 隐层不可微边界误当成末层求导问题；它也不证明 kappa 在不同阈值恒定。

### 6.3 末态结果与可支持的解释

点导数位于 epsilon=0.0001；割线使用 0.0002→0.0001。两列单位均为 bit/log2 精度单位，即 kappa，不是 bit/nat；点与窗口平均不要求完全相等。每条件 3 池等权平均，增强行不与主预算合并。

| 配置/规则 | 预算 | 末层维数 k | 积分点斜率 | SMC 割线斜率 |
|---|---:|---:|---:|---:|
| tanh16/copy | 1 | 17 | 17.964 | 17.611 |
| tanh16/parity2 | 1 | 17 | 61.341 | 60.281 |
| tanh16/parity2 | 2 | 17 | 60.708 | 59.717 |
| relu32x3/copy | 1 | 33 | 8.689 | 8.757 |
| relu32x3/parity2 | 1 | 33 | 12.818 | 12.959 |
| relu32x3/parity2 | 2 | 33 | 12.876 | 13.057 |

四个主预算条件的点/割线相对差绝对值为 0.78%–2.00%。tanh 的点导数较高、ReLU 较低，与各自曲率方向相容。主预算 tanh/copy 与 tanh/parity2 的范数贡献分别为 19.714 与 63.086，维数贡献为 -1.857 与 -1.853，导数修正为 0.107 与 0.108；差异主要落在 margin 加权的末层白化范数贡献，而不是局部秩。这是条件实现成本的联系，不是最低范数或独立逻辑复杂度的普遍等号。

原恒等式校准包含 5 个一维解析尾概率与 3 个二维平均 BCE 曲边积分案例，最大相对差为 6.00e-7。本次复核各池分解相加与原 SMC 割线，未重新读取 checkpoint 计算 margin、范数或积分；其 checkpoint SHA256 在机器摘要中明确标记为原报告记录、未重新哈希。粒子标准差不是标准误，重采样谱系也不增加独立任务数。

## 7. 本次独立核验与可复现来源

[recompute.py](recompute.py) 提供公开包核算与完整来源审计两种模式，均只使用 Python 标准库，不导入 Torch、NumPy、训练入口或历史实验模块。完整来源审计读取原始逐层/逐池 JSON、谱和链摘要，独立重建幸存比例账本、直线/自由 p 拟合、半径层概率合并、KL 恒等式、二次曲率、谱参与维数和末层分项，并用半整数 chi CDF 递推复核半径层先验质量。保存的审计记录涉及 244 个来源文件的 SHA256、冻结源码 hash、pilot/正式日程和配置签名。该模式依赖本包之外的原始结果、冻结源码及上下文说明；下表是该完整审计的误差，不是 `--public-only` 重新验证了原始来源的结果。

| 本次复算项目 | 最大绝对差 |
|---|---:|
| 跨架构逐层累计 K / bit | 1.93e-12 |
| 跨架构留出预测 RMSE / bit | 1.71e-13 |
| 径向逐层累计 K / bit | 1.28e-13 |
| 半整数 chi CDF 层质量 | 2.62e-10 |
| 径向有限分层 KL 恒等式 / bit | 1.37e-14 |
| 径向混合曲率 / bit/a² | 6.05e-14 |
| 谱参与维数 | 3.55e-15 |
| 局部成本组均值 | 1.78e-15 |
| 末层分项相加 | 1.42e-14 |

这些是实现与记账一致性的误差，不是体积估计的统计误差或置信区间。原前向审计、bootstrap 区间和径向 TV 的再计算不在本次 CPU 核验范围内；2026-11-06 的核验对旧 96 曲线面板仅复核配对表与汇总，没有重读其全部历史粒子。没有复制 checkpoint、粒子数组或大型原始文件。

2026-11-07 的补核另以标准库从 `research/radial_mixture_diagnostic_20261101/historical_analysis/all_curves.csv` 复算 96 条曲线的首末斜率、线性拟合及残差，与 `shape_statistics.csv` 的对应值最大绝对差为 `1.78e-14`；随后只取四种测度共有的 20 规则，得到第 1.1 节的漂移及 R²。第 1.2 节四条独立均匀球复测的窗口割线、R² 与相邻斜率，对保存表的复算最大差为 0。这两项旧表核对没有重读粒子或重做 bootstrap；旧面板来源说明见 `research/notes/20260913_natural_measure_symmetry_rank_zh.md` 和 `research/notes/20260913_uniform_ball_36_rule_rank_comparison_zh.md`。这些补核不在原 `audit_summary.json`、`source_manifest.json` 或 `recompute.py` 的覆盖范围内；第 1.3 节的径向重加权另有独立脚本及来源 hash。

| 公开文件 | 内容 |
|---|---|
| [audit_summary.json](audit_summary.json) | 参数维数、18 条曲线拟合、预算差、径向分解、局部/末层摘要及本次误差 |
| [cross_architecture_curves.csv](cross_architecture_curves.csv) | 234 个曲线-节点的 3 池 K，不混合预算 |
| [radial_curves.csv](radial_curves.csv) | 462 个条件-节点的 4 池 K，保留版本与测度身份 |
| [radial_reweighting.json](radial_reweighting.json) | 第 1.3 节两类重加权的 24 条曲线、144 个合池节点 K、形状与最低权重 ESS 汇总、24 个来源文件的 SHA256 |
| [radial_reweighting.py](radial_reweighting.py) | 第 1.3 节的标准库公开检查及 NumPy 来源复算，不启动采样 |
| [source_manifest.json](source_manifest.json) | 仓库相对来源路径、字节数及完整 SHA256；包括原始结果、配置、状态、源码和上下文说明 |
| [recompute.py](recompute.py) | 公开包核算、完整来源审计与双语结构检查，不启动采样 |
| [test_public_only.py](test_public_only.py) | 脱离原始来源的副本测试，检查数据缺失、修改、重复节点与写入拒绝 |

以下原始来源仅以路径标识，不是本包的下载链接；完整 SHA256 见 [source_manifest.json](source_manifest.json)。清单记录审计时的来源路径、字节数和 hash，不代替原始数据，也不证明这些来源已公开下载或实验已独立复现。上下文笔记仅保留来源元数据，正文不随本包发布。

| 主题 | 来源路径与角色 |
|---|---|
| 跨架构原始数据 | `research/cross_architecture_contraction_20261101/results_v2/*/main_x*/result.json`；逐层幸存率、逐池 K、identity 与状态 |
| 跨架构冻结源码 | `research/cross_architecture_contraction_20261101/source_snapshot_v2/research/`；包含实际模型依赖，不用修改后的入口替代旧 hash |
| 径向原始数据 | `research/radial_mixture_diagnostic_20261101/results_v1/` 与 `results_v2_enhanced/` 下的 result、identity、pilot、manifest |
| 径向源码与审计 | `research/radial_mixture_diagnostic_20261101/results_v1/source/`、`analyze.py`、两轮 `AUDIT.json` |
| 局部原始数据 | `research/slt_local_probe_20261101/results_v2/*/geometry.json`、`pcn_mh.json`、`sgld.json`、`sgld_halfstep.json` 与 status |
| 局部失败和源码 | `research/slt_local_probe_20261101/results_v1/FAILURE_NOTE.md`、`experiment_v1_snapshot.py`、`experiment_v2_snapshot.py`、vendored SGLD |
| 末层恒等式 | `research/slt_local_probe_20261101/readout_identity.py`、`readout_identity_results.json`、`READOUT_IDENTITY_ZH.md` |
| 既有说明与边界 | E33 `OUTPUT_CONTRACTION_DERIVATION.md`；E37 `FIXED_RADIUS_20261027.md`；`research/notes/20261105_amplitude_alignment_mechanism_revision_zh.md` 与 `20261105_core_volume_generalization_literature_review_zh.md` |

只下载本包后，可在解压目录运行以下只读检查。`--public-only` 不寻找或读取原始 `research/` 目录；附加 `--documents` 会检查本包双语标题层级、段落结构、公式、数字和链接。

```bash
python -I -S -B recompute.py --public-only --documents
```

公开模式从 234 个跨架构曲线-节点、462 个径向条件-节点的逐池 K 复算端点、窗口斜率、拟合与预算差、概率合并、径向重构、曲率和 KL 恒等式；并用 CSV 割线及 JSON 已存分项核对 6 行末层摘要。它检查 `source_manifest.json` 的格式，不读取其中的来源文件来验证 hash。SLT 原始谱、局部链统计、终态和失败日志、历史面板、bootstrap/TV 与粒子前向没有在公开模式重新验证；这些结果和限制按保存的摘要保留。两份派生文件的数字相容不证明原始数据真实性、独立实验复现或全局混合。

完整来源审计需要本包处于原仓库目录层级，且来源清单中的原始材料齐备；此时在本目录运行下列命令。`--write` 可重新生成本包的四个机器可读文件，但只适用于完整模式；与 `--public-only` 同用会被拒绝。缺少原始来源时应使用公开模式，不能把本包视为 SMC 原始运行的完整复现材料。

```bash
python -I -S -B recompute.py --documents
```

仅下载本包不能重建全部逐层幸存率账本或复现 SMC 原始运行。完整实验复现还需原始协议、冻结运行源码、依赖环境与适当采样预算；本包不启动这些计算。可另运行 `python -I -S -B test_public_only.py`，测试在仅含公开文件的临时副本中完成并自动清理，不读取原始实验目录。

## 8. 与 E33/E37 的关系及证据缺口

第 1 节把旧 E37、自然测度面板与独立均匀球复测接回综合结论；第 3 节补充 E33 的跨架构有限窗口实测；第 4 节补充 E37 的固定半径与测度分析；第 5–6 节说明两者相关的局部几何和末层积分边界，不新增实验编号。非 Gaussian 的有限窗口近似规律、固定半径仍有角向收缩、Gaussian 径向混合可改变曲率，是彼此相容但不相同的证据。只有第一项结合逐节点形状与独立复测，才能支撑相应窗口的近似稳定系数；不能用第二项的“有收缩”替代它。

尚缺 GRU 第三档预算或独立估计器、copy 增强采样、更多架构/规则/测度与更宽 loss 窗口、末层恒等式沿整条曲线的检查、全局最低代价与总质量的独立分离，以及静态质量对实际训练的可达性校准。当前证据不支持所有测度同指数、最低范数等于总概率质量、局部 LLC 等于全局 BCE 收缩率、所有网络复杂度等价，或 SGD 必然按该静态质量采样。
