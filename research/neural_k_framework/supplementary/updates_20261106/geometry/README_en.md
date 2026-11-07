# 2026-11-06 Addendum: Volume Contraction, Radial Mixtures and Local Geometry

This addendum independently audits three experiments dated 2026-11-01 after commit `3834fb3f6555c7c58218462502635b16b9b377f9` (2026-10-27), aligning their definitions with existing E33/E37. Prepared on 2026-11-06, it distinguishes experiment dates from this audit date. The [Chinese version](README_zh.md) corresponds paragraph by paragraph, with identical formulas, numbers and sources. This audit used short CPU checks only: no training, new SMC or GPU computation.

Correction on 2026-11-07: Section 1.3 gives the direct evidence for the weighting question. The radial-reweighting command and output from 2026-11-01 were recovered from the chat history; recomputation from the same Gaussian particle radii and pool costs reproduces the results. Sections 1.1–1.2 retain the older cross-measure tables and independent uniform-ball recheck as supporting evidence; Section 4 separately reports fixed-sphere drift. The original experiments are unchanged.

## 1. Questions and Claim Boundaries

This audit addresses three related but different questions: whether contraction is approximately a power law in finite loss windows; whether relatively straight Gaussian curves partly result from radial reallocation; and whether local output geometry and readout implementation costs explain differences in global contraction slopes. All tasks constrain complete targets on small Boolean networks. They are neither competition among candidates under partial training data nor new generalization or optimizer experiments.

**Synthesis: reproducible approximate contraction laws in finite windows, and effective complexity defined through them, are not unique to Gaussian measures.** An independent-kernel uniform-ball recheck, the uniform-cube panel and several fixed-sphere controls also yield approximately linear K curves with rule-dependent coefficients. The entire pattern therefore cannot be dismissed as an artifact manufactured by Gaussian weighting. Here, "approximately stable complexity" means a relatively slowly varying contraction coefficient in a specified precision coordinate and measured window, not a constant K, nor slope stability inferred merely from decreasing mass. Straightness differs across measures, particularly for t8, whose later curve segments flatten appreciably.

This is an **empirical law conditional on measure, parameterization, rule and window**: weighting changes masses and curve positions, but can also change slopes and curvature, not just an intercept. Gaussian radial mixing explains some straightening without replacing the evidence for non-Gaussian regularity. The finest local rank cannot directly replace a global slope; the readout integral supplies a computable geometric connection without equating all network complexities with margins, whitened norms or SLT learning coefficients. Minimum norm is not the total probability mass of an implementation set.

### 1.1 Tested Measures, Windows and Shapes

Earlier public evidence is available in the original E37 report ([Chinese](https://github.com/ball-lightning6/neural-sculpting-paradigm/blob/3834fb3f6555c7c58218462502635b16b9b377f9/research/neural_k_framework/experiments/experiment_37_measure_robustness/README.md) / [English](https://github.com/ball-lightning6/neural-sculpting-paradigm/blob/3834fb3f6555c7c58218462502635b16b9b377f9/research/neural_k_framework/translations_en/post_upload/e37.md)) and its fixed-radius and natural-measure addendum ([Chinese](https://github.com/ball-lightning6/neural-sculpting-paradigm/blob/3834fb3f6555c7c58218462502635b16b9b377f9/research/neural_k_framework/experiments/experiment_37_measure_robustness/FIXED_RADIUS_20261027.md) / [English](https://github.com/ball-lightning6/neural-sculpting-paradigm/blob/3834fb3f6555c7c58218462502635b16b9b377f9/research/neural_k_framework/translations_en/post_upload/e37_update_20261027.md)). These links identify a committed version and do not depend on unpublished raw directories.

The following 20-rule comparison uses `4→16→1 tanh`, 97 standardized parameters and all 16 inputs encoded as 0/1. The uniform ball has R72; the wide Gaussian uses `sigma=72/sqrt(99)`; spherical t8 is `theta=sigma*sqrt(6/G)*Z`, with `G~chi²_8` independent of `Z~N(0,I)`; and uniform-cube coordinates are independent on `[-sqrt(3)*sigma,+sqrt(3)*sigma]`. All four match prior mean squared norm. The entire t8 vector shares one G, whereas the cube also changes the directional distribution, so these are not all purely radial reweightings. The old unit Gaussian and ball-truncated unit Gaussian are outside this scale-matched group.

The common window is `BCE 0.01→0.001`, with nodes `[0.01,0.0063,0.004,0.0025,0.0016,0.001]` and precision coordinate `a=-ln(expm1(epsilon))`. Table entries `c_a=ΔK/Δa` are whole-window secants in bit/a. "Drift" means the last adjacent-node slope divided by the first, minus 1, not a point derivative or a difference between measures. Pool probabilities are averaged before taking K. R² describes whole-curve near-linearity and must be read together with segment slopes.

| Measure | `c_a` for constant / complemented copy / parity2 / parity4 | Median absolute drift, 20 rules | Linear-fit R² range, 20 rules |
|---|---|---:|---:|
| R72 uniform ball | 0.600 / 1.962 / 5.716 / 12.488 | 13.71% | 0.998712–0.999949 |
| Wide Gaussian | 0.598 / 1.907 / 5.046 / 8.828 | 2.74% | 0.999715–0.999993 |
| Spherical t8 | 0.609 / 1.435 / 1.975 / 2.163 | 25.40% | 0.996368–0.999881 |
| Uniform cube | 0.628 / 2.668 / 6.948 / 9.938 | 12.84% | 0.999338–0.999902 |

This table provides positive evidence while quantifying its limits. Uniform-ball and cube curves admit approximate window-average descriptions but retain visible drift. For t8, parity2 adjacent slopes change from `2.248→1.753` (-22.00%) and parity4 from `2.520→1.857` (-26.30%), so they are not as stable as the wide-Gaussian curves. Relative to the uniform ball, wide Gaussian / t8 / cube have terminal-K Spearman correlations of `0.99699 / 0.98947 / 0.94887` and window-coefficient correlations of `0.99549 / 0.90526 / 0.94135`. Robust principal rankings and stable within-curve slopes are distinct claims; the 190 rule pairs are not independent replicates. The three then-new measures comprise 60 jobs with 8 pools per job and 2048 particles per pool, all reaching 0.001. Uniform-ball results select the same 20 rules from the existing 36-rule panel and are not pooled with the independent recheck below.

### 1.2 Independent Non-Gaussian Recheck and Other Controls

A more direct near-linearity recheck uses the old R72 uniform ball: constant `0x0000`, complemented copy `0x00FF`, parity2 `0x0FF0` and parity4 `0x6996`, with independent new seeds, 8 pools per rule, 4096 particles per pool and 48 updates per level. It replaces the original rotation/radial-refresh kernel with a symmetric random walk and hard rejection. All four curves reach 0.001; the maximum K discrepancy across 64 shared nodes against the main scan is 0.4468 bit. The following table uses the same 6-node window and does not substitute high R² for a drift check.

| Rule | `c_a` / bit/a | Linear-fit R² | First→last segment slope / bit/a | Relative drift |
|---|---:|---:|---:|---:|
| Constant | 0.599 | 0.999621 | 0.584→0.637 | +9.01% |
| Complemented copy | 1.980 | 0.999887 | 1.931→2.041 | +5.71% |
| parity2 | 5.794 | 0.999795 | 5.580→6.019 | +7.88% |
| parity4 | 12.505 | 0.999350 | 11.703→13.440 | +14.84% |

This supports an independently rechecked, non-Gaussian finite-window approximation, not strictly constant slopes. The source report is `research/notes/20260913_uniform_ball_euclidean_volume_zh.md`. This follow-up recalculates `tmp/4bit_uniform_ball_recheck_20260913/*/volume_curve.csv` and cross-checks `analysis/curve_summary.csv` and `analysis/local_slopes.csv` in that directory. These paths identify repository-local sources, not public attachment links. Historical R24 parity2/parity4 runs did not complete the window; R48 parity4 steepened appreciably and raised warnings. Neither is counted as positive slope-stability evidence.

| Other completed controls | Actual window and results | Supported interpretation |
|---|---|---|
| R72 ball-truncated unit Gaussian versus unbounded unit Gaussian | BCE 0.01→0.001; constant `c_a` is 16.864 versus 16.893, complemented copy 36.882 versus 36.880; parity4 239.062 versus 235.376, retaining warnings for both Gaussian deep-tail estimates | A hard boundary alone cannot explain uniform-ball versus unit-Gaussian numerical differences; this is not an additional family-wide stable-reweighting scan |
| E37 uniform spheres at fixed R20/R40 | BCE 0.03→0.001; R40 projection / majority3 / parity2 have `ΔK/log2(30)` of 2.99 / 5.67 / 14.07; only projection completes at R20, with 8 stalled pools across the other two rules | Rule-dependent angular contraction survives without radial migration; a whole-window mean alone does not establish local slope stability, and unreached nodes are not zero mass |
| This report's four conditional Gaussian radial bands, R72 ball and R72/R80/R88/R96 spheres | BCE 0.01→0.001; 3 rules and 10 conditions; the enhanced round covers only 2 rules and 6 conditions, detailed in Section 4 | Independent radial reconstruction, radius dependence and sphere shapes; not every condition is a different unconditional prior |
| Old E37 / supplementary 3-bit uniform-ball MSE target shifts | R72, MSE=0.01; increasing target separation 18→20 gives K increments of 0.25 / 1.77 / 2.60 / 5.51 bit for constant / copy / majority / parity; R72/R96 additionally check endpoint interactions at MSE=0.001 | Non-Gaussian rule-dependent target responses and finite-window approximate additivity; shifted targets are not nested events and do not supply BCE slope-stability evidence |

The same `research/neural_k_framework/supplementary/measure_robustness/RESULTS.md` also records MSE controls for joint target/reference-center translation, target-only translation, removal of readout bias and target half-separations `0.01/0.1/0.5/1/3`. All 40 curves reach `1e-5`; the last 4 nodes are `0.001/0.0003/0.0001/1e-5`, with reported deep-tail slopes around `12.2–13.5 bit/decade`. These intervene on output events, coordinates or architecture, not additional radial priors; their MSE precision slopes cannot be combined with the bit/a values above.

### 1.3 Radial Reweighting of the Same Gaussian Particles (2026-11-01)

**Changing the radial density shape changes the contraction slope, yet the first-to-last segment drift for parity4 remains small under several non-Gaussian weights.** This diagnostic changes the target density using retained particles from the direct conditional Gaussian runs; it does not run SMC afresh for each target prior. Complemented copy uses `results_v1/copy__gaussian`; parity2 and parity4 use `results_v2_enhanced/parity2__gaussian` and `results_v2_enhanced/parity4__gaussian`. Each node has 4 pools, with 768 particles per pool in the first round and 1536 in the enhanced round.

The primary comparison changes the density with respect to parameter-space volume, not the one-dimensional density of radius. Set `d=97` and `sigma0=72/sqrt(99)`; varying p holds the mean squared radius fixed, and `p=2` gives the original Gaussian. The angular distribution is unchanged, and the radial Jacobian cancels in the density ratio:

$$
\rho_p(\theta)=C_p e^{-(r/a_p)^p},\quad r=\|\theta\|,\quad
a_p^2=d\sigma_0^2\frac{\Gamma(d/p)}{\Gamma((d+2)/p)},\quad
C_p=\frac{p}{S_{d-1}a_p^d\Gamma(d/p)},\quad S_{d-1}=\frac{2\pi^{d/2}}{\Gamma(d/2)}.
$$

At BCE nodes `[0.01,0.0063,0.004,0.0025,0.0016,0.001]`, the target-to-original Gaussian density ratio w corrects each pool's cost before pool probabilities are averaged. Rule columns report 6-point least-squares slopes in `a=-ln(expm1(epsilon))` (bit/a), not the Section 1.1 secants. Drift is `100*(last segment slope / first segment slope - 1)`, in %. The weight ESS fraction reported is the minimum across nodes and pools, with N particles in each pool:

$$
K'_b=K_{0,b}-\log_2\!\left(\frac1N\sum_i w_{bi}\right),\qquad
K'=-\log_2\!\left(\frac1B\sum_b2^{-K'_b}\right),\qquad
q_b=\frac{(\sum_i w_{bi})^2}{N\sum_i w_{bi}^2}.
$$

| `p` | Complemented copy | parity2 | parity4 | parity4 drift | parity4 minimum weight ESS fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.8816 | 4.6017 | 7.5530 | -8.93% | 0.00824 |
| 1.5 | 1.9138 | 4.8931 | 8.2793 | -2.88% | 0.21564 |
| 2 | 1.9348 | 5.0735 | 8.8511 | -2.25% | 1.00000 |
| 2.5 | 1.9479 | 5.1934 | 9.2837 | -2.37% | 0.67828 |
| 3 | 1.9561 | 5.2790 | 9.6239 | -1.98% | 0.35339 |

Across `p=1.5/2/2.5/3`, slopes rise with p for all three rules, while the last parity4 segment is only about 2%–3% shallower than the first. **`p=1` is a stress test, not reliable positive evidence**: the minimum parity4 weight ESS fraction is only 0.00824. The old E37 formal scan replaced the planned Laplace prior with spherical t8; reweighting here is not an independent Laplace scan.

The other control changes only Gaussian **variance**. Write `h=v/v0` and `v0=sigma0²`, using the same nodes and probability pooling. The `0.9/1/1.1` values are not standard-deviation multipliers; the separate, wider standard-deviation scan is not included:

$$
\log w=-\frac d2\log h+(1-1/h)\frac{r^2}{2v_0}.
$$

| `v/v0` | Complemented copy | parity2 | parity4 | Minimum weight ESS fraction: complemented copy / parity2 / parity4 |
|---:|---:|---:|---:|---:|
| 0.9 | 2.0623 | 5.4174 | 9.4643 | 0.58401 / 0.54808 / 0.53604 |
| 1 | 1.9348 | 5.0735 | 8.8511 | 1.00000 / 1.00000 / 1.00000 |
| 1.1 | 1.8162 | 4.7837 | 8.3510 | 0.56448 / 0.54852 / 0.54935 |

ESS in both tables diagnoses concentration of importance weights, not MCMC ESS. Reweighting preserves the original SMC genealogy, and high ESS cannot establish coverage of unseen tails. In Section 4 the layer contributions `w_j=p_j*A_j/V` change with loss by conditioning a fixed prior, and `K_frozen` is a mean cost with layer contributions frozen. Neither operation changes the target density as done here.

The original command and output were recovered from chat history; no separate diagnostic output file was saved at the time. The [recheck record](radial_reweighting.json) was generated on 2026-11-07 and contains only configuration, pooled node costs, curve summaries and source SHA256 hashes, not particles. The short CPU recomputation matches the recovered 21 slopes and 4 drift percentages with maximum absolute discrepancies of 0; the maximum discrepancy for the 5 parity4 minimum weight ESS fractions is 4.44e-16. These are arithmetic discrepancies, not statistical errors.

Run the [standalone script](radial_reweighting.py) from this directory. The first command is the default read-only standard-library check: it recomputes slopes and drift from published node costs and checks the configuration and both reports' tables. It cannot reconstruct weight ESS from K or verify actual source-file hashes. The second requires the original files and NumPy; it recomputes K, weight ESS and source hashes read-only and compares them with the release. Only an explicit additional `--write` generates the JSON.

```bash
python -I -S -B radial_reweighting.py --public-only
python -B radial_reweighting.py --from-sources
```

## 2. Objects and Noninterchangeable Protocols

For the complete input table, write label signs, logits, mean BCE, reference-measure mass and bit cost as:

$$
s_i=2y_i-1,\quad m_i=s_i z_\theta(x_i),\quad
L_f(\theta)=\frac1{16}\sum_{i=1}^{16}\log(1+e^{-m_i}),\quad
V_{\mu,f}(\epsilon)=\Pr_\mu[L_f\le\epsilon],\quad
K_{\mu,f}(\epsilon)=-\log_2 V_{\mu,f}(\epsilon).
$$

When `epsilon < ln(2)/16 = 0.0433217`, any wrong sign or zero margin contradicts the mean-BCE bound, so every hard label must be correct. Above this threshold, complete-target events remain BCE events and cannot automatically be treated as disjoint hard-function cells. Mean BCE, pointwise margin lower bounds and equal-amplitude logit densities are different objects.

| Experiment | Actual inputs and output | Parameter measure or local target | Main analysis window |
|---|---|---|---|
| Cross-architecture | All 16 4-bit inputs; MLP/CNN encoded as ±1, GRU/Transformer use bit tokens; one logit | Independent standard whitened Gaussian over all variable parameters; fan-in scaling | BCE 0.03 to 0.0001, 11 nodes |
| Radial mixture | All 16 inputs encoded as 0/1; one logit | Wide Gaussian, conditional Gaussian radial bands, uniform solid ball or uniform sphere in 97 standardized coordinates | BCE 0.01 to 0.001, 6 nodes |
| SLT local probe | Same inputs as the selected cross-architecture MLPs; auxiliary continuous-logit preservation error | Localized Gibbs target around anchors, not the original global prior | Controlled BCE anchors 0.03, 0.003, 0.0001 |
| Readout integral | Complete terminal particle sets from the original cross-architecture global SMC | Original standard whitened Gaussian conditioned on the BCE event | Point derivative 0.0001; secant 0.0002 to 0.0001 |

The cross-architecture main table averages pool K values; the radial main table averages probabilities before taking the negative logarithm. These summaries are unequal and their difference is not itself a scientific effect. Three precision coordinates must also be distinguished:

$$
\overline K=\frac1M\sum_{r=1}^M K_r,\qquad
K_{\rm pooled}=-\log_2\!\left(\frac1M\sum_{r=1}^M2^{-K_r}\right),\qquad
t=\ln(1/\epsilon),\quad u=\log_2(1/\epsilon),\quad
a(\epsilon)=-\ln(\operatorname{expm1}(\epsilon)).
$$

If `V ≈ C epsilon^kappa`, then `dK/dt = kappa/ln(2)`, whereas `dK/du = kappa`. Radial tables use `dK/da`; at finite epsilon, `a` and `t` are not exactly the same coordinate. Fitted intercepts are not estimates of hard-function prior mass.

## 3. Cross-Architecture Finite-Window Contraction

### 3.1 Actual Models and Sampling Protocol

Source definitions, parameter shapes in result identities and parameter counts agree. Counts below are dimensions included in the measure and sampling, including variable biases but excluding fixed positional encodings or fixed normalization affine constants.

| Configuration | Actual width, depth and readout | Sampled parameters |
|---|---|---:|
| tanh16 | 4→16→1; 1 tanh hidden layer | 97 |
| tanh32x3 | 4→32→32→32→1; 3 tanh hidden layers | 2305 |
| relu32x3 | 4→32→32→32→1; 3 ReLU hidden layers | 2305 |
| cnn8x2 | 1D convolutions with channels 1→8→8, kernel 3, padding 1, ReLU; length 4 flattened to 32 before linear readout | 265 |
| gru16 | 2×4 token embedding; 1 hidden-16 GRU layer unrolled over 4 steps; final-hidden linear readout | 1081 |
| transformer8 | 3×8 embedding including CLS; 4 input tokens plus CLS; 1 width-8, 2-head, FF-16 ReLU encoder block with CLS linear readout | 601 |

Both Transformer LayerNorm operations fix gain=1 and bias=0, with epsilon=1e-5; the corresponding 32 fixed affine values are excluded from the 601 dimensions. This is not the same protocol as the existing E33 conditional derivation with jointly scaled random LayerNorm affine parameters and a Gaussian readout. MLP/CNN weights and biases are both divided by the square root of the associated weight fan-in; embedding standard deviation is 1. GRU unrolling steps are not stacked depth.

Rules are first-bit copy and XOR of the first two bits (parity2), each constraining all 16 points, without amplitude regularization, equal-amplitude constraints or SGD. Prespecified BCE nodes are `[0.3,0.1,0.03,0.02,0.01,0.006,0.003,0.002,0.001,0.0006,0.0003,0.0002,0.0001]`. An independent 384-particle pilot uses survival quantile 0.6 to choose intermediate thresholds and tune block-pCN step sizes. Production pools freeze this schedule and accumulate actual survival fractions, not a fixed per-level survival rate substituted for measured mass.

The main budget uses 3 pools per condition, 768 particles per pool and 12 block sweeps. Enhanced budgets check only parity2 in each configuration, using 1536 particles and 24 sweeps, independent production random streams but the same condition-specific pilot. Original records specify Python 3.13.5, PyTorch 2.11.0+cu130, CUDA 13.0, RTX 3080, cuda:1, float64 and disabled TF32. Reported sampling time is not the time of this audit.

### 3.2 Main Results

Main-budget terminal K and window slopes follow. The early window is `0.03→0.006`, the late window `0.0006→0.0001`; slopes are in bit/nat. Enhanced curves are not pooled into this table.

| Configuration | Copy terminal K | Parity2 terminal K | Copy early→late slope | Parity2 early→late slope |
|---|---:|---:|---:|---:|
| tanh16 | 190.00 | 709.63 | 21.24→24.98 | 75.88→85.32 |
| tanh32x3 | 226.15 | 452.54 | 24.81→26.82 | 43.74→39.50 |
| relu32x3 | 172.90 | 281.48 | 17.33→13.17 | 26.14→19.38 |
| cnn8x2 | 152.08 | 274.10 | 15.71→12.89 | 27.56→21.66 |
| gru16 | 307.24 | 441.80 | 28.61→24.81 | 39.53→38.69 |
| transformer8 | 102.57 | 135.40 | 8.70→13.50 | 10.28→14.27 |

Across 12 main curves and 11 low-loss nodes, linear-fit R² ranges from 0.99246 to 0.99995; RMSE is about 0.22%–2.76% of the K span. Yet fitting only the shallower 8 nodes gives prediction RMSE of 0.30–10.95 bit on the deepest 3 nodes; 1 bit already means a factor of 2 in mass. Thus a nearly straight plot is not an accurate volume prediction or evidence of a constant slope.

The 2305-parameter tanh32x3 and relu32x3 differ only in activation but show different curvature directions. The parity2 K disadvantage relative to copy grows as the threshold tightens in all six configurations, but two rules cannot establish a complexity theorem for all functions. Free p in `K=a+b*t^p` is descriptive only: main-budget copy/parity2 values are tanh16 1.27/1.14, tanh32x3 1.08/0.81, ReLU 0.59/0.55, CNN 0.69/0.61, GRU 0.69/0.90 and Transformer 1.64/1.43. These are not identified ultimate asymptotic exponents.

### 3.3 Completion, Budget Sensitivity and Negative Results

Final raw records show 18/18 curves and 54/54 pools reaching 0.0001, with 0 final failed curves. Historical `results_v1/progress.json` retains 3 file-access failures, excluded as additional repetitions. `results_v2/progress.json` also retains 3 enhanced-entry identity tuple/list comparison errors. These occurred before enhanced production sampling; final status must combine the 6 completed entries in `additional_budget_checks.json` with actual result files, not rely only on the older progress file.

| Parity2 configuration | Main terminal K | Enhanced terminal K | Enhanced minus main K | Change in window ΔK |
|---|---:|---:|---:|---:|
| tanh16 | 709.63 | 708.56 | -1.07 | -0.70 |
| tanh32x3 | 452.54 | 451.54 | -0.99 | -0.38 |
| relu32x3 | 281.48 | 280.19 | -1.29 | -0.02 |
| cnn8x2 | 274.10 | 274.78 | +0.68 | +0.69 |
| gru16 | 441.80 | 429.21 | -12.59 | -10.48 |
| transformer8 | 135.40 | 134.62 | -0.77 | -0.71 |

Differences are calculated before rounding. Except for GRU, changes in low-loss contraction increments are at most about 0.71 bit, supporting budget stability of these parity2 curve shapes, not convergence of all absolute masses. The GRU terminal difference of 12.59 bit corresponds to a mass factor of roughly 6000. Without a third budget or independent global estimator, its precision is unsettled; this alone also does not establish a counterexample to a volume law. Copy has no enhanced budget. Pool SD, acceptance rates, smooth curves and ancestry counts cannot exclude shared mode omission; ancestor counts are not ESS.

## 4. Radius-Conditional Volume and Gaussian Radial Mixtures

### 4.1 Model, Measures and Actual Scope

The radial experiment uses `4→16→1 tanh`, 97 parameters and 0/1 inputs. Its first 80 standardized coordinates are divided by 2, and its final 17 by 4. Rule codes are complemented copy `0x00FF`, parity2 `0x0FF0` and parity4 `0x6996`; the first rule must not be relabeled as the cross-architecture copy target. Radius is defined in these standardized theta coordinates, not as the norm of actual scaled weights or the norm of internal unit-Gaussian coordinates u.

$$
\theta=\sigma u,\quad u\sim N(0,I_{97}),\quad
\sigma=72/\sqrt{99}\approx7.23627227,\qquad
\mathbb E\|\theta\|^2=97\sigma^2\approx5079.272727.
$$

This matches the mean squared radius of the R72 uniform solid ball, not the R72 sphere. The ten measure conditions are direct Gaussian; four conditional Gaussian radial bands `[0,72)`, `[72,80)`, `[80,88)`, `[88,+inf)`; the R72 uniform solid ball; and uniform spheres R72/R80/R88/R96. Prior band masses are 0.575510896, 0.381866476, 0.041947433 and 0.000675196. Conditional radial bands still mix radii internally; only sphere controls fix radius completely.

The first round uses 3 rules×10 measure conditions, each with 4 pools, 768 particles, 16 updates at ordinary levels and 48 at designated nodes; an independent 256-particle pilot freezes thresholds and proposal scales. Nodes are `[0.7,0.3,0.1,0.03,0.016,0.01,0.0063,0.004,0.0025,0.0016,0.001]`. The enhanced round covers only direct Gaussian, the four bands and the R72 sphere for parity2/parity4, using independent new seeds, 1536 particles and 32/96 updates. Both rounds use float64, cuda:1, RTX 3080, disabled TF32 and deterministic operations, without gradient warmup.

The first round completed 30/30 conditions, 120/120 pools and 330 condition-nodes; the enhanced round completed 12/12 conditions, 48/48 pools and 132 condition-nodes. Both have 0 final failed conditions. Rounds are not pooled as equal-budget repetitions, and the 22 enhanced rule-nodes are not independent tasks. This mechanism test was designed after inspecting the older panel; it is not claimed as blind prediction or formal preregistration.

### 4.2 Reconstruction and Curvature Decomposition

For a spherical Gaussian, the full radial density must contain the Jacobian. Fixed-radius angular probability and total Gaussian mass are related as follows. Four-band reconstruction uses independent samples from each conditional prior, rather than partitioning the terminal direct-Gaussian samples.

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

Where differentiability and exchanging derivatives with integrals are justified, between-band reallocation can offset upward within-band curvature, but does not guarantee a straight total curve. Here `A_j` is a conditional Gaussian band probability, not the probability `A_f(r,epsilon)` on one sphere. The four bands cover all radii; sparse sphere interpolation does not substitute for the continuous integral.

The average-cost diagnostic with weights frozen at `epsilon=0.01` satisfies an exact finite-partition identity. It is not the probability cost of another actual mixture measure.

$$
K_{\rm frozen}(a)=K(a_0)+\sum_jw_j(a_0)[K_j(a)-K_j(a_0)],\qquad
K_{\rm frozen}(a)-K(a)=D_{\rm KL,bits}(w(a_0)\|w(a)).
$$

### 4.3 Main Results and Fixed-Sphere Controls

Across all nodes, the maximum K point-estimate discrepancy between stratified reconstruction and independent direct Gaussian is 0.414 bit in the first round and 0.085754 bit in the enhanced round. Curvature units below are bit/a², and the last column is in bits. Mixture and frozen columns use whole-window quadratic fits; the central columns use midpoint derivatives of per-band quadratic fits, so exact columnwise subtraction is not required.

| Budget/rule | Mixture curvature | Frozen curvature | Midpoint mean within-band curvature | Midpoint between-band variance term | Terminal frozen-cost gap |
|---|---:|---:|---:|---:|---:|
| Initial/complemented copy | 0.024 | 0.036 | 0.036 | 0.011 | 0.029 |
| Initial/parity2 | 0.045 | 0.138 | 0.137 | 0.090 | 0.257 |
| Initial/parity4 | -0.098 | 0.244 | 0.206 | 0.296 | 0.837 |
| Enhanced/parity2 | 0.091 | 0.213 | 0.207 | 0.114 | 0.316 |
| Enhanced/parity4 | -0.114 | 0.278 | 0.230 | 0.353 | 1.014 |

For enhanced parity4, the conditional contribution from R≥88 rises from 15.54% to 58.69%, although this band's prior mass is only 0.06752%. This is reallocation under conditioning, not an altered sampling prior. The 95% pool-bootstrap interval for enhanced parity2 mixture curvature is [0.042,0.139], and for parity4 [-0.224,-0.015], precluding a claim of a strictly constant exponent. Negative curvature means a decreasing but still positive contraction rate, not increasing mass.

Initial-round linear window slopes for the fixed spheres follow, in bit/a. Rule-dependent contraction persists at fixed radius, and different radii do not share one coefficient.

| Rule | R72 | R80 | R88 | R96 |
|---|---:|---:|---:|---:|
| Complemented copy | 1.930 | 1.709 | 1.536 | 1.343 |
| parity2 | 5.628 | 4.751 | 4.148 | 3.730 |
| parity4 | 12.503 | 10.309 | 8.694 | 7.762 |

Within the same 6-node window, initial-round parity4 first-to-last adjacent-slope drift at R72/R80/R88/R96 is `+19.92% / +8.84% / +4.97% / +5.09%`; enhanced R72 still has `+16.87%`. Thus some non-Gaussian sphere slopes vary relatively little, but not all spheres are equally stable; high linear-fit R² at R72 does not remove residual curvature. These numbers are short CPU recalculations from the poolwise K values in [radial_curves.csv](radial_curves.csv), pooled in probability space, not new samples.

### 4.4 Uncertainty and Negative Results

Similar total masses do not imply identical parameter distributions. Enhanced-round mean/maximum TV between direct-Gaussian and reconstructed four-band distributions is 0.0269/0.0423 for parity2 and 0.0351/0.0598 for parity4. The parity4 95% bootstrap interval for terminal reconstruction K discrepancy remains [-0.606,0.437] bit, and that for window-increment discrepancy is [-0.268,0.291] bit. Intervals resample whole curves from 4 pools per condition, conditional on the frozen pilot, and omit shared mixing bias. This audit checks saved intervals; it does not rerun bootstrap or read particles to recompute TV.

Retrospective analysis of the old 2026-09-13 panel includes 96 curves. Among 20 scale-matched rules, 20/20 have smaller Gaussian first-to-last relative slope drift than uniform-ball drift; median absolute drift is 2.74% versus 13.71%. This is a specified panel and metric, not a universality test on independently sampled tasks. Coarse stratification of old samples is a mechanism clue, not additional replication of the new independent stratified reconstruction.

The original numerical audits recomputed forward passes for 682368 parameter vectors across 462 condition-nodes in the two rounds. This audit checks their saved summaries and provenance, without claiming to recheck those particles. Some initial parity4 sphere pools retain only 1–4 initial ancestors; the enhanced minimum is 8. Neither yields ESS. An independent chi-CDF check triggered the original machine-precision tolerance: the actual maximum absolute discrepancy was 2.62e-10, subsequently retained under 1e-8 absolute/1e-7 relative tolerances. This was not a sampling failure, and no sampled values were changed.

## 5. SLT Local Probes: Positive Evidence and Failure Boundaries

### 5.1 Anchors and Local Target

The four conditions use tanh16 (97 parameters) and relu32x3 (2305 parameters), each with copy/parity2. Selecting 2 terminal particles from each of the original 3 main-budget pools per condition gives 24 prototypes. Hidden parameters are fixed while readout weights and bias are scaled together to BCE 0.03, 0.003 and 0.0001, producing 72 controlled anchors. These are not newly sampled global conditional distributions at the three thresholds, and rescaling is not training. Once all training margins are positive, enlarging only the readout can further decrease BCE, so lower BCE need not require new features; existence of this path also does not determine its reference-measure mass.

Local sampling varies all parameter directions, using continuous-logit preservation error instead of treating nonstationary BCE as a local minimum. A fixed linear preconditioning transform preserves the target; its localization Gaussian differs from the original global Gaussian.

$$
R(\theta;\theta_0)=\frac1{32}\sum_{i=1}^{16}[z_\theta(x_i)-z_{\theta_0}(x_i)]^2,\qquad
q_{\beta,\gamma}(\theta\mid\theta_0)\propto
\exp[-\beta R-\gamma\|\theta-\theta_0\|^2/2].
$$

Measurements are finite-scale cost `beta E[R]`, the Gram spectrum of the logit Jacobian and the BCE generalized Gauss-Newton spectrum. For eigenvalues lambda of the corresponding Gram matrix, spectral participation dimension is:

$$
d_{\rm eff}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2},\qquad
\beta\mathbb E[R]_{\rm linear}=\frac12\sum_i\frac{\beta\lambda_i}{\gamma+\beta\lambda_i}.
$$

### 5.2 Local Results

All 72 anchors have rank 16 at relative singular-value thresholds 1e-6, 1e-8, 1e-10 and 1e-12. The fine-scale benchmark of the local quadratic output-preservation model is r/2=8. This is numerical rank, not an exact-rank proof. The table uses means over 6 anchors at BCE=0.0001; costs are pCN-MH means at gamma=1e7.

| Configuration/rule | Rank | Logit spectral participation | BCE GGN spectral participation | Cost beta/gamma=100 | Cost beta/gamma=10000 |
|---|---:|---:|---:|---:|---:|
| tanh16/copy | 16 | 5.36 | 4.78 | 5.76 | 7.93 |
| tanh16/parity2 | 16 | 7.08 | 6.71 | 6.42 | 7.99 |
| relu32x3/copy | 16 | 4.18 | 3.71 | 7.57 | 8.02 |
| relu32x3/parity2 | 16 | 5.70 | 4.83 | 7.83 | 7.92 |

Finite-resolution spectra and lower-probe-strength means place parity2 above copy within both architectures, but stronger probes approach 8; the small ReLU reversal is not a stable complexity reversal. Equal local rank cannot explain widely different global BCE slopes, and continuous-output preservation cost cannot directly be called hard-rule complexity or a new Neural K.

### 5.3 Initial Failure and Strong-Localization Check

The first round used gamma=1000,10000, beta/gamma=100,10000 and SGLD steps 0.04/0.02. Both tanh conditions completed; ReLU/copy produced nonfinite energy at step 5, its completed pCN run also contained zero-acceptance chains, and ReLU/parity2 was not started. The record retains 2 completed, 1 failed and 1 unstarted condition; the round is not reliable equilibrium replication. The failure is documented in `results_v1/FAILURE_NOTE.md`; the failed condition lacks a final `status.json` and must not disappear from failure counts for that reason.

The formal check increased gamma to 1e6,1e7, retaining beta/gamma=100,10000. pCN-MH uses rho=0.6, burn=250 and draws=1000. The fixed-SHA256 vendored devinterp 1.3.2 SGLD kernel uses step 0.02, burn=600 and draws=2400; half-step 0.01 uses 1200/4800, with thin=1. All four conditions completed with 0 final failures. There are 72×4×3×3=2592 local chain runs, not 2592 independent datasets.

Minimum pCN chain acceptance is 0.293. Maximum group-mean relative discrepancies of the two SGLD budgets from pCN are 4.87%/5.80%; the maximum pCN group-mean difference from the local linear analytic benchmark is 0.1316. Within-chain drift remains, so differences of a few hundredths are not stable rankings. Original official-step, quadratic/quartic-target calibration and preconditioning checks passed; this does not establish sufficient mixing for arbitrary nonlinear local targets.

## 6. Readout Geometry: Conditional Models and Full-Distribution Integration

### 6.1 Fixed Hidden Representations Are Not All Networks

Given hidden parameters eta, absorb readout scaling and bias into feature matrix B_eta. An independent standard whitened readout w then satisfies the formulas below. Minimum unit-common-margin readout cost and minimum equal-amplitude-output norm impose different constraints. The latter equality requires invertible Sigma_eta; otherwise reachability must be checked, rather than adding arbitrary jitter and claiming the original density exists.

$$
z=B_\eta w,\quad w\sim N(0,I_k),\quad \Sigma_\eta=B_\eta B_\eta^T,\qquad
C_\eta(s)=\frac12\inf_{\operatorname{diag}(s)B_\eta w\ge\mathbf1}\|w\|^2,\qquad
\inf_{B_\eta w=as}\|w\|^2=a^2s^T\Sigma_\eta^{-1}s.
$$

These describe conditional geometry for a fixed representation in specified whitened coordinates, not a theorem equating all network complexities. For a fixed feasible representation, the leading large-amplitude Gaussian tail cost can connect to minimum norm. Finite-amplitude total mass still includes feasible directions, neighborhood volume, prior scarcity of hidden representations and integration over implementations. Changing representation, parameterization, reference measure or output event changes the answer. Full-network event mass still requires integration:

$$
V_f(\epsilon)=\int \Pr_w[L_f(\eta,w)\le\epsilon\mid\eta]\,\mu_\eta(d\eta).
$$

This work did not newly optimize global minimum norm, verify the infinite tail or substitute the cost of one best implementation for this integral. The existing E33 conditional amplification-order derivation remains a conditional explanation, not a universal cross-architecture proof.

### 6.2 Readout Identity Under the Original Gaussian Conditional Distribution

Unlike the fixed-representation illustration, this diagnostic averages over all original SMC terminal particles, including the distribution of hidden representations. Let k include the readout bias, L be mean BCE and m be positive margins. With a linear readout, standard whitened Gaussian measure, epsilon<ln(2)/16, regular boundaries and vanishing flux at infinity, the Gaussian-weighted divergence formula gives:

$$
g=w\cdot\nabla_wL=-\operatorname{mean}_i[m_i\sigma(-m_i)],\qquad
h=w\cdot\nabla_wg=g+\operatorname{mean}_i[m_i^2\sigma(-m_i)\sigma(m_i)],
$$

$$
\kappa(\epsilon)=\frac{\epsilon V'(\epsilon)}{V(\epsilon)}
=\mathbb E\!\left[1+\frac{L(k-\|w\|^2)}g-\frac{Lh}{g^2}\ \middle|\ L\le\epsilon\right].
$$

Here sigma is the logistic function, not the earlier Gaussian scale. The right-hand side does not use accumulated survival fractions or a fitted slope, but it still uses SMC conditional particles and is therefore not an independent sampling validation. Integrating conditionally on each hidden representation avoids confusing ReLU hidden-layer nondifferentiabilities with readout derivatives. It also does not prove constant kappa across thresholds.

### 6.3 Terminal Results and Supported Interpretation

The point derivative is evaluated at epsilon=0.0001; the secant uses 0.0002→0.0001. Both columns are in bit per log2 precision unit, namely kappa, not bit/nat. A point derivative need not equal a window average. Each condition equally averages 3 pools; enhanced rows are not pooled with the main budget.

| Configuration/rule | Budget | Readout dimension k | Integral point slope | SMC secant slope |
|---|---:|---:|---:|---:|
| tanh16/copy | 1 | 17 | 17.964 | 17.611 |
| tanh16/parity2 | 1 | 17 | 61.341 | 60.281 |
| tanh16/parity2 | 2 | 17 | 60.708 | 59.717 |
| relu32x3/copy | 1 | 33 | 8.689 | 8.757 |
| relu32x3/parity2 | 1 | 33 | 12.818 | 12.959 |
| relu32x3/parity2 | 2 | 33 | 12.876 | 13.057 |

Absolute relative point/secant discrepancies for the four main-budget conditions range from 0.78% to 2.00%. Higher tanh point derivatives and lower ReLU point derivatives are compatible with their respective curvature directions. Main-budget tanh/copy and tanh/parity2 norm contributions are 19.714 and 63.086, dimension contributions -1.857 and -1.853, and derivative corrections 0.107 and 0.108. The difference primarily lies in margin-weighted readout whitened-norm cost, not local rank. This connects conditional implementation costs; it is not a universal equality with minimum norm or independent logical complexity.

Original identity calibration includes 5 one-dimensional analytic-tail cases and 3 two-dimensional mean-BCE curved-domain integrals, with maximum relative discrepancy 6.00e-7. This audit checks poolwise component sums and original SMC secants, without reopening checkpoints to recompute margins, norms or integrals. Checkpoint SHA256 values in the machine summary are explicitly marked as recorded in the original report and not rehashed. Particle standard deviation is not standard error, and resampling ancestry does not increase the number of independent tasks.

## 7. Independent Audit and Reproducible Provenance

[recompute.py](recompute.py) provides public-package arithmetic checks and a full source audit. Both modes use only the Python standard library, importing neither Torch, NumPy, training entry points nor historical experiment modules. The full audit reads raw level/pool JSON, spectra and chain summaries to independently reconstruct survival ledgers, linear/free-p fits, radial probability pooling, the KL identity, quadratic curvatures, spectral participation dimensions and readout components, and checks prior band masses by a half-integer chi-CDF recurrence. Saved audit records cover SHA256 for 244 source files, frozen-code hashes, pilot/production schedules and configuration signatures. This mode requires original results, frozen code and contextual material outside the package; the table below reports that full audit, not renewed original-source verification by `--public-only`.

| Recomputed quantity | Maximum absolute discrepancy |
|---|---:|
| Cross-architecture cumulative level K / bit | 1.93e-12 |
| Cross-architecture holdout prediction RMSE / bit | 1.71e-13 |
| Radial cumulative level K / bit | 1.28e-13 |
| Half-integer chi-CDF band masses | 2.62e-10 |
| Radial finite-partition KL identity / bit | 1.37e-14 |
| Radial mixture curvature / bit/a² | 6.05e-14 |
| Spectral participation dimension | 3.55e-15 |
| Local-cost group mean | 1.78e-15 |
| Sum of readout components | 1.42e-14 |

These are implementation and ledger consistency errors, not statistical errors or confidence intervals for volume estimates. Rerunning the original forward audits, bootstrap intervals and radial TV computations is outside this CPU audit. The 2026-11-06 audit rechecked only the paired table and summary for the older 96-curve panel, without rereading its full historical particles. No checkpoints, particle arrays or large raw files were copied.

The 2026-11-07 follow-up additionally uses the standard library to recompute first/last slopes, linear fits and residuals for 96 curves from `research/radial_mixture_diagnostic_20261101/historical_analysis/all_curves.csv`. The maximum absolute discrepancy against corresponding `shape_statistics.csv` values is `1.78e-14`; selecting the 20 rules shared by all four measures gives Section 1.1's drift and R² results. For Section 1.2's four independent uniform-ball rechecks, window secants, R² and adjacent-node slopes reproduce the saved tables with maximum discrepancy 0. These two older table checks neither reread particles nor reran bootstrap calculations; the older panel is documented in `research/notes/20260913_natural_measure_symmetry_rank_zh.md` and `research/notes/20260913_uniform_ball_36_rule_rank_comparison_zh.md`. These follow-up checks are outside the scope of the original `audit_summary.json`, `source_manifest.json` and `recompute.py`; Section 1.3's radial reweighting has a separate script and source hashes.

| Public file | Contents |
|---|---|
| [audit_summary.json](audit_summary.json) | Parameter dimensions, fits for 18 curves, budget differences, radial decomposition, local/readout summaries and audit errors |
| [cross_architecture_curves.csv](cross_architecture_curves.csv) | 234 curve-nodes with K for 3 pools, without combining budgets |
| [radial_curves.csv](radial_curves.csv) | 462 condition-nodes with K for 4 pools, retaining version and measure identity |
| [radial_reweighting.json](radial_reweighting.json) | Both Section 1.3 families: 24 curves, 144 pooled node costs, shapes, minimum weight ESS summaries and 24 source-file SHA256 hashes |
| [radial_reweighting.py](radial_reweighting.py) | Section 1.3 standard-library public check and NumPy source recomputation, without sampling |
| [source_manifest.json](source_manifest.json) | Repository-relative source paths, byte sizes and full SHA256; raw results, configurations, statuses, code and contextual notes |
| [recompute.py](recompute.py) | Public-package arithmetic, full source audit and bilingual structure checks, without sampling |
| [test_public_only.py](test_public_only.py) | Tests on copies without original sources, checking missing/changed data, duplicate nodes and write rejection |

Original sources below are path identifiers, not package download links; full SHA256 values are in [source_manifest.json](source_manifest.json). The manifest records source paths, byte sizes and hashes at audit time. It does not substitute for raw data or establish public availability or independent experimental replication. Contextual notes are represented only by provenance metadata; their full text is not distributed in this package.

| Topic | Source paths and roles |
|---|---|
| Cross-architecture raw data | `research/cross_architecture_contraction_20261101/results_v2/*/main_x*/result.json`; per-level survival, per-pool K, identities and statuses |
| Cross-architecture frozen code | `research/cross_architecture_contraction_20261101/source_snapshot_v2/research/`; includes the actual model dependency, without replacing historical hashes with a revised entry point |
| Radial raw data | result, identity, pilot and manifest files under `research/radial_mixture_diagnostic_20261101/results_v1/` and `results_v2_enhanced/` |
| Radial code and audits | `research/radial_mixture_diagnostic_20261101/results_v1/source/`, `analyze.py` and both `AUDIT.json` files |
| Local raw data | `research/slt_local_probe_20261101/results_v2/*/geometry.json`, `pcn_mh.json`, `sgld.json`, `sgld_halfstep.json` and status files |
| Local failure and code | `research/slt_local_probe_20261101/results_v1/FAILURE_NOTE.md`, `experiment_v1_snapshot.py`, `experiment_v2_snapshot.py` and vendored SGLD |
| Readout identity | `research/slt_local_probe_20261101/readout_identity.py`, `readout_identity_results.json`, `READOUT_IDENTITY_ZH.md` |
| Existing explanations and boundaries | E33 `OUTPUT_CONTRACTION_DERIVATION.md`; E37 `FIXED_RADIUS_20261027.md`; `research/notes/20261105_amplitude_alignment_mechanism_revision_zh.md` and `20261105_core_volume_generalization_literature_review_zh.md` |

After downloading only this package, run the following read-only check from its extracted directory. `--public-only` neither locates nor reads original `research/` directories. Adding `--documents` checks packaged bilingual heading levels, paragraph structure, formulas, numbers and links.

```bash
python -I -S -B recompute.py --public-only --documents
```

Public mode uses poolwise K at 234 cross-architecture curve-nodes and 462 radial condition-nodes to recompute endpoints, window slopes, fits and budget differences, probability pooling, radial reconstruction, curvature and the KL identity. It also checks 6 readout-summary rows using CSV secants and saved JSON components. It validates the format of `source_manifest.json`, without opening the listed sources to verify their hashes. Original SLT spectra, local-chain statistics, terminal statuses and failure logs, the historical panel, bootstrap/TV and particle forward passes are not revalidated in public mode; their saved summaries and limitations remain reported. Agreement between derived files does not prove original-data authenticity, independent experimental replication or global mixing.

The full source audit requires this package in its original repository hierarchy and all original materials listed in the manifest. In that setting, run the following command from this directory. `--write` regenerates the four machine-readable package files only in full mode; combining it with `--public-only` is rejected. When original sources are absent, use public mode rather than treating this package as a complete reproduction bundle for the original SMC runs.

```bash
python -I -S -B recompute.py --documents
```

Downloading this package alone cannot reconstruct all per-level survival ledgers or reproduce the original SMC runs. Full experimental reproduction additionally requires the original protocols, frozen runtime code, dependencies and suitable sampling budgets; this package does not launch those computations. Run `python -I -S -B test_public_only.py` separately to test temporary copies containing only public files; they are cleaned automatically, and original experiment directories are not read.

## 8. Relation to E33/E37 and Evidence Gaps

Section 1 reconnects old E37, the natural-measure panel and the independent uniform-ball recheck to the synthesis; Section 3 supplements E33's finite-window cross-architecture measurements; Section 4 supplements E37's fixed-radius and measure analysis; Sections 5–6 explain related local-geometry and readout-integral boundaries without adding an experiment number. Non-Gaussian finite-window regularity, angular contraction at fixed radius and curvature changes from Gaussian radial mixing are compatible but distinct evidence. Only the first, supported by nodewise shapes and independent rechecks, supports approximately stable coefficients in the relevant windows; the second's mere presence of contraction cannot replace that evidence.

Remaining gaps include a third GRU budget or independent estimator, enhanced copy sampling, more architectures/rules/measures and wider loss windows, readout-identity checks along complete curves, independent separation of global minimum cost from total mass, and calibration of actual training accessibility against static mass. The evidence does not support identical exponents across measures, minimum norm equal to total probability mass, local LLC equal to global BCE contraction, equivalence of all network complexities, or SGD necessarily sampling this static mass.
