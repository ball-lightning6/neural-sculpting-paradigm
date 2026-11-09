# Amplitude and Norm Interventions in Actual Training: Read-Only Audit

<!-- block:scope -->
## 1. Scope and Evidence Levels

Audit date: 2026-11-06; baseline commit: `3834fb3f6555c7c58218462502635b16b9b377f9` (2026-10-27). This supplement covers three training interventions dated 2026-11-03 after that baseline. Source files are identified by paths and byte fingerprints in [sources_sha256.csv](sources_sha256.csv). The [Chinese version](README_zh.md) corresponds fully to this page through matching paragraph identifiers.

<!-- block:sources -->
Original source directories are `research/copy_mse_amplitude_sgd_20261103/`, `research/rule30_high_norm_penalty_20261103/`, and `research/rule30_low_norm_penalty_20261103/`. The audit used their code, protocols, MANIFEST, SUMMARY, HISTORY, per-model statistics, and saved arrays. These directories and full original reports are not included; paths are provenance identifiers, not download links. Clickable evidence consists of this package's [copy summary](copy_summary.csv), [Rule30 summary](rule30_summary.csv), [per-seed paired differences](rule30_paired.csv), and [audit record](audit_summary.json). This supplement is an offline audit of existing results, not new training, SMC, or CIFAR/MNIST label-noise controls. **Training endpoint frequencies are not static parameter mass, and parameter norm is not function complexity.**

<!-- block:copy_design -->
## 2. Copy: Target Amplitude and Full-Batch SGD

The fixed network is `4→16→1 tanh`, with 97 trainable parameters, linear output, 0/1 inputs, and the rightmost-input-bit copy rule. Raw coordinates have independent standard Gaussian initialization; hidden weights and biases are divided by 2, and output weights and biases by 4. The old Adam archive supplies 16 distinct splits, each with 12 training points, 4 unseen points, and 32 initializations: 512 paired models per condition, with initialization seeds 71001 through 71512. All amplitudes use the same initial state without scaling the output-head initialization. Full rule recovery means correct classification on all 16 inputs; unseen accuracy averages each model's correct fraction over its 4 held-out inputs across models.

<!-- block:copy_loss -->
### 2.1 Objective, Relative Error, and Absolute Error

Let `s=2y-1` and let `z` denote the raw network output. MSE has no sigmoid; classification consistently uses `z>0`. Optimization uses relative MSE, with BCE as a separate reference:

$$
L_{\mathrm{raw},A}=\operatorname{mean}[(z-As)^2],\qquad
L_{\mathrm{rel},A}=L_{\mathrm{raw},A}/A^2=\operatorname{mean}[(z/A-s)^2],\qquad
L_{\mathrm{BCE}}=\operatorname{mean}\operatorname{softplus}(-sz).
$$

<!-- block:copy_thresholds -->
Sufficient fitting requires every training classification to be correct and `L_rel<=1e-4` for MSE, or `L_BCE<=1e-3` for BCE. Thus the absolute MSE cutoff is not held constant across amplitudes. Actual learning rates below apply to relative MSE; the primary BCE learning rate is 0.3.

| A | Raw MSE fitting cutoff | Primary actual learning rate | Half-rate actual learning rate |
|---:|---:|---:|---:|
| 0.001 | 1e-10 | 3e-7 | 1.5e-7 |
| 0.01 | 1e-8 | 3e-5 | 1.5e-5 |
| 0.1 | 1e-6 | 3e-3 | 1.5e-3 |
| 1 | 1e-4 | 0.3 | 0.15 |
| 10 | 1e-2 | 30 | 15 |

<!-- block:copy_rates -->
The optimizer is full-batch SGD without momentum or weight decay, equivalently full-batch gradient descent, not a minibatch-noise control. Calibration uses only training labels and 32 new initializations (seeds 91001 through 91032, 2 per split). Candidate raw-MSE-equivalent learning rates are `{0.001,0.003,0.01,0.03,0.1,0.3}`, each run for 2000 steps. The selected candidate must be numerically valid for every model and minimize the 90th percentile of training relative loss; all six conditions selected 0.3. Calibration comprises 1152 condition runs but only 32 independent new initializations. Primary groups contain 512 models each; half-rate groups use the first 8 original initializations per split, 128 each, for 3840 formal models in total.

<!-- block:copy_equivalence -->
For MSE, the actual relative-loss learning rate is `eta_rel=0.3*A^2` (`0.15*A^2` for half-rate controls), making each update exactly equivalent to optimizing raw MSE at learning rate 0.3. Every layer uses the same scalar, without inter-layer preconditioning. This equivalence does not make different amplitudes the same objective or make SGD equivalent to Adam.

$$
\theta^+=\theta-\eta_{\mathrm{rel}}\nabla L_{\mathrm{rel},A}
=\theta-0.3\nabla L_{\mathrm{raw},A}.
$$

<!-- block:copy_stop -->
### 2.2 Stopping Rule and Actual Completion

Evaluation occurs every 200 steps, with at least 18000 steps. Fitting-based stopping requires all 3840 models to be sufficiently fitted at three consecutive evaluation nodes; otherwise the cap is 100000 steps. Individual models are not frozen after first fitting. Final, best-training-loss, and first-fit states are saved, without test-based checkpoint selection. Wall-clock limits, STOP, and manual interruption only pause a run and are not scientific endpoints. The actual status is `completed_step_cap` at 100000 steps, which does not mean every model converged. Old Adam instead ran for a fixed 18000 steps; fitting cutoffs were used for assessment and first-fit recording, not early stopping.

<!-- block:copy_primary -->
### 2.3 Primary Results and Underfitting

The table uses each model's saved best-training-loss checkpoint and separates all-model and fitted-subset denominators. Final checkpoints give the same classification statistics shown here; first-fit checkpoints also give the same fitted-subset statistics.

| Target | Sufficiently fitted/512 | Unseen accuracy, all models | Unseen accuracy, fitted subset | Full recovery, fitted subset |
|---|---:|---:|---:|---:|
| MSE ±0.001 | 352/512 | 48.9746% | 48.7216% | 26/352 |
| MSE ±0.01 | 473/512 | 60.1074% | 59.1966% | 68/473 |
| MSE ±0.1 | 512/512 | 97.7051% | 97.7051% | 471/512 |
| MSE ±1 | 512/512 | 100% | 100% | 512/512 |
| MSE ±10 | 512/512 | 100% | 100% | 512/512 |
| BCE | 512/512 | 100% | 100% | 512/512 |

<!-- block:copy_selection -->
At ±0.001 and ±0.01, respectively, 160 and 39 models fail the strict fitting cutoff and must not be counted as generalization failures after sufficient fitting. However, 352 sufficiently fitted ±0.001 models still have near-chance unseen accuracy, so underfitting does not explain the entire phenomenon. The fitted subset is selected, not an unbiased estimate for all original models. Initializations are repeated within 16 splits, so models are not independent dataset replications. These results support an amplitude-associated difference in this protocol, not a universal monotonic law or equality of complete function distributions.

<!-- block:copy_half -->
Half-rate controls use the same 100000-step cap. Small amplitudes fit more slowly; the trend remains, but these are not fully converged group comparisons.

| Target | Sufficiently fitted/128 | Unseen accuracy, fitted subset |
|---|---:|---:|
| MSE ±0.001 | 40/128 | 50.6250% |
| MSE ±0.01 | 80/128 | 63.4375% |
| MSE ±0.1 | 128/128 | 96.8750% |
| MSE ±1 | 128/128 | 100% |
| MSE ±10 | 128/128 | 100% |
| BCE | 128/128 | 100% |

<!-- block:copy_adam -->
### 2.4 Comparability with Original Adam

Directly shared old Adam conditions are only BCE, ±1, and ±0.001. The old series also contains ±0.1 and ±0.01 follow-ups, but those are not the three-condition original archive supplying the reused initial states. Shared conditions have identical data, initialization, network, and relative fitting criteria. Old Adam used relative MSE, initial learning rate 0.01, multiplication by 0.3 after steps 2000, 5000, 9000, 13000, and 16000, `eps=1e-8`, weight decay 0, and CPU float64. New SGD uses independently calibrated constant rates, CUDA1 float64 with TF32 disabled, and a budget extended to 100000 steps. This is an optimizer-robustness retest, not an equal-budget causal comparison changing only the optimizer name. Scaling an entire Adam objective must not be described as strictly numerically equivalent without checking.

<!-- block:copy_adam_results -->
In the old three-condition archive, all 512 models per group were sufficiently fitted: ±0.001 achieved 49.3164% unseen accuracy and 30/512 full recovery; ±1 and BCE both achieved 100% and 512/512 full recovery. The direction agrees with new SGD, but the new small-amplitude group has a lower training-fit rate. Old values come from `tmp/4bit_copy_small_target_local_20260910/analysis.json`; this audit checks provenance and shared initial states without rerunning Adam.

<!-- block:copy_intermediate -->
The retained 18000-step SGD intermediate snapshot was also checked: its status is `paused_at_checkpoint`, not final completion. Only 8/512, 45/512, and 329/512 primary models were sufficiently fitted at ±0.001, ±0.01, and ±0.1. Main results were obtained only after reaching 100000 steps. Thus old Adam and this intermediate snapshot, both at 18000 steps, must not be described as an equal-budget comparison in which all models have fitted.

<!-- block:copy_scale -->
### 2.5 What Last-Layer Scaling Was Actually Tested

A unit check was performed: SGD code multiplies both last-layer weights and bias by `c`, also multiplies target amplitude by `c`, and checks relative loss and gradient-scaling relations. Saved maximum gradient errors for `c=0.001,10` are both approximately `1.11e-16`; the old Adam independent pure-unit-change gradient check reports approximately `5.55e-17`. These historical unit tests were not rerun in this audit.

$$
z_{(\theta_h,c\theta_o)}=c z_{(\theta_h,\theta_o)},\qquad
L_{\mathrm{rel},cA}(\theta_h,c\theta_o)=L_{\mathrm{rel},A}(\theta_h,\theta_o),\qquad c>0.
$$

<!-- block:copy_scale_limit -->
**No executed formal generalization control freezing hidden layers and only retraining or scaling the last layer was found.** The identity, gradient check, and post-training layer-displacement statistics must not be presented as that causal control. In the primary fitted subsets, median relative hidden-feature displacement is 0.2080, 0.2050, and 0.2104 at ±0.001, ±0.01, and ±0.1, despite very different generalization; it is 0.4034 and 1.0880 at ±1 and ±10. Displacement is a correlational diagnostic: it neither establishes that larger changes guarantee generalization nor isolates feature direction, readout scale, and residual output.

<!-- block:ca_design -->
## 3. Rule30: High-Norm Penalty and Low-Norm Repulsion

The task uses one step with periodic boundaries, 30-bit inputs, and 30-bit outputs, labeled by `left XOR (center OR right)`. The network is `30→1024→1024→1024→30`; all three hidden layers use Linear, GELU, and LayerNorm (`eps=1e-5`), with 2167838 total parameters. Training uses nested prefixes `{600,750,900,1200}` of one historical pool. Seeds `{0,1,2,3}` are initialization repeats, not four independent training datasets. Monitor and probe each contain 4096 inputs and are disjoint from the training pool and each other. Primary results use probe alone, not a newly combined 8192-input result.

<!-- block:ca_optimizer -->
Each run has 32 models (16 intervention models and 16 rerun baselines), totaling 64 model runs. Baselines are exact duplicates, leaving only 48 distinct conditions in the three-arm comparison; repeated baselines are not additional seeds. Old initial weights are reused. Models at the same sample count share with-replacement batch indices, with index seed 314159 and batch size 512. AdamW uses learning rate 0.001, weight decay 0, float32, TF32 enabled, and CUDA1 for a fixed 20000 steps. Evaluation occurs every 500 steps, with additional initial and early nodes saved. High/low interventions share the same new batch schedule, not necessarily the stepwise schedule of the old 900/1500 scan. BCE uses numerically stable `softplus(-signed_y*z)`; old BCEWithLogits differs numerically at extremely low loss, motivating freshly paired baselines.

<!-- block:ca_norm -->
### 3.1 Norm, Formulas, and Actual Scan Coverage

The penalty includes only weights and biases of the four linear layers, totaling `P_linear=2161694` parameters. LayerNorm gamma and beta remain trainable but are not directly penalized. Original linear initialization is uniform; variance normalization does not imply a Gaussian measure over the whole network.

$$
r^2=\frac{1}{P_{\mathrm{linear}}}\sum_j\frac{\theta_j^2}{\sigma_j^2},\qquad
\sigma_j^2=\frac{1}{3\,\mathrm{fan\_in}_j}.
$$

<!-- block:ca_penalties -->
The reference is the median radius of eight old BCE endpoints at `n=900`, `r_med=1.2291480798`. Cutoffs were fixed in advance at `r_H=0.85*r_med=1.0447758679` and `r_L=1.5*r_med=1.8437221198`, without adjustment to current test results. Both runs compare only `lambda=0` against `lambda=0.1`; **there was no dense lambda-range scan or cutoff search**.

$$
J_H=L_{\mathrm{BCE}}+\lambda[\max(0,r^2/r_H^2-1)]^2,\qquad
J_L=L_{\mathrm{BCE}}+\lambda[\max(0,1-r^2/r_L^2)]^2.
$$

<!-- block:ca_directions -->
The high-norm penalty pushes inward only when `r>r_H`; low-norm repulsion pushes outward only when `r<r_L`, becoming zero beyond its cutoff. Neither is a hard radius constraint, decoupled weight decay, or a strength-symmetric positive/negative weight-decay pair. Low-norm repulsion does not push norms to infinity. Its actual endpoint radii are 1.886650 to 1.892699, approximately 2% to 3% above cutoff, with zero endpoint penalties. High-penalty radii are 1.044651 to 1.044777, with maximum endpoint penalty only `5.1301e-13` and a tiny cutoff exceedance. Thus endpoints are not all strictly inside a hard constraint, nor are trajectories continuously subject to an unchanged radial force.

<!-- block:ca_success -->
### 3.2 Fitting, Success Criteria, and Test Denominators

Sufficient fitting means 100% training whole-sample accuracy and raw BCE no greater than `1e-4`. Stable generalization additionally requires probe whole-sample accuracy of at least 99% at each of the last three evaluation nodes (19000, 19500, and 20000 steps). This is a success classification after a fixed budget, not test-driven early stopping; training actually reached 20000 steps. All 32 models in each run were sufficiently fitted. Maximum training BCE is `2.9231732e-9` in the high-penalty run and `2.8179692e-9` in the low-repulsion run. This audit checks saved metrics for that conclusion; it does not load large-network weights to recompute BCE by forward evaluation.

<!-- block:ca_accuracy_def -->
<span id="rule30-norm-agreement"></span>Whole-sample/exact accuracy requires all 30 output bits of an input to be correct simultaneously; bit accuracy averages individual output positions. Both accuracies are computed per model and then averaged equally across four seeds. **Whole-ring agreement** ignores the label: did two seeds give the same 30-bit answer to the same input? The table averages over 4096 held-out inputs and all 6 distinct pairs of the 4 seeds at step 20000. This does not require identical functions on the entire probe set, nor is it an exhaustive test of all `2^30` inputs.

$$
\mathrm{Acc}_{\mathrm{whole}}=\frac1N\sum_i\mathbf1[\forall b,\hat y_{ib}=y_{ib}],\qquad
\mathrm{Acc}_{\mathrm{bit}}=\frac1{30N}\sum_{i,b}\mathbf1[\hat y_{ib}=y_{ib}].
$$

<!-- block:ca_table -->
| n | Arm | Whole-sample accuracy | Whole-ring agreement | Bit accuracy | Stable successes | Sufficiently fitted |
|---:|---|---:|---:|---:|---:|---:|
| 600 | baseline | 68.5181% | 58.0973% | 98.6302% | 0/4 | 4/4 |
| 600 | high_norm | 79.6570% | 71.4559% | 99.1982% | 0/4 | 4/4 |
| 600 | low_norm | 39.5203% | 29.1748% | 96.5871% | 0/4 | 4/4 |
| 750 | baseline | 99.2188% | 98.6125% | 99.9738% | 3/4 | 4/4 |
| 750 | high_norm | 99.7620% | 99.5483% | 99.9921% | 4/4 | 4/4 |
| 750 | low_norm | 95.0256% | 91.8335% | 99.8234% | 0/4 | 4/4 |
| 900 | baseline | 99.9939% | 99.9878% | 99.9998% | 4/4 | 4/4 |
| 900 | high_norm | 100% | 100% | 100% | 4/4 | 4/4 |
| 900 | low_norm | 99.9756% | 99.9512% | 99.9992% | 4/4 | 4/4 |
| 1200 | baseline | 100% | 100% | 100% | 4/4 | 4/4 |
| 1200 | high_norm | 100% | 100% | 100% | 4/4 | 4/4 |
| 1200 | low_norm | 100% | 100% | 100% | 4/4 | 4/4 |

With 600 baseline training examples, all four seeds got 1544 of the 4096 unseen inputs right, some but not all got 2238 right, and all missed 314. Now count seed-pair/input cases: both models missed 4014 times, giving the same wrong string 1162 times and different wrong strings 2852 times (71.05%). The runs did not always give the same wrong answer; this does not test error independence **across different inputs**. [Per-pair counts](seed_pairs.csv) · [Group totals](group_counts.csv) · [Hard endpoint predictions](endpoints.csv.gz) · [Check script](check_agreement.py) (run `python check_agreement.py`; no GPU required).

<!-- block:ca_grid -->
### 3.3 Only a Coarse Transition Grid

At 600 and 750 samples, high-norm penalties improve all four paired seeds over baseline, averaging +11.1389 and +0.5432 percentage points; low-norm repulsion worsens all four, averaging -28.9978 and -4.1931 percentage points. The smallest **tested grid levels** with at least 3/4 stable successes are baseline 750, high_norm 750, and low_norm 900. High-norm penalization does not establish a leftward threshold shift on this grid. The low-repulsion change from 750 to 900 is a coarse fixed-budget trend on one data sequence, not an exact n50/n90 or an estimate that the true threshold increases by 20%. Differences are small or absent at 900 and 1200. Longer training might recover the gap; no additional training is performed here.

<!-- block:ca_loss_match -->
Records at `1e-2`, `1e-3`, and `1e-4` are merely the first saved evaluation nodes at which each model has all training classifications correct and crosses the BCE cutoff. Actual BCE and update counts differ. `MATCHED_THRESHOLDS.csv` is not an exact matched-loss comparison. Final BCE is slightly higher with the high-norm penalty than with baseline despite better generalization, so the improvement does not result from a lower terminal training BCE. This also does not establish equal endpoint distributions or optimization paths.

<!-- block:ca_gauge -->
### 3.4 LayerNorm and the Complexity Boundary

Multiplying a hidden layer's gamma and beta by a positive scalar and dividing the next linear layer's weights by it can preserve the function while changing raw norm. Historical code also absorbs the preceding LayerNorm affine parameters into the next linear layer to compute an effective radius:

$$
W_{\mathrm{eff}}=W\operatorname{diag}(\gamma),\qquad b_{\mathrm{eff}}=b+W\beta.
$$

<!-- block:ca_gauge_limit -->
Relative to paired baseline, effective radius is approximately 0.864 to 0.871 with the high-norm penalty and 1.459 to 1.475 with low-norm repulsion, so the change is not solely that particular scale compensation. But effective radius is neither a complete reparameterization invariant nor a complexity measurement. Interventions simultaneously change objective geometry, effective step sizes, feature formation, and reachable paths. There is no static sampling or candidate-complexity measurement here, so this experiment alone cannot establish a static-volume mechanism, much less a unique mechanism.

<!-- block:compare -->
## 4. E30/E31 Connections and Non-Comparable Protocols

This supplement provides **an SGD training retest related to E31's copy amplitude series** and **Rule30 training-intervention evidence related to E30's static-selection results**. E30 uses a fixed Gaussian reference measure, an `8→16→16→1 tanh` network, 40 training inputs, and constrained SMC for the static AND task. Current Rule30 results are training outcomes in a large LayerNorm network. Task, architecture, measure/initialization, norm definition, conditioning event, and output denominators differ. Numerical results or regularization coefficients cannot be pooled or presented as a closed causal chain.

<!-- block:e30 -->
Historical primary E30 conditions define `J=BCE+lambda*||theta||^2/2`, keeping the reference measure fixed while changing the low-objective constraint region. This table is transcribed and checked against the published report, not recomputed with SMC. Raw BCE is approximately, not exactly, matched.

| lambda | Mean raw BCE | Static mass of complete AND | Observed complete functions |
|---:|---:|---:|---:|
| 0 | 0.00258150 | 53.683% | 4190 |
| 5e-5 | 0.00251046 | 58.423% | 1246 |
| 1e-4 | 0.00258978 | 73.318% | 86 |

<!-- block:e30_limit -->
This is direct static-selection evidence, not something to replace with actual training endpoints. Conversely, E30's unpenalized, loss-stratified low-norm AUC median is 0.5088, not evidence that correct functions inherently always have lower norm. Endpoint raw BCE differs by only about 0.32%, but all three points are not exactly matched. Deeper all-AND observed modes have insufficient sampling diagnostics and must not be upgraded to an exact static probability of 100%. Source path: `research/neural_k_framework/experiments/experiment_30_weight_decay_static_landscape/RESULTS_AND_CONCLUSION.md`; the original file is not included.

<!-- block:e31_ca -->
Old E31 centered Rule30 MSE also differs from current copy normalization: old optimization uses raw MSE without dividing by squared amplitude. For ±0.5, ±0.25, and ±0.1, the fitting diagnostic is `MSE/(2A)^2<=1e-6`, corresponding to absolute cutoffs `1e-6,2.5e-7,4e-8`. Old runs use 900/1500 samples, eight seeds, 25000 steps, and two evaluation sets totaling 8192 inputs. Current norm interventions use four seeds, 20000 steps, and a separate 4096-input probe. These are protocol background only; old accuracies are not appended to the new three-arm curve. Source paths: `research/neural_k_framework/experiments/experiment_31_loss_amplitude_grokking/RULE30_CENTERED_MSE_AMPLITUDES.md` and `research/neural_k_framework/experiments/experiment_31_loss_amplitude_grokking/RULE30_MSE_PM01.md`; the original files are not included.

<!-- block:audit -->
## 5. Verification Modes and Package Scope

[audit_cpu.py](audit_cpu.py) offers two verification modes, neither running training, sampling, or Torch. `--public-only` checks consistency of the bundled tables; the default mode requires unbundled local original results and NumPy. The default mode previously independently recalculated final/best/first-fit copy outputs, losses, classifications, complete function IDs, and layer-norm/feature diagnostics, and checked the intermediate snapshot. It reconstructed Rule30 labels from input IDs and decoded monitor/probe predictions at every saved node, checking whole-sample and bit accuracy, the last-three-node success criterion, first threshold crossings, and duplicated baselines. CA BCE, parameter norms, and training fitting came from saved metrics, not new forward passes through large-network weights. Old radius-reference weights were hashed, not reprocessed to recalculate tensor radii. Historical gradient, CUDA Graph, resume-consistency, and file-lock tests were inspected as records, not rerun. Errors, source counts, and bilingual checks from that local audit are recorded in [audit_summary.json](audit_summary.json); they do not mean that public mode repeated these original-data checks.

<!-- block:commands -->
After downloading this package, run the following from its directory. Public mode requires only the Python standard library, not NumPy, the full repository, original weights, or datasets. It recalculates insufficient-fit counts, mean paired differences, and the smallest sample levels implied by published success counts from CSV, and checks document tables, bilingual numbers, and package-file hashes. It does not read original files named in the source manifest or revalidate predictions, training, or original-source hashes.

```powershell
python -I -S audit_cpu.py --public-only
```

<!-- block:local_command -->
Full local verification requires the original research directories, `tmp/` archives, and historical data ZIP, acquired separately and kept in the original repository layout. Downloading this package alone is insufficient for that mode. With those materials present, run the following from the repository root; Python and NumPy are required. `--archive` can specify a relocated historical data package. The script neither installs dependencies nor downloads files, and missing original material is not silently treated as a successful full audit.

```powershell
python -I research/neural_k_framework/supplementary/updates_20261106/training_controls/audit_cpu.py
```

<!-- block:write_command -->
`--public-only` is always read-only and cannot be combined with `--write` or `--archive`. Default local verification fails on differences; only explicit `--write` updates the five derived summaries. The script has no training, resume, SMC, or GPU entry point and does not execute historical `run.py`, `analyze.py`, or `compare.py`. The SHA256 list is a historical record of source-byte fingerprints, not a download manifest or cryptographic proof of chronology or scientific validity. Public mode can only check package-file hashes and consistency of manifest records.

<!-- block:deliverables -->
Package files: [README_zh.md](README_zh.md), [README_en.md](README_en.md), [audit_cpu.py](audit_cpu.py), [copy_summary.csv](copy_summary.csv), [rule30_summary.csv](rule30_summary.csv), [rule30_paired.csv](rule30_paired.csv), [sources_sha256.csv](sources_sha256.csv), and [audit_summary.json](audit_summary.json). CSV files preserve unrounded fractions; document percentages are rounded for display. Bilingual paragraph numbers, displayed formulas, and tables use identical conventions. The copy summary includes all three checkpoint types and both learning-rate arms. Rule30 summaries include whole-sample/bit accuracy, fitting/stable success, radius, and penalty; the paired table retains per-seed differences. The endpoint predictions, agreement counts and check script added on 2026-11-09 are in the same package and linked in Section 3.2.

<!-- block:publish -->
**Public-package capabilities:** the bilingual descriptions, compact CSV/JSON, source-hash records, and script support consistency checks on published summaries. A public-mode `PASS` means only that these package-level checks succeeded. **It does not revalidate original training or static sampling, nor mean that downloading the package enables experiment reruns or original-result recalculation.** E30 appears only as historical-report context, without particle archives; public mode cannot independently validate the old-experiment values in this package.

<!-- block:local -->
**Not included:** original copy `RESULTS.npz`, intermediate snapshots, and old Adam archives; Rule30 `DATA.npz`, `PREDICTIONS.npz`, initial/final/best weights, and optimizer checkpoints; the historical data ZIP, old radius-reference weights, raw logs, and machine-path records. Original research notes and full reports are referenced only through source paths and hashes, not published in full. The local audit hashes consumed arrays, reference files, and text material, not every large checkpoint byte. Excluding these materials defines the package's scope; it does not imply that synthetic data are inherently confidential.
