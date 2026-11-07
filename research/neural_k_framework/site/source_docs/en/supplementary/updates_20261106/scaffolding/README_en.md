<!-- sync:title -->
# Scaffolding Reduces Input Sample Requirements: Paired Mod3 Results and Optimization Accessibility

<!-- sync:scope -->
Review date: 2026-11-06. Reference commit: `3834fb3f6555c7c58218462502635b16b9b377f9`, dated 2026-10-27. This public package reports the 14/16-bit Mod3 sample-trend results of 2026-11-02 and briefly connects them to the 30-bit Mod3 optimization-accessibility evidence of 2026-09-13. This revision organizes and recomputes existing records; it adds no new training measurements.

<!-- sync:takeaway -->
**Scaffolding can both help optimization and reduce the input examples needed for high generalization.** Under the measured fixed training budget, related prefix supervision shifts the high-generalization region toward fewer Mod3 inputs; earlier interventions provide complementary evidence on optimization accessibility. Optimization difficulty and rule/sample difficulty are different axes: fitting the training set, reaching a high-generalization solution, and requiring fewer inputs must be assessed separately. The reduction here concerns input examples, with auxiliary labels still providing extra supervision.

<!-- sync:status-title -->
## Public Results

<!-- sync:status-table -->
| Evidence | Actually completed | Role of the results |
|---|---|---|
| 14-bit Mod3 | 5 sample counts, 2 datasets with 3 initializations each, 60 models across both arms, all reaching 16000 steps | Complete paired sample trend; strongest gains at intermediate counts |
| 16-bit Mod3 | Original 5 counts: 60 models; counts 3000/6000: another 24; count 2500: only 1 dataset with 6 models; total 90 | Complete counts support improved sample efficiency; 2500 is reported separately as a partial panel |
| Historical 30-bit Mod3 | References to saved matched-BCE, weight-perturbation, and supervision-withdrawal records | Optimization-accessibility context, not new sample-requirement measurements |

<!-- sync:status-source -->
Status sources are `research/mod3_scaffold_sample_trend_20261102/REPORT_ZH.md` and `research/mod3_scaffold_sample_trend_16bit_20261102/results_refined/STOPPED_SUMMARY.json`. This package provides clickable [key results](key_results.json) and [source SHA256](source_sha256.json); original repository paths identify provenance, not public download links.

<!-- sync:definition-title -->
## Actual Tasks and Hints

<!-- sync:mod3-definition -->
The Mod3 input is a binary integer ordered from most to least significant bit, not the number of set bits modulo 3. For effective length L, the prefix states and output labels are:

$$
v=\sum_{j=1}^{L}x_j2^{L-j},\qquad r_0=0,\qquad r_j=(2r_{j-1}+x_j)\bmod 3,\qquad y_j=(\lfloor r_j/2\rfloor,\ r_j\bmod 2).
$$

<!-- sync:mod3-encoding -->
The final target is `y_L`. Remainders 0/1/2 are encoded as `00/01/10`, not a three-class softmax. Exact accuracy requires both thresholded endpoint bits to be correct; `11` is not a valid remainder. The 14/16 effective bits are left-padded with zeros to dimension 30. The network predicts all effective prefixes in one forward pass; auxiliary outputs are not fed back as true intermediate answers.

<!-- sync:hint-types -->
Here, scaffolding means explicitly supervising related prefix subtasks through auxiliary outputs. `direct` and `scaffold` use the same inputs and network, while the latter additionally supervises shorter prefixes. It adds neither input dimensions nor intermediate labels at test time.

<!-- sync:information-boundary -->
**Input-example and auxiliary-label budgets are counted separately.** Each distinct input has 2 endpoint label bits; 14-bit scaffold additionally supervises 26 prefix label bits, and 16-bit adds 30. These count label coordinates, not independent bits of information. The arms match input examples, initialization, and training steps, but not total supervision information; fewer inputs must not be described as a reduction in the total supervision budget.

<!-- sync:protocol-title -->
## Paired Mod3 Protocol

<!-- sync:architecture-table -->
| Item | 14-bit | 16-bit |
|---|---|---|
| Network | 30→1024→1024→1024→28 | 30→1024→1024→1024→32 |
| Parameters per model | 2165788 | 2169888 |
| Complete input domain | 16384 | 65536 |
| Nested training pool / fixed test set | 8192 / 8192 | 32768 / 32768 |
| Complete sample counts | 256, 1024, 2048, 4096, 8192 | 1024, 2048, 3000, 4096, 6000, 8192, 32768 |

<!-- sync:architecture-detail -->
Each hidden layer uses GELU followed by LayerNorm, with LayerNorm epsilon `1e-5` and no dropout. Both arms use exactly the same multi-output network. `direct` supervises only the endpoint bits, with zero objective gradient on unused auxiliary outputs; `scaffold` adds prefix supervision. Width, output dimension, and parameter count are matched between arms at each length. Output dimensions differ across lengths, so initial weights must not be described as identical across lengths. See the `research/mod3_scaffold_sample_trend_20261102/experiment.py` ([source SHA256](source_sha256.json)).

<!-- sync:loss-definition -->
Let B_j be the mean BCE over training inputs and both bits of prefix j. The current objectives are:

$$
L_{\mathrm{end}}=B_L,\qquad L_{\mathrm{aux}}=\frac{1}{L-1}\sum_{j=1}^{L-1}B_j,\qquad L_{\mathrm{direct}}=L_{\mathrm{end}},\qquad L_{\mathrm{scaffold}}=L_{\mathrm{end}}+L_{\mathrm{aux}}.
$$

<!-- sync:loss-boundary -->
The endpoint coefficient remains 1, and the auxiliary-mean coefficient is also 1. The historical successful 30-bit protocol averaged all 60 outputs, giving `(L_end + 29 L_aux)/30`; these objectives are not constant rescalings of each other. The sample-trend experiment uses a separate protocol, with historical results serving only as optimization-accessibility context.

<!-- sync:randomization -->
Dataset seeds are `202611020` and `202611021`, each with model seeds `0/1/2`. A complete count therefore has 6 runs per arm but only 2 independent dataset permutations. The first half of each permutation forms the training pool and the second half the fixed test set; each count takes the first n distinct training-pool inputs, giving strictly nested training sets. Initializations are reused across counts at the same length. Each direct/scaffold pair shares initial weights and the minibatch sequence. Identical numeric seeds across lengths do not produce identical training sets.

<!-- sync:budget -->
AdamW uses `betas=(0.9,0.999)`, `eps=1e-8`, and weight decay 0. The learning rate is `0.001`, with linear warmup over 10 steps and fixed batch size 512; every sample count uses minibatch sampling with replacement. Every started formal model reached 16000 steps, with comparisons at budgets 4000/8000/16000. Historical execution used an RTX 3080 on `cuda:1`, float32, and TF32 disabled; recorded Torch was `2.11.0+cu130`.

<!-- sync:criteria -->
At budget T, test exact accuracy is averaged over the recorded steps `T-2000, T-1000, T`, then averaged across runs for that count; the best checkpoint is not selected. Stable 99% means each of those three observations is ≥99%, not every intervening step. Passing the fit criterion means endpoint mean BCE≤0.001 on the entire training set at the budget endpoint. Strict interpolation is counted separately as training exact=100%; the BCE threshold does not imply zero training errors. pp denotes percentage points, and differences are calculated before rounding.

<!-- sync:result14-title -->
## 14-bit: Gains at Intermediate Sample Counts

<!-- sync:table-14 -->
| Training inputs | direct test exact | scaffold test exact | Paired change / pp | Stable 99% counts |
|---:|---:|---:|---:|---:|
| 256 | 31.23% | 31.05% | -0.18 | 0/6 -> 0/6 |
| 1024 | 32.21% | 44.71% | +12.50 | 0/6 -> 0/6 |
| 2048 | 80.16% | 99.14% | +18.98 | 0/6 -> 5/6 |
| 4096 | 99.47% | 99.76% | +0.30 | 5/6 -> 6/6 |
| 8192 | 99.98% | 99.97% | -0.002 | 6/6 -> 6/6 |

<!-- sync:result14-interpretation -->
All values use the 16000-step window. The scaffold arm already has 5/6 stable successes at 2048, whereas direct first shows 5/6 at 4096, supporting lower input sample requirements on the measured grid. Count 2048 was an exploratory bridge added after observing that 1024 did not generalize highly while 4096 was nearly saturated; it does not establish an exact halving factor. There is no gain at 256, and 1024 remains below the high-generalization criterion. See `research/mod3_scaffold_sample_trend_20261102/results_combined/ANALYSIS.json` ([source SHA256](source_sha256.json)).

<!-- sync:table-14-datasets -->
At count 2048, averaging the 3 initializations within each dataset:

| Dataset seed | direct | scaffold | Change / pp |
|---:|---:|---:|---:|
| 202611020 | 97.72% | 99.23% | +1.51 |
| 202611021 | 62.60% | 99.06% | +36.46 |

<!-- sync:table-14-budgets -->
Budget comparison at count 2048:

| Steps | direct | scaffold | Change / pp |
|---:|---:|---:|---:|
| 4000 | 74.44% | 98.67% | +24.23 |
| 8000 | 76.94% | 98.93% | +21.99 |
| 16000 | 80.16% | 99.14% | +18.98 |

<!-- sync:result14-fit -->
All 12 models at count 2048 end with training exact=100%. Mean endpoint BCE is approximately `4.95e-9` and `7.71e-9` in the respective arms. The generalization gain therefore is not merely a failure of direct to fit its training set; training fit and input sample efficiency require separate assessment. The gap narrows with budget, and the unlimited-budget gap was not measured. Across the experiment, 60/60 models pass the BCE criterion and 59/60 strictly interpolate; the exception is fully retained in [key results](key_results.json).

<!-- sync:result16-title -->
## 16-bit: High Generalization from Fewer Inputs

<!-- sync:table-16 -->
| Training inputs | direct test exact | scaffold test exact | Paired change / pp | Stable 99% counts |
|---:|---:|---:|---:|---:|
| 1024 | 31.86% | 31.87% | +0.01 | 0/6 -> 0/6 |
| 2048 | 30.62% | 37.62% | +7.00 | 0/6 -> 0/6 |
| 3000 | 30.60% | 91.59% | +60.99 | 0/6 -> 0/6 |
| 4096 | 50.00% | 98.90% | +48.91 | 0/6 -> 2/6 |
| 6000 | 61.38% | 99.86% | +38.48 | 0/6 -> 6/6 |
| 8192 | 90.33% | 99.92% | +9.59 | 3/6 -> 6/6 |
| 32768 | 99.98% | 99.99% | +0.01 | 6/6 -> 6/6 |

<!-- sync:result16-interpretation -->
All values use the 16000-step window, with 6 runs per arm for complete counts. At 6000, scaffold achieves 6/6 stable 99% successes versus 0/6 for direct; direct still has only 3/6 at 8192. Both arms reach 6/6 at 32768, showing that direct also generalizes highly with more inputs. Although scaffold at 3000 has a higher mean than direct at 8192, it still has 0/6 stable successes: mean accuracy and the stability criterion are distinct. There is almost no gain at 1024. See `research/mod3_scaffold_sample_trend_16bit_20261102/results/ANALYSIS.json` ([source SHA256](source_sha256.json)) and `research/mod3_scaffold_sample_trend_16bit_20261102/results_refined/ANALYSIS.json` ([source SHA256](source_sha256.json)).

<!-- sync:table-16-budgets -->
| Training inputs | Steps | direct | scaffold | Change / pp |
|---:|---:|---:|---:|---:|
| 3000 | 4000 | 30.14% | 86.30% | +56.16 |
| 3000 | 8000 | 30.31% | 89.10% | +58.79 |
| 3000 | 16000 | 30.60% | 91.59% | +60.99 |
| 6000 | 4000 | 54.43% | 99.82% | +45.39 |
| 6000 | 8000 | 57.28% | 99.85% | +42.57 |
| 6000 | 16000 | 61.38% | 99.86% | +38.48 |

<!-- sync:result16-datasets -->
Count 6000 improves on both datasets: dataset seed `202611020` gives 83.65%→99.85%, and `202611021` gives 39.12%→99.88%. These are means over 3 initializations per dataset, not confidence intervals. Gain sizes vary with the data permutation; 2 independent datasets do not support a robust population sample-threshold estimate.

<!-- sync:result16-fit -->
At 3000, 4096, and 6000, both arms end with training exact 100%; mean endpoint BCE at 3000 is approximately `9.52e-9` and `1.02e-8`. Of all 90 models, 88 pass BCE≤0.001 and 84 strictly interpolate; every exception is retained in [key results](key_results.json). High test accuracy, the BCE criterion, and strict training interpolation remain separately reported.

<!-- sync:result16-partial -->
Count 2500 completed only the 3 model pairs for dataset seed `202611020`: direct 31.72%, scaffold 54.77%, change +23.05 pp, and stable successes 0/3→0/3. All reached 16000 steps; the second dataset was not run, so this point is separate from the complete 6-run main table.

<!-- sync:result16-stop -->
Complete refinements are 3000/6000; counts 3500/5000/7000 were not started, and proposed sample counts 12000/16000 were cancelled. The 16000 training steps are not another executed sample count. The `STOP_AFTER_GROUP` marker and stopped status remain unchanged; [key results](key_results.json) retain complete, partial, unstarted, and cancelled counts.

<!-- sync:history-title -->
## Connection to Earlier Optimization-Accessibility Evidence

<!-- sync:history-summary -->
The 30-bit Mod3 records dated 2026-09-13 separately observed that scaffolding from initialization reached higher generalization in an endpoint-BCE-matched comparison; changing weight location and the subsequent path with fixed labels helped access high-generalization states; and continued training after withdrawing auxiliary supervision could recover high generalization. These support an optimization benefit and path-dependent accessibility, but are not new measurements of sample reduction.

<!-- sync:history-sources -->
Historical sources are `research/notes/20260913_mod3_endpoint_scaffold_matched_bce_zh.md`, `research/notes/20260913_mod3_weight_perturbation_generalization_zh.md`, and `research/notes/20260913_mod3_scaffold_withdrawal_zh.md` ([source SHA256](source_sha256.json)). This package only references saved notes, without reloading historical weights or rerunning forward passes; recovery after withdrawal does not imply uninterrupted stability.

<!-- sync:history-synthesis -->
Optimization accessibility concerns whether training reaches a high-generalization solution under specified data, objectives, and budgets. Rule/sample difficulty concerns structure, supervision information, and the data needed to distinguish candidate rules. The current paired 14/16-bit experiments show higher generalization at matched input counts and fewer required inputs for high generalization under the measured fixed budget. This complements the historical optimization evidence without equating finite-budget sample curves with purely statistical lower bounds or separating the contributions of optimization and structural benefits.

<!-- sync:limits-title -->
## Scope of the Conclusions

<!-- sync:limits -->
- Gains concern related prefix supervision and measured training budgets; fewer inputs do not mean less total label information.
- Grid extensions were exploratory and independent datasets are limited; no exact transition, universal saving factor, or unlimited-budget conclusion is reported.
- This package does not measure a relative reduction in static Neural K or identify a specific internal DFA; unrelated auxiliary-task and cross-architecture controls remain open.

<!-- sync:audit-title -->
## Recomputation Scope and Public Materials

<!-- sync:audit-results -->
The [recomputation script](recompute.py) uses only the Python standard library, imports neither NumPy nor Torch, and loads no weights. Full local mode recomputes 150 models, 78 summary rows, and 225 paired rows from per-run JSON. The maximum statistical discrepancy is approximately `2.22e-16`, consistent with numerical rounding; frozen sources, completion states, and the stop marker are also checked. Counts and errors are reported in the [recomputation audit](audit.json).

<!-- sync:audit-boundary -->
Full mode requires original records under `research/mod3_scaffold_sample_trend_20261102/` and `research/mod3_scaffold_sample_trend_16bit_20261102/`, plus implementations under `research/ca_phase_transition/` and historical notes under `research/notes/` listed in the source manifest. Data separation, gradient pairing, and forward checks come from historical verification records. This revision reruns none of those computations and does not modify original experiment files.

<!-- sync:files -->
| File | Purpose |
|---|---|
| `README_zh.md`, `README_en.md` | Paragraph-aligned bilingual explanations and paired data tables |
| [key_results.json](key_results.json) | Paired Mod3 results, budgets, label counts, fitting exceptions, and historical references |
| [source_sha256.json](source_sha256.json) | Original-source paths and SHA256, plus checksums for the other package files |
| [audit.json](audit.json) | Scope, errors, and bilingual consistency of read-only local recomputation |
| [recompute.py](recompute.py) | `--public-only` independently checks the public package; default mode requires original records; `--write` updates generated JSON |

<!-- sync:commands -->
After downloading and extracting only this package, run the following from its directory. No original repository, third-party dependencies, or GPU are required:

```powershell
python -I -S -B -X utf8 recompute.py --public-only
```

<!-- sync:public-boundary -->
Public mode reads only package files, checking Mod3 summary relationships, complete/partial panels, auxiliary-label budgets, text tables, bilingual numbers and formulas, package links, and file hashes. It reads no original research directories and does not independently recompute per-model metrics or historical measurements. Source hashes record provenance and integrity but do not replace source contents; the manifest does not hash itself and is not a digital signature. Neither mode starts training or sampling.

<!-- sync:full-command -->
With all original materials and the repository layout preserved, run full read-only recomputation from the repository root. `--write` only updates generated JSON in this directory and cannot be combined with `--public-only`.

```powershell
python -I -S -B -X utf8 research/neural_k_framework/supplementary/updates_20261106/scaffolding/recompute.py
```
