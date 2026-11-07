# Original Mod3 training sources: 14-bit and 16-bit

[简体中文](README_zh.md) · [Download the original source ZIP](original_sources.zip)

This attachment contains the sources for two paired training experiments run on 2026-11-02. To read the implementation, start with the 14-bit `experiment.py`. To check the download without running it, go to the last section. Packaging involved no training, smoke test, model forward pass or GPU use. All accuracies below come from the original author's saved runs.

## What the two experiments compare

Given a binary integer, most significant bit first, predict its remainder modulo 3, not the number of set bits. After bit `j`, the prefix remainder is `r[j] = (2*r[j-1] + x[j]) % 3`, starting at zero. Remainders 0, 1 and 2 are encoded as `00`, `01` and `10`. The arms use identical networks. The `direct` arm supervises only the last two output bits. The `scaffold` arm adds the mean BCE over all shorter prefix labels to that same endpoint BCE, with coefficient 1 on each term. At test time both arms receive only input bits. Fewer input examples do not mean less total supervision.

| Task | Input and outputs | Split per dataset permutation |
|---|---|---|
| 14-bit: sweep, then add one intermediate count | 14 bits left-padded to 30; 28 outputs, with auxiliary targets for the first 13 prefixes | 8192 inputs in the training pool; 8192 in the fixed test set |
| 16-bit: repeat the comparison on longer inputs | 16 bits left-padded to 30; 32 outputs, with auxiliary targets for the first 15 prefixes | 32768 inputs in the training pool; 32768 in the fixed test set |

Each count takes the first `n` distinct inputs of its training pool, so the training sets are nested. A complete count uses two permutations (seeds `202611020`, `202611021`) and three initializations (`0/1/2`) per permutation: six pairs of models, not six independent datasets. Each pair shares initial weights and minibatches. The backbone is a three-hidden-layer, width-1024 GELU + LayerNorm MLP. Training uses AdamW, zero weight decay and dropout, learning rate 0.001 with 10 warmup steps, and batches of 512 sampled with replacement.

## What actually ran

Endpoint exact accuracy requires **both** remainder bits to be correct. At each budget of 4000, 8000 or 16000 steps, the analysis averages test exact accuracy at `T-2000, T-1000, T`, then averages across runs. It does not select the best checkpoint. “Stable 99%” requires at least 99% at each of the three measurements; training fit is a separate endpoint training BCE threshold of ≤ 0.001 at the budget endpoint. Examples below use the 16000-step window, in **direct → scaffold** order.

| Historical stage | Training input counts actually run | Models trained | Example test exact accuracy |
|---|---|---:|---|
| 14-bit sweep and midpoint | 256, 1024, 4096, 8192; then 2048, all complete | 48 + 12 = 60 | 2048: 80.16% → 99.14%; stable 99%: 0/6 → 5/6 |
| 16-bit sweep | 1024, 2048, 4096, 8192, 32768, all complete | 60 | 32768: 99.98% → 99.99%; stable 99% in 6/6 runs in each arm |
| 16-bit refinement, complete counts | 3000, 6000 | 24 | 3000: 30.60% → 91.59%; 6000: 61.38% → 99.86% (stable 99%: 0/6 → 6/6) |
| 16-bit refinement, partial count | 2500, dataset seed `202611020` only; the second dataset was never run | 6 | 31.72% → 54.77%; stable 99% in 0/3 runs in either arm. Excluded from full-panel comparisons. |

**The original author stopped the 16-bit queue manually.** The active group finished and was saved. Counts 3500, 5000 and 7000 never started; the proposed counts of 12000 and 16000 **inputs** were cancelled. The 16-bit total is 90 models. Every formal model that started in either experiment reached 16000 **steps**. The measurements show better input sample efficiency on the tested grid and budgets, not an exact transition point. The original 14-bit record is `results_combined/ANALYSIS.json`. For 16-bit, consult `results/ANALYSIS.json`, `results_refined/ANALYSIS.json` and `results_refined/STOPPED_SUMMARY.json` separately. These records are not in the ZIP.

## Source map

Extract into a **new, empty directory**, preserving the paths starting at `research/`. The two files named `validate_results.py` are not interchangeable. Below, 14-bit means `research/mod3_scaffold_sample_trend_20261102/`, 16-bit means `research/mod3_scaffold_sample_trend_16bit_20261102/`, and the shared modules live in `research/ca_phase_transition/`.

| Directory | File | Role |
|---|---|---|
| 14-bit | `launch.py` | Adds the Windows package path, sets environment variables including the cuBLAS configuration, and forwards arguments to the trainer. Loading this file invokes the entry point. |
| 14-bit | `experiment.py` | Core used at both lengths: builds data, checks prefix labels, defines the paired loss, checks gradients and CUDA Graph execution, trains or resumes groups, and saves evaluations. Its `main` drives the 14-bit sweep. |
| Shared | `experiment_mod3_endpoint_scaffold_matched_bce.py` | Supplies `prefix_labels`, AdamW, evaluation, state restoration and checkpoint writing. The current tasks do **not** use its historical loss or standalone training entry point. |
| Shared | `experiment_mod3_small_data_grokking.py` | Supplies integer/remainder bit encoders and `BatchedIndependentMLP`, which batches computations across models with independent parameters. Do not run its earlier experiment as a prerequisite. |
| 14-bit | `analyze.py` | Shared `analyze(out)` function. Reads the protocol and group records, computes fixed-window and paired statistics, and writes CSVs and a plot. Called by the drivers; no standalone CLI. |
| 14-bit | `extend_midpoint.py` | Reads the four-count sweep protocol and summary, checks sources, trains only the 2048 count, checks initialization and permutations, then merges the five counts. |
| 14-bit | `validate_results.py` | Uses NumPy to verify the five nested splits, saved forward differences, windows and paired statistics. Writes the combined audit; does not train. |
| 16-bit | `run_16bit.py` | Imports the 14-bit training and analysis core, configures the 16-bit sweep or smoke test, and writes separate results. If a formal 14-bit manifest exists, checks the protocol and frozen analysis source. |
| 16-bit | `validate_results.py` | Uses NumPy to reconstruct permutations and check the coarse or smoke data, pairing, saved forward differences and statistics. Does not validate the refinement. |
| 16-bit | `refine_counts.py` | Adds counts in the transition region after the sweep. Checks for a stop marker before each (sample count, dataset seed) group. |
| 16-bit | `audit_refinement.py` | Audits the coarse sweep and **all** planned refinement groups, then merges and plots them. Its completeness assertions cannot validate the historical partial run. |

## Environment and run order

Reading the source or checking the ZIP needs no third-party packages. Training needs Python, NumPy, **CUDA-enabled PyTorch** and Matplotlib (imported even when loading the older model module; plots use Agg). The historical Windows commands use the base interpreter `C:/Python313/python.exe`. The launchers add the hard-coded `.venv-gpu/Lib/site-packages` under the extraction root to the import path. The archive supplies neither the environment nor a lockfile; interpreter-compatible packages and drivers must already be available. The formal manifests record an RTX 3080 and Torch `2.11.0+cu130`; the 16-bit manifest also records NumPy `2.4.1`. Training uses float32 with TF32 off.

**Both formal trainers and both training smoke tests hard-code logical device `cuda:1` and use CUDA Graphs. There is no CPU fallback or device-selection flag.** A machine exposing only `cuda:0` cannot run these entry points unchanged. The commands below are for a separate run, not for building the site. In PowerShell, start at your own extraction root. Do not run against the original experiment directories or start two writers on one output directory.

```powershell
$py = 'C:/Python313/python.exe'
$flags = '-E', '-s', '-X', 'utf8', '-u'
$d14 = 'research/mod3_scaffold_sample_trend_20261102'
$d16 = 'research/mod3_scaffold_sample_trend_16bit_20261102'
```

| Step | PowerShell command | Prerequisite and output (relative to that experiment's directory) |
|---|---|---|
| Optional 14-bit smoke | `& $py @flags "$d14/launch.py" --smoke` | No prior results. Runs 20 actual training steps with 6 input bits, width 32 and n=16/32; writes `smoke/`. The 14-bit validator has no smoke mode. |
| 14-bit sweep | `& $py @flags "$d14/launch.py"` | No prior results; default `--hours 1.5`. Writes `results/`. The launcher accepts `--out`, but the midpoint driver always reads the default `results/`. |
| 14-bit midpoint | `& $py @flags "$d14/extend_midpoint.py"` | Needs the default sweep manifest, frozen analysis source and a summary with all 8 groups complete. Fixed 30-minute budget. New groups go to `results_midpoint/`; merged tables, plot and `PROVENANCE.json` go to `results_combined/`. |
| 14-bit five-count audit, CPU | `& $py @flags "$d14/validate_results.py"` | Requires both protocols, frozen sources, preflight records, all 10 groups' JSON/NPZ files, and the combined analysis. Writes or replaces `results_combined/AUDIT.json`. |
| Optional 16-bit smoke | `& $py @flags "$d16/run_16bit.py" --smoke` | No prior results. Keeps 16 input bits; width 32, n=256, 20 steps. Writes `smoke/`. After completion, `& $py @flags "$d16/validate_results.py" --smoke` audits it on CPU and writes `smoke/AUDIT.json`. |
| 16-bit sweep | `& $py @flags "$d16/run_16bit.py"` | Requires the 14-bit **sources**, not a completed 14-bit run. If its formal manifest exists, its matching `results/source/analyze.py` must exist too. Default `--hours 2`; writes the 16-bit `results/`. |
| 16-bit coarse audit, CPU | `& $py @flags "$d16/validate_results.py"` | Requires all 10 coarse groups, manifest, frozen sources, preflight, group JSON/NPZ files and analysis. Writes or replaces `results/AUDIT.json`. |

**The 16-bit refinement command does not replay the historical stop.** The original entry point, `& $py @flags "$d16/refine_counts.py"`, needs the 16-bit coarse `results/ANALYSIS.json` with 10 completed groups, its `MANIFEST.json`, matching sources and the same Torch version. Its default budget is one hour; it attempts counts 3000, 6000, 2500, 3500, 5000, 7000 in that order, writing to `results_refined/`. The original author's `results_refined/STOP_AFTER_GROUP` is not in this source archive. A fresh directory does not inherit that stop. Only with the whole coarse sweep and **all 12** planned refinement groups complete can `& $py @flags "$d16/audit_refinement.py"` produce `results_combined/` and `transition_refined.png`. The historical refinement finished only 5 groups; no such full merged output exists for it.

## Outputs and integrity

Each training output directory (e.g. `results/`) holds `MANIFEST.json`, `source/`, `VERIFICATION.json` and group progress in `RUN_STATUS.json`. Each `data{seed}_n{n}/` holds `result.json` (train/test trajectories), `data_ids.npz`, `forward_audit.npz` and `checkpoint.pt` (weights, optimizer and sampling state). After the driver calls the analysis, look in the training output directory for `ANALYSIS.json`, `summary.csv`, `paired_changes.csv`, `window_results.csv`, `trajectory.csv` and `sample_requirement_trend.png`. CPU result audits read saved records; the ZIP alone does not supply these inputs.

The ZIP contains only the 11 original `.py` files listed above: no logs, data, weights, result directories or third-party packages. The reference is `research/neural_k_framework/supplementary/updates_20261106/scaffolding/source_sha256.json` in the repository; compare ZIP member paths with the keys under `files`. All 11 member hashes match. Ten files were taken from formal-run snapshots and checked byte for byte against all 22 `source/*.py` copies across four run directories. The exception is the 14-bit `validate_results.py`: no formal-run copy exists, so the original experiment-directory file was included unchanged after checking its listed hash.

Run these read-only checks from the directory containing the ZIP. They need no GPU and do not execute the archived code. The ZIP is **44289 bytes**; its SHA-256 is `7a0474f6502a6cee70797a6f14771f321d3b454dc0bac143392576db39cbe91f`.

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath original_sources.zip
python -I -S -B -m zipfile -l original_sources.zip
python -I -S -B -m zipfile -t original_sources.zip
```
