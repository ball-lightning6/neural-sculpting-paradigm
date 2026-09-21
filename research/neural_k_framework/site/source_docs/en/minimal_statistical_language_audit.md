# Itemized Audit and Corrections for the Minimal Statistical Language

Date: 2026-09-20. Corresponds to the revised minimal definition.

## 1. Substantive Problems Found in the Previous Draft

The previous draft still substituted one object for another. It was not enough to add subscripts and distinguish definitions from empirical claims. The most serious issue was this: after conditioning a joint distribution on an old average-loss event `E`, the resulting calculation only describes `E ∩ A_y^+`, but it was named as the new distribution after adding a sample. The new average-loss event `A_y^+` can absorb parameters from outside `E`, which the previous draft omitted.

A second gap concerned the level at which information is measured. KL between parameter distributions and KL between hard-function distributions are generally different. A continuous-loss restriction may only select implementations inside one hard-function cell while leaving the function distribution unchanged. Writing only “surprise equals KL” does not identify which KL is meant.

This audit reread the earlier discussion of non-nested average-loss events, hard/soft prediction, surprise, KL, shells, and full-target objects, and compared it with the old tutorial and related derivations. The changes are recorded below.

## 2. What Was Changed

| Issue | Risk in the old wording | Current correction |
|---|---|---|
| Loss of one hard function | A function macrostate was treated as having one deterministic energy | Sections 1–2 of the main text retain continuous outputs inside each cell; loss is the parameter function `L_D(theta)`, described by a branch loss measure `xi_f` |
| New average-loss update | Only parameters in the old set that pass the new sample were counted | Section 4.3 defines `in_y(f)` and `out_y(f)` and reconstructs the mass of the entire new event |
| Retaining the old requirement versus resampling | Both operations shared one `Q_soft` name | Sections 4.2–4.3 use separate `Q_cap_{E,y}` and `Q_new_y` events |
| Label prediction | Low-loss mass after adding a candidate label replaced hard voting under the old distribution | Sections 4.1–4.3 separately define mutually exclusive hard branches, soft acceptance rates, and independent new-event mass |
| Candidate-event normalization | Candidate label events were assumed to partition the parent set | Section 4.3 allows overlap and omitted regions; a new/old mass ratio can exceed one and a normalized score is not automatically a hard prediction |
| Query input outside the old function table | An unseen label was inferred from old-function marginals alone | Sections 1 and 4.1 extend the input table to `U'` and retain the required refinement information |
| Parameter KL versus function KL | One “distribution update” mixed the two | Section 5 gives the KL chain rule and the within-cell conditional KL term |
| Expected surprise | The distribution used for the label expectation was omitted | Section 5.1: model-distributed labels give model Shannon entropy; externally distributed labels give cross-entropy |
| Chain rule for information | Each new sample redefined average loss while a nested multiplication rule was still applied | Section 5.1 restricts the chain rule to genuine event intersections with a fixed final intersection |
| Minimum KL | A measure-relative information cost was described as data or compute required | Section 5.2 keeps the exact projection decomposition and states that it measures deviation from the reference measure |
| Full-target versus fixed-D | Full-target low-loss mass was treated as the candidate hard-function mass | Sections 6.1–6.2 add the continuous precision pass rate inside the target cell |
| Mean loss versus total loss | One threshold symbol was reused across different sample counts | Sections 2 and 6.2 convert total budget `tau` to mean threshold `tau/n` explicitly |
| Cumulative region versus shell | The distinction was stated without a precise bridge | Section 2 first defines the loss measure; when a density exists, a local contraction index gives the exact reweighting |
| Exact loss density | Every loss distribution was assumed to have an ordinary density | Section 2 allows atoms and singular components and uses `rho_f` only under absolute continuity |
| Neural K versus remaining cost | `-log Omega_f` and `-log Q(f)` were mixed | Section 3 separates them by the negative log mass of the parent event |
| Neural K naming | The paper's full-target profile and fixed-D cell profile shared an unqualified symbol | The paper object is `K^(N,full)` and the fixed-D object is `K^(N,FD)`; the generic name is used only after the protocol is declared |
| Complexity distribution | Function-count histograms, probability-weighted distributions, and current K histograms were mixed | Section 3 first fixes an external cost scale `c(f)`, then pushes the current function distribution forward through it |
| Elimination versus complex-function contraction | Set shrinkage was treated as implying lower mean complexity | Sections 3.1 and 7 use the covariance identity; the overall distribution still has to be measured |
| Feasibility and positive mass | An infimum was used to assert that a boundary was attained; zero mass was treated as no solution | Section 7 distinguishes an attained infimum, a nonempty feasible set, and a positive-mass feasible set |
| Agreement and concentration | A narrow cost distribution was treated as one dominant function | Section 8 separates full collision, pointwise agreement, modal label share, and cost variance |
| Dataset randomness | Repeated datasets and a fixed dataset were mixed | Section 8 separates average conditional collision from collision under a mixture distribution |
| Function families and probability budgets | Overlapping function-property events were added directly | Sections 1 and 8 require a fixed partition for aggregation; budgets do not apply directly to overlapping full-target events |
| SGD and the reference measure | A parameter density ratio was forced onto every training distribution | Section 9 keeps the measure-intersection definition generally and uses density ratios only under absolute continuity |
| Geometric volume and Gaussian mass | Output translation was declared to preserve fixed-Gaussian mass | Section 10 distinguishes same-measure translation invariance from synchronized pushforward of the measure |
| Moving-target volume ratios | A non-nested MSE-neighborhood ratio was called a survival probability | Section 10 calls it a geometric mass change; survival has that meaning only for nested protocols |
| Linear contraction | Translation symmetry was used to derive an exponential law directly | Section 10 requires an independent mixed-difference bound and a positive-slope condition |
| AIT and MDL | Coding-theorem constants were treated as a loss-controlled mixture correction | Section 11 fixes the optimal prefix machine and object coding; the neural mixture correction is a separately defined entropy quantity |
| Old statistical terminology | “Relative entropy” and “free energy” had no specified object | The terminology table maps each name to a precise object, without deriving mechanisms from labels |

The main object map is:

| Old name | Precise object |
|---|---|
| Microstate | Parameter `theta`; a continuous joint output may also be used after declaring the pushforward measure |
| Function macrostate | A cell `C_f` or a predeclared function family under a fixed input domain and readout |
| Energy | `L_D(theta)` or `T_D(theta)` on specified data, with normalization stated; generally not `L_D(f)` |
| Density of states | A branch loss measure `xi_f`; write `rho_f` only when a density exists |
| Hard free energy | If retained, only `J(E)=-log mu(E)` |
| Branch “entropy” `log Omega_f` | Log mass of one event; not Shannon entropy and not relative KL entropy |
| Predictive entropy | Shannon entropy of a specified label variable under a specified distribution |
| Function Renyi entropy | `-log2 sum_f Q(f)^2` for a specified complete-function distribution |

## 3. Three Finite Examples That Catch Errors Directly

### 3.1 A New/Old Mass Ratio Greater Than One

Take three parameter points with reference masses `0.4, 0.2, 0.4`. The old training set has one sample; old losses are `0.10, 0.19, 0.30`, with threshold `0.20`.

After adding a candidate label, the single-example losses are `0.10, 2.00, 0.01`. The new mean losses are `0.10, 1.095, 0.155`. Thus:

- the old event keeps the first two points, with mass `0.6`;
- the new event keeps the first and third points, with mass `0.8`;
- filtering only inside the old event keeps only the first, with mass `0.4`.

The new/old mass ratio is `4/3`, so it cannot be called a prediction probability. Assigning different complete-function IDs to the points also makes the old-event filter and the new-event resampling induce different function distributions.

### 3.2 Internal Filtering inside One Hard Function

Two equal-mass parameter points implement the same hard function. Both have old loss `0.19`; at the query point their BCE values for the same correct label are `0.10` and `0.30`, and both predict correctly. After adding the query sample, their mean losses are `0.145` and `0.245`.

At threshold `0.20`, only the first passes. The hard prediction probability is 1, the continuous-loss acceptance rate is `1/2`, the parameter KL is 1 bit, and the function KL is 0. This single example rules out both substitutions: hard prediction rate equals continuous acceptance, and parameter KL equals function KL.

### 3.3 The Internal Factor Missing from Full-Target Mass

Inside a fixed total-loss event, suppose the target hard-function mass is `1/2`. Requiring an additional held-out continuous-loss condition retains only `2/5` inside that cell.

The full-target mass is `1/5=(1/2)(2/5)`. It cannot be used directly as the target hard-function mass under the parent event.

## 4. What the Verification Checked

Script: the finite-probability check in `checks/check_minimal_statistical_language.py`.

The round ran on CPU and involved no training or sampling. Ten groups of checks passed. Two hundred random finite probability models checked conditional KL, coarse-grained KL chains, minimum-KL decomposition, retention updates, fixed-cost covariance, collision bounds, and probability budgets. Set masses and most algebra were checked with exact rationals; the largest floating-point residual involving logarithms was about `8.88e-16`.

It also checked entering and leaving the same mean-threshold event, within-cell loss differences, overlapping candidate events, nested total budgets, target probability rising while collision falls, zero cost variance with low collision, shell versus cumulative odds, and the full-target precision factor and discrete-curvature error bound.

These are checks of formulas and counterexamples. General claims still rely on the definitions, conditions, and derivations in the main text; finite verification does not prove that a general neural network satisfies exponential contraction.

## 5. Additional Corrections to Two Related Derivations

The joint-output rate derivation now makes the pushed-forward measures explicit, separates Gaussian point density from the small-box limit, and states differentiability and differentiation-under-the-integral conditions. The large-deviation part separates output scale from rate scale and does not automatically use the center value for a fixed-ratio box.

The finite-window contraction derivation now defines the four corner errors `e_ij`; if each `|e_ij|<=tau`, the general bound is `eta_int+4 tau`, not the earlier unsupported `2 tau`. It also records the time-state qualification and links to the later measurements.

These derivations remain conditional models. They do not prove that complex rules necessarily have larger contraction coefficients.

## 6. Audit Scope

This revision covers all chapters of the minimal theory and the two directly cited contraction derivations. The old statistical tutorial remains as historical context, with corrections and links to the new definitions at its entry points. Old experiment code and results were not rerun or rewritten because of the definition changes.

It does not claim to have audited every historical document in the repository, and it is not certified by a proof assistant. Future uploads should use the current minimal definitions and audit table; historical documents that retain old names must be checked against their concrete events and distributions rather than joined by terminology alone.
