# Public Related-Work Addendum: Loss Conditions, Function Mass, and AIT Positioning

<!-- pair: p001 -->
[Simplified Chinese](README_zh.md) | [References and verification record](references.json). Paragraph identifiers, table-row identifiers, and citations correspond one-to-one across the two languages; R01-R17 identify the reference records.

<!-- pair: p002 -->
The user-specified integration baseline is `3834fb3` (2026-10-27). The update batch remains `20261106`; primary-source verification spans 2026-11-06 to 2026-11-07 (Asia/Shanghai). This is a bounded related-work addendum, not a systematic review, proof of priority, or new experimental report. It does not imply that the entire baseline has been re-audited.

## 1. Claims Requiring Correction

<!-- pair: t001 -->
| ID | Overstatement to avoid | Wording adopted here |
| --- | --- | --- |
| C01 | Previous work did not consider loss or fitting precision. | Existing work compares CE/MSE, loss strata, and behavioral precision cutoffs. Specify this study's constraints and measurement objects instead of presenting the question as previously unexplored. See R01-R05. |
| C02 | Every G&C counterexample merely failed to control loss. | Incorrect. Peleg discusses scale normalization and margins; Alexander's acceptance event itself requires low continuous MSE. Condition matching deserves examination but cannot dismiss counterexamples in advance. See R07-R08. |
| C03 | Static mass can explain generalization, so the optimizer is irrelevant. | A static reference can have explanatory and predictive value without equaling the training pushforward distribution. Preserve cases where high-entropy references outperform training, as well as width and dataset-size dependence. See R01, R03, R08-R09. |
| C04 | Neural function mass, contraction slopes, shortest descriptions, and Solomonoff probabilities are interchangeable. | These concern different objects and protocols. AIT methods can supply conditional connections, not automatic universal identities for this study. See R13-R17. |

## 2. Distinguish the Objects First

<!-- pair: p003 -->
Here, "mass" means probability mass, not predictive quality. Fix the architecture, encoding, decoding rule, probe set, and reference probability measure $\mu$ over parameters. Let $C_f$ contain parameters producing the same decoded function $f$. The following notation aligns the literature; it is not a new theoretical result.

<!-- pair: t002 -->
| ID | Object | Conditions and distinctions |
| --- | --- | --- |
| O01 | Function prior $\mu(C_f)$ | No training constraint; uniform weights do not imply uniform functions. |
| O02 | Hard-classification-compatible mass $\mu(C_f\cap H_D)$ | $H_D$ requires correct training labels, without fixing logits, margins, or continuous loss. |
| O03 | Continuously constrained mass $\mu(C_f\cap\{L_D\leq\epsilon\})$ | Only normalization produces a conditional function distribution; it depends on loss, scale, threshold, and measure. |
| O04 | Loss-stratified joint density of states | Joint mass of a loss bin and a test metric; thin strata, cumulative sublevel sets, means, and modes are not interchangeable. |
| O05 | Local behavioral volume around trained parameters | A loss- or output-KL-defined neighborhood, not all disconnected realizations of the same hard function. |
| O06 | Optimizer output distribution | Induced jointly by initialization, updates, randomness, and stopping; low loss alone does not determine it. |

<!-- pair: p004 -->
G&C, SMC, and Wang-Landau name methods, not complete target measures. A comparable protocol must specify at least the reference measure, acceptance event, normalization, training stopping rule, and evaluation quantity. Full-target constraints cover the entire target, whereas fixed-D constraints cover only the training set; these are also distinct.

## 3. The Loss-and-Volume Line

<!-- pair: t003 -->
| Reference | Object and verified conditions | Relation to this study and limits |
| --- | --- | --- |
| R01 [Mingard et al., JMLR 2021](https://www.jmlr.org/papers/v22/20-676.html) | Function probabilities on finite test probes; MSE uses a Gaussian-likelihood NNGP, while CE comparisons use an EP approximation with a 0-1 likelihood. Training usually stops at the first zero-classification-error epoch, with additional continued-training comparisons. | Direct precedent for correlations and residual differences between function posteriors and optimizer distributions, beyond mean accuracy. This does not establish equality between finite-network low-BCE sublevel sets and arbitrary training distributions. Relevant passages in Sections 2.3, 3.1, 5, and Appendix B were checked. |
| R02 [Mingard et al., Nature Communications 2025](https://arxiv.org/html/2304.06670v2) | Function priors, candidate counts within complexity levels, and data filtering jointly determine competition. Supplementary Section 8.6.2 acknowledges that a 0-1 posterior is not the exact training posterior; Fig. S10 compares CE, fixed-target-amplitude MSE, and initialization scales. | Neither candidate multiplicity nor loss sensitivity should be presented as originating here. A possible distinction is per-function measurement of continuous-constraint mass profiles and rank changes under a fixed finite network and measure; this does not establish priority. |
| R03 [Yang et al., 2503.13145v2; journal version 2026](https://arxiv.org/html/2503.13145v2) | Wang-Landau estimates joint density over training loss and test performance, with specified finite parameter ranges, comparing maximum-entropy performance with SGD/Adam at matched loss strata. Test metrics participate in landscape exploration. | Direct precedent for continuous-loss conditioning. Static high-entropy performance sometimes exceeds training performance, rather than equaling it. Retrospective landscape diagnosis is not a training algorithm without test-label access. A loss-stratum mode is not cumulative function mass. |
| R04 [Zhang et al., 2505.11411v5](https://arxiv.org/html/2505.11411v5) | Loss/test-accuracy entropy landscapes for modular-arithmetic Transformers; no entropy barrier was observed in the examined settings. Section 5's WanD explores using training loss and finds high-norm generalizing solutions. | Covers both static landscapes and dynamical intervention, not automatic dynamical irrelevance at low loss. WanD is a limited-task proof of concept. Appendix B.3's hybrid uses test accuracy $A_{test}$ to set its mixing weight, so it cannot be grouped with Section 5 as a test-feedback-free protocol. |
| R05 [Scherlis and Belrose, 2501.18812v2](https://arxiv.org/html/2501.18812v2) | Anchor-centered local loss/KL behavioral neighborhoods under Gaussian/uniform measures; Section 6.3 varies cutoffs and reports approximate local-volume power laws and numerical limitations. | Precision profiles and information interpretations of negative log-volume have precedents. Local KL volume and its slope are not global hard-function mass or this study's contraction coefficient. The estimator also has underestimation and sampling-error issues. |
| R06 [Buzaglo et al., 2402.06323v3](https://arxiv.org/html/2402.06323v3) | A single predictor is sampled from a prior conditioned on correct training classification. Lemma 3.2 gives a sufficient sample bound involving negative log total teacher-equivalent prior mass; network instances impose additional quantized or continuous-model conditions. | Supports generalization by random compatible predictors without selecting a shortest-description argmin. This conditional sufficient bound is not an equality for empirical grokking sample thresholds, nor an identification of negative log mass with a BCE contraction slope. |
| R07 [Peleg and Hein, 2407.03848v2](https://arxiv.org/html/2407.03848v2) | Low-sample binary classification: G&C accepts zero classification error; SGD uses CE. Sections 3.1-3.3 explicitly discuss ReLU scale freedom, Lipschitz-normalized loss, and margins; Section 4 compares initialization, width, and depth. | Not matching raw continuous-CE sublevel sets is not ignoring loss. Preserve results under their original protocol. Improving a prediction by changing this study's filtering event does not by itself invalidate their optimizer-difference findings. |
| R08 [Alexander et al., 2506.03931v2](https://arxiv.org/html/2506.03931v2) | In matrix sensing/factorization, Section 3.5's G&C explicitly accepts $L_{train}<\epsilon_{train}$ with MSE loss. Section 4.2 establishes cases inferior to GD under specified prior, activation, and width conditions. | Directly limits the claim that every counterexample omitted loss control. A continuous threshold does not guarantee static equivalence; conversely, this model-specific result is not a rejection across all architectures. GD initialization, step-size, and other conditions must remain explicit. |
| R09 [Pakman et al., 2606.31282v2](https://arxiv.org/html/2606.31282v2) | Uniform binary-weight prior; REWL estimates joint training/test-accuracy density on MNIST/Fashion-MNIST with 30/300/600 training examples, comparing the interpolation-stratum mode with gradient training. | The gradient advantage shrinks with more data in most examined conditions, making dataset size an important comparison axis. This is not continuous-CE matching or a universal resolution of Alexander's result and other continuous-network disagreements. |

## 4. Evidence Levels for Agreement and Accuracy

<!-- pair: p005 -->
Mean single-model accuracy, pointwise prediction frequencies, the joint prediction-vector distribution, and majority-vote accuracy are distinct quantities. Agreement on the first two does not imply close joint distributions or identical pointwise predictions from two trained models. The following works explain how different error locations can coexist with stable aggregate performance; they do not establish equivalence between static sampling and training.

<!-- pair: t004 -->
| Reference | Object and conditions | Relation to this study and limits |
| --- | --- | --- |
| R10 [Jordan, 2304.01910v4](https://arxiv.org/html/2304.01910v4) | Error structure across repeated runs with fixed training configurations. Section 3 distinguishes finite-test-set variance from test-distribution variance in settings including CIFAR-10/ImageNet, examining approximately independent errors after convergence and class-wise calibration. | A reference for accuracy variation and pointwise error probabilities; short training does not satisfy the same independence approximation. Neither a universal independence law nor a license to infer identical function distributions from stable accuracy. |
| R11 [Jiang et al., 2106.13799v2](https://arxiv.org/html/2106.13799v2) | Independent training runs on the same dataset. Section 4 connects expected disagreement and expected test error under ensemble calibration on the test distribution. | Disagreement is a conditional generalization diagnostic. The expectation identity is not exact equality for every model pair or a formula for majority-vote accuracy. It does not automatically establish calibration for arbitrary G&C distributions. |

## 5. The Actual Anti-Grokking Intervention

<!-- pair: p006 -->
R12 [Tian et al., 2610.00620v1](https://arxiv.org/html/2610.00620v1), Section 3.2/Fig. 4, keeps modular addition (modulus 97), approximately half the examples for training, the model, and training hyperparameters unchanged. Only the partition changes: $(a,b)$ and $(b,a)$ are placed in the same partition. Reported anti-grokking means persistently poor validation performance without delayed recovery, not earlier generalization or separating reversed pairs across training and validation.

<!-- pair: p007 -->
Appendices H/H.1 reduce the same-partition ratio; their specific record shows no completed grokking at 70% and clear recovery at 60%, with symmetry-orbit coverage used to compare information coverage. Our comparative inference is that partition structure must be controlled even at fixed sample count; this is not a new experiment. Phase-cell counts indirectly probe regional alignment, not global volume. No recovery within a finite budget does not prove that recovery can never occur.

## 6. Concise AIT Method Positioning

<!-- pair: t005 -->
| Reference | Actual object or method | Useful connection and unsupported identification |
| --- | --- | --- |
| R13 [Shaw et al., 2509.22445v3](https://arxiv.org/html/2509.22445v3) | Asymptotically optimal two-part/variational description-length objectives for Transformers; minimum codelength under increasing resources and a differentiable Gaussian-mixture-prior objective. | A conditional coding bridge. Section 4.2 specifies finite-resource, additive-constant, and minimization limits; Section 6.1 shows that random-initialization optimization misses low-codelength manually initialized solutions. This neither makes ordinary initialization probabilities universal nor guarantees SGD reaches the minimum. |
| R14 [Bakhtiarifard et al., QuBD, 2605.15551v1](https://arxiv.org/html/2605.15551v1) | Quantize weights to a finite alphabet, aggregate CTM/BDM estimates across bit-planes, and track weight structure during training. | A complexity proxy for quantized weights, affected by quantization, partitioning, finite CTM tables, and shared information across bit-planes. It measures neither complete-function algorithmic complexity nor global parameter mass. |
| R15 [Li and Teng, Approximate Solomonoff, 2610.29289v1](https://arxiv.org/html/2610.29289v1) | Impose a KL approximation condition between an environment mixture $\xi$ and predictor $\rho$, then analyze cumulative prediction error, distinguishing prior averages from low-prior-mass environments. | Approximation error can be amplified substantially in rare environments. The condition concerns sequence-prediction distributions, not neural parameter volume. Small ordinary training loss does not establish the required population KL condition or a universal prior. |
| R16 [DeMoss et al., 2412.09810v2](https://arxiv.org/html/2412.09810v2) | Coarse-grain weights subject to allowed training-loss distortion, then use compression to estimate rate-distortion curves; study grokking complexity dynamics and spectral-entropy regularization. | Direct methodological precedent connecting loss tolerance and compression. Practical compressed codelength is not exact Kolmogorov complexity; compression changes along a training trajectory are not changes in the aggregate mass of compatible functions. |
| R17 [Liu, Zhong, and Tegmark, LMN, 2310.05918v1](https://arxiv.org/html/2310.05918v1) | Define Linear Mapping Number from an input-interpolation linear-connectivity matrix and its spectrum, estimating the effective count of local linear computations and tracking grokking on algorithmic tasks. | A comparison for nonlinear computational structure and compression, dependent on probes and estimation protocol. Its candidacy as a neural Kolmogorov complexity is the authors' positioning, not a proven universal equivalence or a parameter-volume theorem. |

<!-- pair: p008 -->
Minimum representation cost, compressed codelength of particular trained weights, and probability mass over all parameter realizations of one decoded function must therefore be accounted for separately. Correlation under some conditions does not make them interchangeable without a bridge. Lowering BCE, changing a program-length penalty, and increasing representation precision are likewise different operations. No new AIT theorem is claimed here, and strong wording in recent preprints does not replace independent verification.

## 7. Bounded Positioning of This Study

<!-- pair: p009 -->
Following existing function-prior, conditional-sampling, and compression approaches, this study examines how continuous training constraints redistribute function probability mass in fixed finite networks, and which training outcomes that redistribution predicts versus which differences still require dynamics. Concrete comparison points are the full-target/fixed-D distinction, loss-resolved profiles, amplitude alignment, reference-measure interventions, and training comparisons under explicit protocols. These identify research objects and candidate distinctions, not a priority list established by this search.

<!-- pair: p010 -->
This addendum supplies no new quantitative conclusions about baseline experiments. Each empirical claim still requires support from its existing experimental protocol and discrepant cases; related literature on volume or compression cannot substitute for this project's own validation.

## 8. Actual Verification Boundary

<!-- pair: p011 -->
17 works are included. Authors, titles, version dates, and the definitions, methods, result statements, or captions used in the tables were checked. `references.json` separately records what was and was not examined. There was no line-by-line audit of every proof or independent reconstruction of plotted data. Journal publication dates and arXiv revision dates are recorded separately.

<!-- pair: p012 -->
Public evidence links point only to author papers, arXiv, or official journal sources. Direct access to Mingard 2025's journal page initially redirected; publication metadata was checked through the publisher's search result, while supplementary content was checked in arXiv v2. Chiang's OpenReview pages encountered browser verification and supply no additional independent evidence here.

<!-- pair: p013 -->
The cited papers' code and experiments were not independently run or replicated, and no new sampling was performed. Citation counts were neither collected nor inferred.
