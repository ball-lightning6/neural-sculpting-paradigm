from __future__ import annotations


EXPERIMENTS_EN: dict[int, dict[str, str]] = {
    1: {
        "title": "Static Hard Conditioning versus Optimizer-Induced Function Transport",
        "purpose": "Test whether repeated neural-network training merely samples the initialization function prior conditioned on training-label consistency.",
        "overview": """
This unit contains two direct tests in a fully enumerable 3-bit Boolean space. E01a compares the initialization prior, the hard-conditioned prior, and the complete-function distribution produced by Adam. E01b uses a much wider network and a single training example to make the transport effect visually and statistically unmistakable.

The complete output on all eight inputs is stored for every model, so no complexity proxy or test-accuracy summary is needed to identify the selected function.
""",
        "motivation": """
The function-prior program of [Dingle et al. (2018)](https://www.nature.com/articles/s41467-018-03101-6), [Valle-Pérez et al. (2019)](https://arxiv.org/abs/1805.08522), and [Mingard et al. (2021)](https://www.jmlr.org/papers/v22/20-676.html) supplies a strong static baseline: architecture induces a highly nonuniform function prior, and conditioning that prior on the observed labels can predict many first-order training statistics.

The falsifiable strong version says that training only removes incompatible functions. If so, a network already compatible with the training labels should not be systematically transported to another complete function, and post-fit training should not continue changing function odds.
""",
        "results": """
E01a sampled 1,048,576 untrained networks and trained 8,192 models per condition. At first fit, total-variation distances from the hard-conditioned prior were about 0.10–0.31; 100 post-fit steps raised them to about 0.54–0.64, while repeated-sampling noise was only about 0.01–0.04. Function rankings reversed in several conditions, and different sample orders left measurable finite-time path dependence.

E01b sampled 4,096 wide `3 -> 1024 x 3 -> 1` networks and trained 128 paired initializations per one-example condition. Roughly 72% of hard-compatible prior mass was nonconstant, so the hard-conditioned target-constant mass was only 27.5%–28.0%. Actual training made all 128/128 models the target constant after one optimizer step. Even the subset already correct on the training point moved unanimously.

**Conclusion.** Initialization simplicity bias is a real zeroth-order term, but optimization can transport mass between functions that already satisfy the training set. The result rejects exact hard-conditioning as a general training mechanism; it does not reject approximate Bayesian descriptions on richer continuous objects.
""",
    },
    2: {
        "title": "Rule-Bit Counterfactual Invariance",
        "purpose": "Determine whether a network automatically discards a control bit that is semantically irrelevant under the external rule but constant in training.",
        "overview": """
Inputs contain one rule bit plus a 30-bit cellular-automaton state. The output is one step of Rule 30 or Rule 110. Stage 1 trains on only one fixed rule-bit value. Stage 2 flips the bit and adds an increasing number of counterfactual examples. Binary and centered encodings, both fixed-bit directions, warm starts, and cold controls are compared.
""",
        "motivation": """
The external generator says the trained branch should remain correct when an irrelevant control bit is flipped, but the optimizer never receives that counterfactual constraint. A strong “training recovers the cleanest semantic rule” claim predicts spontaneous invariance; a training-set-centered view predicts that constant features may be used as bias or gating resources.
""",
        "results": """
All Stage-1 branches fit their observed data exactly, but counterfactual behavior depended strongly on encoding and fixed value. In binary encoding, training at bit 0 and testing at bit 1 often remained exact; training at bit 1 and deleting that contribution at test time could nearly destroy Rule 110. Centered encoding produced still larger reversals.

Small counterfactual datasets could write invariance into the objective, but sample requirements varied from one example to hundreds. Warm starts were not uniformly superior to cold training.

**Conclusion.** Human semantic irrelevance is not the same as geometric irrelevance in a parameterized network. The experiment supports protocol-relative complexity and data-defined invariance; it does not show that clean semantic programs are always selected automatically.
""",
    },
    3: {
        "title": "Prior-Consistent Function Dynamics and Width Intervention",
        "purpose": "Measure complete-function motion without defining complexity from the same outcomes that the theory is meant to explain.",
        "overview": """
E03a initializes ordinary models and models that already satisfy all labels of one-, two-, or four-example 3-bit datasets. Complete function IDs and logits are recorded after one update and through 5,000 post-fit steps. E03b repeats the analysis for tanh networks of widths 16, 32, 64, and 128.
""",
        "motivation": """
After E01 rejected passive hard conditioning, an immediate temptation was to label every destination “lower Kolmogorov complexity.” That would be circular. E03 deliberately measures no complexity proxy. It establishes the raw dynamical facts that any later complexity theory must explain.
""",
        "results": """
Among prior-consistent models, one update changed the complete function in 72.7%, 99.0%, and 97.9% of the one-, two-, and four-example conditions. Long after first fit, 32.8%–50.8% of ordinary models still changed function. Marginal histograms could look stable while paired models moved in opposite directions.

Width 16 through 128 preserved the qualitative post-fit transport, although priors and near-tied rankings changed. The tanh family and the much wider GELU–LayerNorm family also had systematically different priors and attractors.

**Conclusion.** Function transport is robust across the tested widths, but its destination is architecture-relative. The experiment does not identify transport with [Solomonoff induction](https://mlanthology.org/misc/1964/solomonoff1964misc-formal/) or machine-independent compression.
""",
    },
    4: {
        "title": "Post-Fit Drift in the Mingard 2025 Boolean Protocol",
        "purpose": "Test whether first zero training error is the endpoint of function selection in a protocol used to support a static function-prior explanation.",
        "overview": """
The experiment reproduces the central 7-bit Boolean setup of [Mingard et al. (2025)](https://www.nature.com/articles/s41467-024-54813-x): 128-state truth tables, a `7 -> 40 x 10 -> 1 tanh` network, three target-complexity groups, two initialization scales, and advSGD example selection. The only conceptual intervention is to continue training for 100 and 1,000 steps after first zero classification error while saving the complete 128-bit function.
""",
        "motivation": """
Static hard conditioning has no time coordinate after interpolation. If the first zero-error snapshot is already representative of the conditioned endpoint, complete-function changes afterward should be sampling-level noise. Systematic post-fit drift would instead show that continuous loss and optimizer transport remain active.
""",
        "results": """
Across 12,288 trajectories, 59.96%–97.66% of models changed complete function at initialization scale 1, and 99.66%–99.80% changed at scale 8. Mean Hamming distances reached roughly 1–4 bits at scale 1 and about 15 bits at scale 8. Permutation tests for per-input marginal drift were significant in all six conditions.

Direction was protocol-dependent: some groups improved over the full input space while others worsened. Lempel–Ziv trends were not uniform and are retained only as auxiliary statistics.

**Conclusion.** First interpolation is not the endpoint of function selection in this protocol. The claim is post-fit function drift, not monotone improvement under an artificial complexity proxy.
""",
    },
    5: {
        "title": "Feature Learning When a Fixed NTK Is Already Sufficient",
        "purpose": "Separate the existence of a fixed-kernel solution from whether a finite network still reorganizes its features during end-to-end training.",
        "overview": """
The task is one-step 30-bit Rule 110 with 8,000 training states and 20,000 unseen states. A bias-free `30 -> 1024 x 3 ReLU -> 30` network is trained with BCE and MSE. The experiment distinguishes the analytic infinite-width NTK, the finite-width empirical NTK at initialization, and the evolving multi-output block NTK during training.
""",
        "motivation": """
[Göring et al. (2025)](https://arxiv.org/abs/2507.19680) emphasize that feature-learning strength and feature usefulness are different. E05 asks the reverse sufficiency question: if the fixed [NTK of Jacot et al. (2018)](https://arxiv.org/abs/1806.07572) already solves the rule, does the finite network remain lazy?
""",
        "results": """
The analytic infinite-width NTK made zero errors over 600,000 unseen output bits. The width-1,024 empirical NTK was already near-perfect on three representative output bits. Nevertheless, all six finite-network trajectories reached complete zero error while changing hidden representations, ReLU gates, and empirical block NTKs.

Centered kernel alignment, following [Kornblith et al. (2019)](https://proceedings.mlr.press/v97/kornblith19a.html), was used together with gate flips and target alignment. Five of six trajectories crossed the preregistered post-fit structural-change thresholds; the sixth retained smaller but nonzero changes.

**Conclusion.** Kernel sufficiency does not force finite-network training to remain in the initialization-kernel mechanism. Feature learning can occur even when it is not necessary for generalization, so representation change cannot be equated with generalization or compression.
""",
    },
    6: {
        "title": "Loss-Conditioned Prior Annealing",
        "purpose": "Measure whether continuous raw training loss reweights complete functions beyond hard label consistency.",
        "overview": """
The experiment samples 4,194,304 `3 -> 1024 x 3 -> 1` GELU–LayerNorm networks, stores all eight logits, and reconstructs complete-function distributions under nested raw-BCE tails, normalized-loss tails, fixed-scale controls, and Gibbs-style annealing for one-, two-, and four-example datasets.
""",
        "motivation": """
If every hard-compatible function has the same loss-dependent volume factor, tightening loss cannot change function odds. Function-specific reweighting would make loss depth a static selection coordinate even before optimizer dynamics are considered.
""",
        "results": """
For one example, the target constant rose strongly in raw-loss tails, but much of that effect disappeared under scale control. For two examples, a symmetric threshold function rose from 1.57% of the hard-exact population to 23.0% in the deepest raw tail, 50.7% in the deepest normalized tail, and 44.1% in a fixed-scale tail.

The four-example condition rejected global monotonicity: the main SGD-attractor pair rose at intermediate depth and fell again in the deepest tail. Real SGD still concentrated far more strongly on that pair than the static sample.

**Conclusion.** Continuous loss produces architecture- and function-specific static reweighting, but no prespecified function is guaranteed to rise throughout all loss ranges, and the static distribution is not the optimizer distribution.
""",
    },
    7: {
        "title": "Function-ID Wandering and Matched-Loss Static Controls",
        "purpose": "Observe post-fit movement between neighboring complete functions and test whether matching a scalar loss makes static and optimizer-induced distributions equal.",
        "overview": """
A 4-bit, six-example dataset leaves 1,024 compatible hard extensions. 1,024 width-16 tanh networks are aligned at first complete fit and followed for 20,000 steps. A separate 4,194,304-network static sample is annealed to a normalized-BCE level matching late SGD.
""",
        "motivation": """
E06 showed static reweighting but not actual transport. E07 asks whether trajectories move through adjacent function cells and whether the late function distribution is determined by loss alone.
""",
        "results": """
1,019 of 1,024 trajectories changed complete function at least once after first fit, producing 4,379 transitions; 90.9% flipped only one unseen bit. Entropy fell from 5.217 bits to 1.291 bits and the top-function mass rose from 12.1% to 79.8%.

At matched normalized loss, static sampling already amplified the eventual SGD attractor from 0.94% to 23.9%, showing strong first-order geometric relevance. Yet the static and SGD distributions still had Jensen–Shannon divergence 0.378 bits and reversed important pairwise odds.

**Conclusion.** Transport occurs mainly across neighboring hard cells. Static loss geometry identifies candidate structure, while optimizer history determines inflow and residence. Equal scalar loss does not imply equal function distribution.
""",
    },
    8: {
        "title": "Shared Intermediate Computation under Finite Capacity",
        "purpose": "Test whether reusable intermediate computation causally helps a network reach deeper loss with less capacity and data.",
        "overview": """
Four matched task families are built from multi-step Rule 30/Rule 110 prefixes and two output summaries. In Shared conditions, both outputs can reuse the same expensive prefix; in Separate conditions, they require different prefixes. Joint networks can share hidden layers, while parameter-matched Split controls cannot.
""",
        "motivation": """
The phenomenological compression account requires a reason that reusable computation should be favored by the explicit loss objective. If sharing only changes label statistics, Shared tasks should also be easier in the Split control. A Joint-only advantage isolates finite-capacity computation reuse.
""",
        "results": """
The experiment scans prefix depth 0–3, widths 256/512/1024, and three seeds. Split controls show essentially zero Shared/Separate gap. At prefix depth 0, the Joint model has no sharing benefit. At depths 2 and 3, Shared wins all 18 paired comparisons.

At depth 2, the Separate-to-Shared best-loss ratio decreases from 8.06 to 3.58 as width increases. At depth 3 it decreases from about 399 to 6.07. Fixed-loss crossing counts show the same sample-efficiency direction.

**Conclusion.** When common intermediate computation is sufficiently expensive, reuse lets a finite network reach lower loss with less capacity and data. The experiment does not prove that the learned hidden variables exactly equal the researcher-named CA prefixes.
""",
    },
    9: {
        "title": "Long Training with 80% Hidden Label Noise",
        "purpose": "Separate early learning of shared digit structure from later memorization of fixed incorrect labels.",
        "overview": """
Forty thousand of fifty thousand MNIST training labels are permanently corrupted without marking which inputs are noisy. An overparameterized CNN and a smaller CNN, each with three paired seeds and clean-label controls, are trained for 300 epochs.
""",
        "motivation": """
A widely repeated result says networks can reach roughly 91% test accuracy even with 80% corrupted labels. The crucial question is whether this occurs after noise interpolation or in an early phase before the wrong labels are memorized.
""",
        "results": """
The overparameterized CNN reached 91.89% test accuracy at epoch 4 and the small CNN reached 93.89% at epoch 5. At those times, noisy-train accuracy was only about 19.5%; the models were still roughly 91%–93% aligned with hidden clean labels and about 1% aligned with corrupted labels on the noisy subset.

Continued noisy-loss reduction drove test accuracy down to roughly 23%–32%. No noisy model reached 99.99% interpolation within 300 epochs, so the experiment does not claim the final fully memorized endpoint.

**Conclusion.** The famous high accuracy is an early shared-structure phase, not evidence that a fully noise-memorizing network retains the rule. Overfitting is visible as loss pressure shifts from reusable structure to example-specific residuals.
""",
    },
    10: {
        "title": "Raw-BCE Competition between Preregistered Simple and Complex Function Pairs",
        "purpose": "Test lower-loss simplicity enrichment without defining complexity from the observed winner.",
        "overview": """
Simple targets are projections, AND, or OR. Complex targets match them on every training example and have the same full output balance, but introduce two unseen exceptions. Linear programming certifies that the simple member is a threshold function and the complex member is not. Both tanh and GELU–LayerNorm reference networks are tested at training-set sizes 10, 12, and 14.
""",
        "motivation": """
A broad external-complexity scan produced many apparent failures because datasets were underconstrained and several supposedly complex functions were easy threshold functions for the MLP. The pair design removes those ambiguities and uses the actual raw BCE optimized by training.
""",
        "results": """
After requiring reliable expected counts for both members in the chosen tail, all 12 of 12 comparisons increased the simple-to-complex odds and all were significant. Mean odds amplification was about 10.6-fold for GELU and 5.1-fold for tanh. Examples include about 9.96-fold for GELU AND at k=12 and 8.49-fold for GELU OR at k=10.

**Conclusion.** Lower raw BCE favors independently ordered simple compatible functions in this controlled family. The result is not a universal all-loss monotonicity theorem and does not identify the optimizer distribution with static odds.
""",
    },
    11: {
        "title": "Rule-30 Train/Validation Gradient Alignment across Dataset Size",
        "purpose": "Measure how more rule-consistent data changes the direction of loss descent at matched training-loss levels.",
        "overview": """
One-step 30-bit Rule 30 is trained with nested datasets from 256 to 4,096 examples using a `1024 x 3` GELU–LayerNorm MLP and five paired seeds. At fixed raw-BCE levels, the experiment computes the cosine alignment between the training gradient and an independent validation gradient.
""",
        "motivation": """
If increasing data makes the empirical objective a more faithful estimate of the full rule objective, the train and validation gradients should remain aligned to deeper loss. This directly tests the idea that additional examples extend the rule-aligned descent channel rather than merely changing the endpoint.
""",
        "results": """
Across BCE levels from 0.3 to 0.003, gradient alignment followed a strict dataset-size ordering in all five seeds. Larger datasets maintained positive rule-aligned descent at deeper loss, while small datasets diverged earlier.

**Conclusion.** More data changes the geometry of the descent direction throughout training and pushes the train/validation separation point deeper. This supports, but does not by itself prove, a monotone increase of target-function mass at every fixed raw-loss threshold.
""",
    },
    12: {
        "title": "AND Shortcut Static Geometry and a One-Example Balance Intervention",
        "purpose": "Determine whether the apparent failure of AND under lower loss is caused by loss geometry or by an imbalanced training set.",
        "overview": """
The task is 4-bit AND with ten training examples and a panel of compatible hard functions. Static prior samples are sorted by raw training BCE to track function odds over the full accessible loss range. A minimally modified dataset replaces one example to balance the four local AND patterns.
""",
        "motivation": """
The initial dataset appeared to contradict the claim that lower loss favors the obvious simple AND rule. Because the pattern counts were strongly imbalanced, the correct intervention was to change one example while preserving as much of the original dataset as possible.
""",
        "results": """
In the original data, a conditional shortcut gained mass at intermediate loss and AND did not dominate throughout. Candidate odds crossed. After the one-example balance intervention, deep-loss AND mass increased substantially and the corresponding long-run optimizer ensemble moved much closer to AND.

**Conclusion.** The training set itself defines which compatible explanation is economical. A simple generator is not automatically the preferred extension of an imbalanced finite sample. The experiment also rejects permanent function ordering across loss depth.
""",
    },
    13: {
        "title": "SGD Dynamics of the AND Shortcut and Balanced Intervention",
        "purpose": "Test whether static loss-stage reordering appears in actual optimizer trajectories and whether one sample causally changes the long-run function distribution.",
        "overview": """
Thousands of small tanh networks are trained on the original and minimally balanced ten-example AND datasets. Complete 16-bit hard functions are logged throughout long training, together with AND and shortcut masses, function entropy, agreement, and exception counts. Checkpointing and interrupt-safe saving support very long runs.
""",
        "motivation": """
Static odds alone do not prove that an optimizer follows the same ordering. The paired data intervention allows a causal test: if the training-set geometry matters, one example should alter both the static tail and the optimizer ensemble.
""",
        "results": """
The original dataset entered the shortcut family and reduced exceptions without concentrating on AND. The balanced dataset shifted dramatically: at step 32,000, about 64.3% of models were AND and the preregistered shortcuts had disappeared; the final audited distribution assigned roughly 64.9% to AND.

Static and optimizer distributions were related but not identical, and the optimizer did not simply sample one static slice.

**Conclusion.** A one-example intervention can change the long-run function ensemble, confirming that sample composition is causal. The result supports static geometry as an important influence while preserving an independent optimizer layer.
""",
    },
    14: {
        "title": "Weighted Rule-Bit Function Switching across Loss Scales",
        "purpose": "Construct an explicit dataset in which one fixed complete function is suppressed at shallow loss and rises only when rare constraints become unavoidable.",
        "overview": """
A rule bit selects Rule 110 or Rule 30, but training weights the majority and minority branches at ratios from 1:1 to 10,000:1. Each ratio uses 512 paired seeds with plain full-batch SGD, zero momentum, and zero weight decay. Complete local rule functions are recorded at matched weighted raw-BCE crossings.
""",
        "motivation": """
The experiment is designed to reject the strongest monotonic claim. At high loss, fitting the majority Rule-110 branch removes nearly all objective value; only at very low loss do the rare Rule-30 conflicts force the complete composite rule.
""",
        "results": """
The fixed `Rule110-both` hard function rose to peaks near 52%–55% and then fell as the minority constraints took over. Under the 10,000:1 ratio, total loss initially decreased while the Rule-30 branch loss was actively made worse, before the trajectory returned to the full composite map in the deep tail.

After rescaling by minority weight, extreme-ratio curves approximately collapsed, and takeover occurred near the theoretical loss floor of the majority-only solution.

**Conclusion.** One complete hard function can rise and fall as loss is tightened. Training follows the weighted objective rather than a researcher-designated “true composite rule.” The construction proves possible nonmonotonicity; it does not claim every natural dataset follows this path.
""",
    },
    15: {
        "title": "Complete Functions versus Marginal Concentration in Modular-97 Grokking",
        "purpose": "Distinguish a shared complete shortcut from a shared target skeleton with seed-specific residual errors before grokking.",
        "overview": """
Modular-97 division is trained with nested 60%, 70%, 80%, and 90% datasets, 32 seeds per fraction. Every model is evaluated on the full 9,312-input map. The experiment records exact target-function mass, complete-function entropy, pointwise agreement, coordinate-wise modal function, train/validation cross-entropy, and accuracy.
""",
        "motivation": """
Low agreement and zero exact-target seeds do not reveal whether all models share one complete shortcut or whether they share a mostly correct rule skeleton but make different residual mistakes. Exact function IDs and coordinate-wise marginals must be measured together.
""",
        "results": """
At 60% and 70%, no final seed exactly recovered the full target; at 80%, only 1/32 did. At 90%, target mass began appearing around step 12,500 and reached 18/32. Yet the coordinate-wise modal function became the complete target much earlier: step 18,000 at 80% and step 2,000 at 90%.

Even the 70% group ended with zero exact target seeds while its modal accuracy reached 95.41%.

**Conclusion.** Pre-grokking models need not share one complete shortcut. They can share a target-aligned skeleton while retaining different off-training residuals. Complete-function mass, entropy, agreement, and coordinate-wise modal behavior are distinct observables.
""",
    },
    16: {
        "title": "Parity Reachability, Half-Space Generalization, and Scaffold Recovery",
        "purpose": "Separate intrinsic low-loss parity preference from global optimization barriers and holdout-symmetry artifacts.",
        "overview": """
The unit combines a leave-one-out dimension scan, a 12-bit half-space and holdout-structure study, a 14-bit 50/50 train-test experiment, targeted error-reveal interventions, and a 16-bit scaffold–perturb–recovery test with wider networks.
""",
        "motivation": """
Parity is short in human algebra yet difficult for ordinary gradient descent. Poor leave-one-out prediction could mean the parity endpoint has tiny static support, or it could reflect special symmetry breaking and an entrance barrier that appears before endpoint preference can be measured.
""",
        "results": """
Leave-one-out behavior changed qualitatively with dimension and holdout structure. In a balanced 14-bit half-space, all 64 models escaped the `ln 2` plateau, fit 8,192 training states, and reached mean test accuracy about 99.935%, although none was exactly correct over all 16,384 states. Revealing an average of 6.22 targeted errors recovered the exact function in 23/64 models; equally many random correct points recovered 0/64.

For 16-bit parity, endpoint-only random starts recovered 0/8. A prefix scaffold produced 5/8 exact models, and after removing all auxiliary supervision, pure endpoint training reached 7/8. Strong perturbations destroyed every exact function, yet endpoint-only retraining recovered most models.

**Conclusion.** The dominant obstacle is global entrance, not absence of a locally supported endpoint. Holdout symmetry, information content, sample weight, and optimization accessibility must be separated.
""",
    },
    17: {
        "title": "Static Deep-Loss Mass versus Optimizer Transport",
        "purpose": "Measure how closely static constrained-SMC function distributions predict AdamW/SGD ensembles and identify conditions where optimizer history dominates.",
        "overview": """
The reference task is balanced AND with ten examples and a `4 -> 16 x 2 -> 1 tanh` network. The unit includes a 33.5-million-network brute-force prior, calibrated constrained SMC, AdamW/SGD/momentum crossings, a 32,768-seed AdamW retraining experiment, training from SMC deep-tail states, projection controls, and an 8-bit seed-versus-time mixing audit.
""",
        "motivation": """
Static volume can be an important first-order influence without being the optimizer distribution. A decisive test must compare both at matched loss, start optimization from real SMC states, and distinguish multi-seed ensembles from one trajectory sampled over time.
""",
        "results": """
At BCE 0.60, the static top-100 functions covered 98.7%–100% of optimizer mass, showing strong first-order relevance. SMC also matched brute-force prior distributions at accessible thresholds with very small JSD. In the deep tail, however, static mass concentrated on D440-like candidates while 32,768-seed AdamW concentrated roughly 76.5% on AND; the final JSD was about 0.760.

Starting 4,096 models from real SMC D440 states and training with AdamW did not produce AND, whereas projected controls retained D440. The same hard ID had different futures depending on continuous state and history. One long trajectory covered only a small local set of functions and did not reproduce the IID multi-seed distribution.

**Conclusion.** Static volume can strongly constrain candidates yet fail to determine optimizer endpoints. Function cells contain nonexchangeable continuous states, and a time series is not an IID Bayesian posterior sample.
""",
    },
    18: {
        "title": "Teacher-Free Consensus and Symbolic Readability",
        "purpose": "Test whether training sets that induce near-deterministic complete-function agreement also induce short human-readable symbolic rules.",
        "overview": """
The unit first scans 540 8-bit datasets with 64 seeds, intervenes on network width and additional samples, then scales to 8,192 random n=12 datasets. Candidate high-consensus endpoints are re-evaluated with up to 4,096 fresh seeds and audited using essential variables, ANF, decision trees, ROBDDs, and linear-threshold tests.
""",
        "motivation": """
Agreement does not require knowledge of a teacher, so it can detect function concentration on unknown datasets. The nontrivial question is whether the concentrated function is symbolically simple or a neural-only, high-complexity lookup map.
""",
        "results": """
The pilot found one teacher-free n=12 dataset with 61/64 seeds on the same complete function and collision 0.908. Its modal rule used four variables and admitted a short signed-threshold/2-CNF description.

The large scan produced 46 strict high-consensus endpoints after fresh-seed confirmation. A unified post-hoc audit found all 46 to be equal-magnitude signed threshold functions; 120 broader candidates were also linearly separable. No high-consensus random lookup-table counterexample was observed.

**Conclusion.** Under this 8-bit MLP protocol, strict high consensus systematically coincided with human-readable low symbolic complexity. The signed-threshold family was discovered post hoc and is not a preregistered theorem over architectures or all functions.
""",
    },
    19: {
        "title": "Full-Truth-Table Rule Volume",
        "purpose": "Measure target-specific low-loss volume directly under complete truth tables, avoiding ambiguity about which rule a partial dataset induces.",
        "overview": """
A `4 -> 16 x 2 -> 1 tanh` reference network is evaluated with constrained SMC on full 16-state targets: parity1–4, majority3, and a preregistered balanced random function. Eight independent replicas track volume over shared loss thresholds.
""",
        "motivation": """
Partial datasets can favor unexpected extensions. Complete targets provide an unambiguous object for testing whether loss-volume contraction differs across rules and whether human logical length agrees with the network’s representational language.
""",
        "results": """
Over the common reliable BCE range, parity1 had the largest volume and parity4 the smallest, with nonoverlapping replica ranges. Majority3 and the random target were nearly tied at shallow loss, but majority3 gained an advantage of roughly 5.6 thousand at BCE 0.40 and about 1.5×10^18 at BCE 0.30.

Parity2, although short in ordinary XOR notation, could have lower prior and target volume than the random function under this tanh architecture.

**Conclusion.** Different complete targets have genuinely different and scale-dependent volume curves. Human formula length is not a universal neural complexity ordering; architecture and encoding define the effective language.
""",
    },
    20: {
        "title": "Measure Closure between Full Rules, Fixed Datasets, and Margin Cores",
        "purpose": "Establish a strict bridge between full-target volume and candidate-function mass under a fixed partial training set.",
        "overview": """
Three linked tests are used: a 15-of-16 constant leave-one-out closure, a balanced-AND ten-example closure for four candidate functions, and a hard-to-margin bridge that fixes training loss and hard function while tightening margins on six held-out inputs.
""",
        "motivation": """
Full-rule self-loss volume and fixed-dataset candidate mass are related but distinct. Without a conditional-measure bridge, exponential full-rule differences could not legitimately explain competition inside a partial-dataset posterior.
""",
        "results": """
For constant leave-one-out, independent full-rule volume ratios and conditional crossing-event ratios closed over seven measurable thresholds with maximum log residual about 0.071. The four-candidate AND experiment closed with maximum residual about 0.049.

At the hard-cell boundary, the four branch masses reproduced the parent hard posterior with TV about 0.0086. Tightening only held-out margin then separated the cells by large continuous factors while hard IDs remained fixed.

**Conclusion.** Full-rule volume and fixed-dataset mass are not the same number, but they obey exact set inclusion and conditional-probability identities. Hard functions are coarse macrostates with substantial internal margin geometry.
""",
    },
    21: {
        "title": "Agreement Control and a Consensus Complexity Frontier",
        "purpose": "Turn agreement from a passive statistic into an intervention tool that can induce concentration or preserve competing extensions.",
        "overview": """
Starting from random balanced n=8 datasets, each round queries the unseen input with committee prediction closest to 50:50. Paired branches are retrained under labels zero and one. Anti-consensus chooses the lower-agreement branch, pro-consensus the higher-agreement branch, and a random-label control chooses randomly among fit branches. Longer anti prefixes are later switched to pro completion and audited with 512 fresh seeds.
""",
        "motivation": """
The design is related to [Query by Committee](https://doi.org/10.1145/130385.130417), but the committee is an empirical neural function ensemble. It tests whether disagreement is causally informative and whether longer resistance to concentration leads to more complex final functions.
""",
        "results": """
At n=24, median unseen agreement was about 0.970 for anti, 0.999 for pro, and 0.976 for random. Under a strict narrow-posterior criterion, 15/16 pro paths concentrated, while 0/16 anti and 0/16 random paths did. Stable pro endpoints were linear threshold functions.

Across 96 anti-prefix/pro-completion endpoints, 80 passed fresh narrowness. Longer anti prefixes required more pro examples and increased essential-variable, ANF, ROBDD, and quadratic-threshold complexity. All 80 narrow endpoints were representable by degree-at-most-two polynomial thresholds, whereas none of 64 random balanced functions were.

**Conclusion.** Labels chosen at high-disagreement inputs can causally control concentration and trace a symbolic complexity frontier. The procedure is greedy, agreement is not truth, and quadratic-PTF unification was discovered post hoc.
""",
    },
    22: {
        "title": "Per-Example Free Energy and an Information Invariant",
        "purpose": "Determine whether order-dependent per-example surprise sums to an order-independent complete-rule endpoint cost under one static measure.",
        "overview": """
The experiment stores logits from 2,097,152 prior 3-bit networks, evaluates all 6,561 partial label states, all 256 complete rules, and all 40,320 sample orders per rule. Hard-conditioned and multiple Gibbs-beta measures are computed together with Shapley allocations and microcanonical full-rule volumes.
""",
        "motivation": """
Earlier sequential-sample experiments found that each sample’s information contribution depended strongly on order. A coherent partition function predicts that increments are path dependent but their sum telescopes to one endpoint free-energy difference.
""",
        "results": """
Maximum path-invariance error was 2.84×10^-14 bits, stage-decomposition error 1.42×10^-14 bits, and Shapley-efficiency error 5.68×10^-14 bits. Rule 150/parity3 had hard endpoint cost about 29 bits and ranked hardest; under beta=1 it ranked second.

The result connects per-example surprise, endpoint difficulty, [prequential coding](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3b712de48137572f3849aabd5666a4e3-Abstract.html), and the [Levin–Tishby–Solla statistical-mechanics framework](https://mlanthology.org/colt/1989/levin1989colt-statistical/).

**Conclusion.** Generalized surprise is a difference of one static potential and totals are path independent. This algebraic closure does not imply that real SGD is a Gibbs/Bayesian sampler or that its trajectory conserves the same quantity.
""",
    },
    23: {
        "title": "Prospective Prediction from Full-Rule Volume to Data Transition",
        "purpose": "Prospectively test full-rule volume against n50/n90, then isolate why one cross-family ordering depends on fixed-dataset competitors and sampling.",
        "overview": """
Stage A uses full 256-state truth tables and constrained SMC for eight balanced 8-bit rules under a `8 -> 16 x 2 -> 1 tanh` network. A preregistered contraction score and rank are hashed before Stage B. Stage B trains 19 nonfull sample sizes × 8 rules × 64 random datasets, plus full-data qualification controls, with 24 paired seeds per dataset.

Three follow-ups close the sole stable parity2/MUX3 mismatch: a shared-parent five-target Gaussian deep-tail SMC; a 270,336-model uniform/cell/conflict sampling intervention; and a 48-condition fixed-$D$ Gaussian SMC under the same three sampling protocols.
""",
        "motivation": """
Defining low volume as difficulty is not predictive by itself. An independent sample-complexity measurement is required. The nested parity family is the confirmatory target; all eight rules form a cross-family stress test.
""",
        "results": """
The frozen parity ranking exactly predicted transition order: n50 values 24, 64, 96, 160 and n90 values 48, 80, 112, 160 for parity1–4. Transition intervals did not overlap; within-family Spearman correlation was 1.

Because every shared sample count already has 64 random datasets, continuous agreement can refine the coarse grid. A post-hoc target-aligned agreement=0.99 interpolation with 2,000 dataset-bootstrap replicates estimated parity2 at 59.98, MUX3 at 69.47, parity3 at 88.94, parity4 at 151.93, and random-balanced above 240. The parity2/MUX3 95% intervals were disjoint. This is a resolution diagnostic rather than a replacement for the preregistered decision.

Across all eight rules, Spearman correlations were 0.898 and 0.868. The balanced random target’s contraction accelerated with depth and crossed parity3/4, raising profile-to-transition correlation to about 0.97/0.95. MUX3 remained the one stable exception: at the deepest shared Gaussian threshold, its complete-target volume exceeded parity2 by roughly 42 decimal orders even though uniform-random training recovered parity2 earlier.

Sampling intervention resolved the mechanism. Uniform parity2/MUX3 gave `n50=64/80` and `n90=80/104`; strict eight-cell balancing gave `56/72` and `64/88`; increasing MUX selector-conflict examples to 75% reversed the order to `n50=72/56` and `n90=88/72`. Relative to uniform sampling, MUX3 moved 24/32 examples earlier while parity2 moved eight later at both thresholds. Two thousand paired-bootstrap replicates and every evaluated step from 500 through 40,000 preserved the direction.

Fixed-$D$ Gaussian SMC reproduced the reversal without an optimizer. At `n=32, epsilon=0.02`, uniform parity2/MUX3 exact-target masses were 0.266/0.000214, cell masses were 0.469/0.284, and conflict masses were 0.498/0.782. Uniform-MUX3 reached agreement 0.959 while almost never selecting the target; conflict-MUX3 made the target modal in 8/8 datasets. Uniform posterior accuracy was 0.993 on ordinary MUX cells but only 0.777 on selector-conflict cells; conflict enrichment raised the latter to 0.995.

Deep-tail lineages fell to roughly one or two per replica, so absolute decimals are coarse. The qualitative decision rests on direction across loss thresholds, all eight datasets for conflict-versus-uniform MUX3, replica support, the causal intervention, and the independent optimizer result.

**Conclusion.** Full-rule Neural K-profile is a strong first-order predictor, exact within the preregistered parity family, but `n50/n90` is a separate protocol-relative identification/recovery complexity. Cross-family prediction additionally requires the fixed-$D$ competitor denominator, sampling coverage, and optimizer transport. The MUX3 reversal is already present statically; AdamW is not required to create its qualitative direction.
""",
    },
    24: {
        "title": "Deep Neural K Crossing of Parity4 and a One-Point Exception",
        "purpose": "Test whether absolute target-volume rankings can reverse after both hard functions are already fixed and far inside the low-loss tail.",
        "overview": """
The first stage runs Gaussian-prior pCN SMC on 54 complete 4-bit targets under a 97-parameter tanh network: six structured rules, all parity4 one-point exceptions, balanced two-point exceptions, and preregistered random functions. The sole credible challenger, parity4 with input 0000 flipped, is then compared with parity4 in a 32,768-particle-per-replica shared-parent lockstep bridge.
""",
        "motivation": """
E23 showed scale-dependent slopes and rankings at shallow loss. The deep experiment asks whether the ordering stabilizes beyond the hard-exact boundary or whether continuous within-cell geometry can still reverse it.
""",
        "results": """
Parity4 had the smallest volume over all 25 thresholds in the 54-target panel. The 0000-flip target remained larger at BCE 0.01913 but contracted reliably faster, predicting a crossing near 0.0025.

The shared-parent run observed median crossing at approximately 0.002308. At the deepest threshold, median volume of the exception was about 2×10^7 times smaller. Seven of eight replicas had crossed in absolute volume and all eight retained positive relative contraction-rate differences. The stricter eight-of-eight five-window stop rule was not met.

The post-crossing ordering agrees with an earlier fixed-dataset observation: when `0000` was held out from parity4, the majority missing-label prediction favored parity4 rather than `flip0000`, although not with 100% probability. This is a local leave-one-out consistency check, not a 4-bit `n50/n90` measurement; a full random-dataset transition would require a separate E23-style scan.

**Conclusion.** Complete-function ranking can reverse inside a hard-exact tail through accumulated continuous contraction. Hard ID is not a sufficient complexity state, and the result should be reported with the 7/8 replica boundary intact.
""",
    },
    25: {
        "title": "MNIST Sample Complexity and Static Branch Prediction",
        "purpose": "Move the static-volume framework from enumerable Boolean maps to real images and test whether it predicts labels and validation-NLL turning points.",
        "overview": """
Stage 0 average-pools MNIST to 7×7 and trains a 1,633-parameter tanh MLP on 0/1 and 3/8 classification for sample sizes 4–512, four datasets per size, and eight seeds. Stage 1 freezes one n=4 dataset per task and nine train-BCE thresholds, then uses six-replica Gaussian-pCN SMC to compare the two label branches of each unseen image before scoring.
""",
        "motivation": """
The Boolean framework would remain a toy unless a fixed training set’s static low-loss mass could predict unseen natural inputs. The 0/1 versus 3/8 pair supplies a same-architecture, same-sample-count contrast with radically different data sufficiency.
""",
        "results": """
At n=4, median best-validation checkpoints reached 98.31% test accuracy for 0/1 and 69.66% for 3/8. Under a 95% validation threshold, 0/1 crossed at n=4 and 3/8 at n=512.

Static hard prediction rose from 92.97% to 96.35% for 0/1 and from 76.69% to about 78.13% for 3/8. Static soft NLL formed a U-shape whose minimum aligned within one preregistered grid cell of AdamW for 0/1 and directly covered the 3/8 training-loss turning interval. Prediction concentration continued after accuracy plateaued, while persistent mistakes became more confident.

The deep-loss mass disadvantage of 3/8 relative to 0/1 expanded from 0.56 to 55.43 decimal orders, aligned with the four-versus-512 sample requirement.

**Conclusion.** Static loss-conditioned volume has strong sample-level predictive power on real images and contains a geometric source of the validation-NLL U-shape. The tasks and loss range were calibration-selected, so a new digit pair is still required for a fully blind test.
""",
    },
    26: {
        "title": "Balanced Unlabeled MNIST Label Volume",
        "purpose": "Test whether low-loss parameter volume can blindly recover the natural MNIST 0/1 grouping among all 126 balanced labelings.",
        "overview": """
Each evaluation panel contains five zeros and five ones. One image is anchored to label zero, and all candidate assignments must remain 5:5 balanced. The 126 candidates share paired Gaussian-prior particles under a 49-to-32-to-1 tanh MLP. Calibration and the two blinded evaluation panels use disjoint images.
""",
        "motivation": """
Random-label experiments show that networks can fit both natural and arbitrary labels. E26 asks a different question: when fit is possible for every balanced assignment, which labeling retains the most low-loss parameter mass? All candidate volumes are written and hashed before the natural labels are revealed.
""",
        "results": """
At BCE 0.8 the natural split ranked last in both panels. At 0.6 it became top one in both; its mean normalized mass then rose to 0.9743 at 0.4 and 0.99972 at 0.3. Volume-weighted expected hidden-label accuracy reached 0.99994.

**Conclusion.** Under a fixed class count, 5:5 balance constraint, one anchor, preprocessing, architecture, and Gaussian reference measure, the natural digit split changes from least favored to overwhelmingly dominant as loss is tightened. This is not unconstrained unsupervised learning.
""",
    },
    27: {
        "title": "Agreement Before and During Grokking",
        "purpose": "Determine whether seeds already share one stable wrong function after hard fit but before rule generalization.",
        "overview": """
This supplemental analysis reuses the 32-seed Mod97 trajectories from E15. Pairwise agreement is recomputed only on unseen inputs. Hard fit requires every seed to reach at least 0.999 training accuracy; 0.90 mean unseen accuracy is used as a descriptive grokking milestone.
""",
        "motivation": """
Low validation accuracy before grokking could hide either one shared shortcut function or seed-specific residual functions that agree only on memorized training inputs. Full-domain agreement is biased upward after hard fit, so the held-out-only statistic is required.
""",
        "results": """
Across the 60%, 70%, 80%, and 90% training conditions, unseen agreement at hard fit was only 0.027–0.033. In the 90% condition, final unseen accuracy was 0.916 and agreement 0.839. Through the middle and late trajectory, observed agreement was almost completely explained by seeds independently recovering the same correct targets; remaining errors were largely seed-specific.

**Conclusion.** The experiment rejects a simple picture in which one complete wrong function is already shared before grokking. Agreement rises with recovery of the target rule rather than in a separate function-condensation jump.
""",
    },
    28: {
        "title": "Whole-Network HMC versus Adam on 50k MNIST",
        "purpose": "Test whether a finite-width static HMC ensemble can reach optimizer-level prediction quality on the full MNIST training set.",
        "overview": """
A 4,266-parameter small CNN is sampled with 16 full-batch HMC chains at beta equal to 50,000. Thirty retained snapshots yield 480 parameter samples. Plain Adam and Gaussian-prior MAP Adam use 32 matched initializations, batch size 256, and 50 epochs on the identical split without augmentation.
""",
        "motivation": """
Boolean and tiny-image results would remain scale-limited unless the complete static parameter ensemble could be constructed on a standard natural classification task and compared directly with practical optimization.
""",
        "results": """
HMC, plain Adam, and MAP Adam reached ensemble test accuracies of 99.03%, 98.95%, and 98.82%. HMC and plain Adam differed on only eight net test examples; exact McNemar p was 0.2005. HMC mean pointwise agreement was 0.9877, although all 480 retained samples represented different complete 10,000-image functions. Between-chain function distance remained above within-chain distance.

**Conclusion.** Whole-network HMC and Adam reach the same performance level at 50k MNIST, with a small nonsignificant HMC advantage under the frozen protocol. Strong prediction and pointwise concentration do not prove global MCMC mixing or equality of static and optimizer distributions.
""",
    },
    29: {
        "title": "Loss-Resolved Robustness to a Dead Input Bit",
        "purpose": "Test whether deep-loss margins can restore counterfactual stability while an unobserved input direction remains distributed as the prior.",
        "overview": """
Training covers all eight states of three active bits while a fourth dead bit is always zero. Counterfactual tests set the dead bit to 0.25, 0.5, 1, or 2 for five Boolean targets. Gaussian constrained SMC, 512-seed no-decay Adam, L2/MAP Adam, an MC prior-covariance kernel, and direct temperature-one posterior integration are compared.
""",
        "motivation": """
Exact Bayesian model averaging can be fragile when training features are linearly dependent because unidentifiable weights retain their prior. E29 asks whether loss depth supplies a second mechanism: active-path margins may grow until prior-scale dead-direction perturbations no longer flip hard predictions.
""",
        "results": """
The dead column retained variance near one and squared norm near sixteen under both SMC and no-decay Adam. Nevertheless, at the deepest measured loss the z=1 strict-correct mass reached 0.9959–1.0000. Against genuinely loss-matched no-decay Adam, mean absolute error was 0.0010 and the worst function error 0.0042, passing the preregistered 0.10/0.20 limits. L2 drove the dead column to zero and supplied a distinct MAP robustness mechanism. Low-ridge NNGP also passed z=1, while z=2 exposed larger static/optimizer gaps. Direct integration showed that the standard temperature-one posterior remained too shallow in this eight-example task.

**Conclusion.** Deep-loss static mass accurately predicts ordinary binary dead-bit behavior without contracting the dead direction. The result is protocol- and shift-scale-specific and does not overturn real-image covariate-shift failures of Bayesian model averaging.
""",
    },
    30: {
        "title": "Weight Decay Reshapes the Static Complete-Function Landscape",
        "purpose": "Clarify that grokking does not require weight decay and that explicit L2 helps by reshaping the same static function landscape rather than introducing a special mechanism.",
        "overview": """
The frozen task is balanced AND with 40 examples under an exact finite-width 8-to-16-to-16-to-1 tanh network and an iid Gaussian reference measure. Direct constrained SMC uses 16 replicas of 8,192 particles. Complete functions are recomputed on all 256 inputs. Three approximately raw-BCE-matched conditions use lambda 0, 5e-5, and 1e-4.
""",
        "motivation": """
The unregularized raw-loss landscape can already make a reusable rule the leading extension, so weight decay is not necessary for grokking. For explicit L2, the optimizer receives the gradient of one scalar objective, J = BCE + lambda ||theta||^2/2. The regularizer should therefore be understandable as a deformation of the same static landscape, not as a mysterious optimizer-specific mechanism.
""",
        "results": """
The raw-BCE means were 0.0025815, 0.0025105, and 0.0025898. Exact AND mass increased from 53.683% to 58.423% to 73.318%, while the number of observed complete functions fell from 4,190 to 1,246 to 86. The endpoint lambda=0 versus lambda=1e-4 difference was 19.635 percentage points, positive in all 16 replicas; the replica bootstrap 95% interval was 17.508--21.570 points.

All three conditions hard-fit the training set. The no-decay and middle-dose samplers passed all automated gates. The lambda=1e-4 raw archive exceeded an old float32 boundary check by 8.9e-10; all substantive diagnostics and saved samples passed, and the frozen code now audits relative tolerance plus float32 ULP error. A deeper lambda=1e-4 layer observed only AND but did not converge in absolute log volume, so it remains supplementary.

**Conclusion.** The no-decay landscape already gives AND majority static mass, while explicit L2 strengthens that preference at nearly matched training fit. Weight decay is neither required nor theoretically special; its present effect is explained by reshaping the same static function competition. This controlled result does not establish the same quantitative mechanism for decoupled AdamW or the original Mod97 Transformer.
""",
    },
}
