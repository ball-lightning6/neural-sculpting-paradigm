# Neural Network Generalization and Algorithmic Information Theory

Revised 2026-11-10: [how one more observed input refines a hard function](#hard-function-resolution), and [how cooling on a fixed training set affects the Levin analogy](#levin-temperature). This revision clarifies the interpretation; it reports no new experiment. See the [update overview](research_updates_20261106_en.md) for earlier evidence and negative results, and the [conceptual clarification](interpretation_update_20261106_en.md) for the distinction between representation, measure, and training dynamics.

**Why neural networks work can first be understood through a few factors: they can represent a rich set of functions, their parameterizations do not assign equal implementation mass to those functions, and data and output requirements further filter implementations. Once reusable explanations gain an advantage, they can support many predictions that were never supplied individually.**

Algorithmic information theory (AIT) connects description length, generation probability, and shared information. This research measures these relationships in specific neural systems: which functions have greater implementation mass, how continuous-output requirements change their relative advantage, and how far these static distributions can predict actual training.

## 1 Three Simple Factors

1. **Rich representational capacity.** A network can accommodate many candidate mappings, and a finite training set usually permits many complete extensions. Universal approximation ideas connect to this level, but the existence of an implementation does not mean that training can easily find it.
2. **A prior simplicity bias.** Architecture, encoding, parameterization, and the reference measure jointly determine the implementation mass of different functions. Structurally economical rules that fit the architecture are often easier to obtain from random parameters; this does not mean that every rule humans regard as simple is cheap.
3. **The bias can be adjusted.** In some tested protocols, increasing joint-output precision or changing amplitude requirements causes different candidates to lose mass at different rates, strengthening the preference for more economical explanations. BCE is a major studied route, not the only possible form.

The first two factors, combined with informative data, already provide a basic account of induction. The third is an additional mechanism, not a necessary condition for all generalization. The finite target amplitude reached by sufficiently fitting raw-output MSE can be studied as a fixed endpoint; continuing to drive BCE toward zero can keep increasing the correctly signed margins. [E31](experiments/experiment_31_loss_amplitude_grokking/README.md) shows that finite-amplitude MSE can also generalize effectively, and that target center, amplitude, and alignment all matter.

Broad effectiveness therefore need not depend on one engineering detail having a uniquely indispensable role. The more general research question is whether different systems have rich representations, suitable nonuniform implementation mass, and the ability to use data to select structure. Practical efficiency remains constrained by architecture and optimization reachability.

The update on 2026-11-07 connects the early encoding experiments: all four addition symbol/position-shuffling conditions in the earlier paper reached low evaluation BCE, and historical source code also specifies real-valued input and output encodings. These make "learning structure" operational as learning task relations under fixed re-encoding. Encoding conversions can themselves be reused, but costs, priors and internal algorithms need not remain unchanged. Recorded results and source-code evidence are distinguished in the [encoding-decoding experiment connections](interpretation_update_20261106_en.md).

## 2 The Training Set Is the Object of Study

For a fixed training set, the object is the parameter region satisfying its requirements and the mass occupied there by different complete functions. Higher-mass candidates occur more often; this does not mean that every training run deterministically selects the largest-mass function. Correspondence between static SMC distributions and SGD, Adam, or STP outcomes is an empirical relationship to be tested.

**The simplest rule that explains the training set need not be the true rule.** Fix a way to describe rules and ask: given these inputs, what is the shortest rule that returns all the observed labels? Suppose the true rule is $y=a\land b$, but every training input has $b=1$. Then $y=a$ fits every label. A stronger simplicity bias cannot supply the missing evidence; one new example with $a=1,b=0,y=0$ rules out that explanation.

The prior and the data play different roles: the prior sets initial preferences among compatible explanations, while observations exclude incompatible extensions. A simple target with a substantial initial advantage may dominate after a small amount of informative data; a more complex target usually needs more samples to exclude other explanations with substantial mass and then reduce the total remaining competitor mass. This connects to the candidate-prediction picture of [Solomonoff induction](https://raysolomonoff.com/publications/solo1.pdf).

**Adding samples and increasing precision on fixed data should be named separately.** Changing the training set from D to D' expands data constraints and identifies extensions; a candidate legitimate under the old D should not retroactively be called a shortcut under that D merely because it later disagrees with a new label.

With the same D fixed, tighter output requirements can have two effects: shortcuts that only fit coarsely cease to suffice, and continuous implementation mass is reallocated among candidates that remain feasible. These connect respectively to adequately explaining the data and choosing economical descriptions in MDL. They motivate possible narrowing of the complexity distribution from both sides, but do not guarantee that variance or the winning function changes monotonically with precision. [MDL background](https://arxiv.org/abs/math/0406077)

The joint output distribution is the underlying object throughout; loss and amplitude conditions select regions within it. The same hard function can have different continuous implementations, and separate full-target volumes for different rules cannot simply be summed as the fixed-D conditional distribution of one training set. See the [strict probability language](translations_en/theory/probability_language.md) and [E20](experiments/experiment_20_measure_closure/RESULTS_AND_CONCLUSION.md) for precise definitions.

## 3 Why Implementation Mass Can Relate to Complexity

Levin's coding theorem starts with one discrete object $x$. On a fixed universal prefix machine, the length $K_U(x)$ of its shortest program differs from the negative log of $m_U(x)$, the total probability of all programs producing it, by a constant independent of $x$. The short description thus tells us the scale of the total probability. The theorem does not say that one shortest program carries a majority, nor does it prove the same constant bound for a neural network's conditional distribution. [Coding theorem, Eq. (2)](https://ir.cwi.nl/pub/26818/1-s2.0-S0304397513005124-main.pdf)

There are two different sums for a network. Specify a complete function, and we can add up the mass of all weights that implement it. Specify only the training set, and we must also add over the functions that fit it but may disagree on unseen inputs. A simple function can be more common than any one competitor and still be less common than a whole group of slightly more complex competitors.

### How does one more observed input refine a hard function? {#hard-function-resolution}

**A hard-function table records discrete predictions on specified inputs. It is not the smallest unit of network behavior.** If two models give identical top-1 labels on all [50,000 ImageNet validation images](https://www.image-net.org/download), they have the same table on those images. One more image can distinguish them even if nobody supplies its correct label. Hold the sampling distribution over weights fixed: the mass of the old table equals the sum of the masses of its refinements, one for each answer to the new image. If all models give the same answer, this observation does not split the group. What changed is the resolution of observation. Adding the new label as a training requirement is a different operation: it changes the training constraint.

A 4-bit binary task gives a useful contrast: once we have queried all 16 inputs, repeating them cannot refine the truth table on that domain. Measuring output amplitudes can still distinguish implementations with the same table. On a large task, observing more inputs can make the mass of an individual table tiny even if models agree on almost all natural inputs. We can instead specify an input distribution and a disagreement tolerance in advance, then add the masses of candidates whose predictions stay that close to the same rule. For continuous outputs, too, we need the probability of an observable region; a density at a point cannot be substituted into the coding theorem.

### Which group of functions dominates after cooling? {#levin-temperature}

Fix the training set and reference measure. Independently specify a group $S$ of simplest or near-simplest compatible explanations, without selecting it by its eventual mass. Let $t$ index the training constraint, $V_D(t)$ be the total mass satisfying it, and $V_{D,S}(t)$ the part belonging to the group. When both are positive, the cost in bits of keeping that group instead of all candidates is:

$$
\Delta_D(S,t)=\log_2\frac{V_D(t)}{V_{D,S}(t)}=-\log_2 Q_{D,t}(S).
$$

Here $Q_{D,t}(S)$ is the group's share of the constrained region. Extensions resembling random lookup tables are numerous; we can neglect them as a group only if the penalty per extension outweighs their advantage in numbers. For illustration, suppose simple, intermediate, and complex groups start with 10, 30, and 60 units of mass. A tighter BCE threshold leaves 8, 15, and 6. Even the simple group has lost absolute mass, but its share has risen from 10% to about 28%, and the majority has shifted from the complex group to the intermediate one. **Cooling shifts the distribution toward simpler explanations; it need not select the same simplest function in every run.**

The empirical prediction is conditional: if the more complex groups keep shrinking faster than $S$, stronger bias gives $S$ a larger share and makes its mass a better approximation to the total. Tightening BCE is one route to test. Where large norms favor complex rules, penalizing large norms may shift mass in the same direction, but the interventions are not equivalent. The [Rule30 norm comparison](width_double_descent_en.md#rule30-norm-data-gap) measures changes in trained predictions, not this mass share directly. Whether the explanations favored after cooling generalize still depends on which competing rules the training data ruled out.

### How contraction rates relate to short descriptions

For an effectively encodable discrete neural generation protocol, probability coding can yield a short universal description. The reverse direction additionally requires showing that short descriptions admit inexpensive implementations with sufficient parameter mass in the specified network. Universal approximation does not replace inexpensive decoding, and a compiler's fixed program length does not imply fixed computation in a finite network. A total-mass budget limits the individual mass available to numerous complex objects, but a uniform lookup model also satisfies that budget; it alone does not establish high probability for simple functions.

Under one measurement protocol, let $V_f(t)$ be implementation mass along precision coordinate $t$, $K_f$ its negative logarithm, and $\kappa_f$ its local growth rate. On a positive-mass interval where negative log mass is absolutely continuous:

$$
\begin{aligned}
K_f(t)&=-\log_2 V_f(t),\\
\kappa_f(t)&=\frac{dK_f(t)}{dt},\\
V_f(t)&=V_f(t_0)\,2^{-\int_{t_0}^{t}\kappa_f(u)\,du}.
\end{aligned}
$$

**Accumulated contraction rates determine mass at a fixed precision; they are not just another isolated metric.** If measurement starting points are equal and branches approximately contract exponentially at constant rates, the coefficients directly relate to probability rankings at a fixed cross-section. If starting points differ or rates change, prefactors and the full path must be retained. The resulting distribution is first an effective distribution under specified output conditions, not automatically the original function prior integrated over all amplitudes.

A training set itself can also have a cost profile. Its local contraction rate is the mass-weighted average of branch rates within the same fixed-D event; under suitable deep-tail conditions, the more slowly contracting group can dominate total mass. This provides a conditional mechanism for approaching more economical compatible explanations, but separate full-target coefficients cannot be substituted without a bridge.

Interpreting contraction rate as complexity requires calibration against independent logical structure, sharing, or external predictions; it cannot rest on the circular argument that faster contraction means greater complexity and greater complexity explains faster contraction. Proxy-complexity comparisons in [E33](experiments/experiment_33_cross_architecture/COMPLEXITY_CALIBRATION_20261027.md) and the prospective amplitude-intervention prediction in [E31](experiments/experiment_31_loss_amplitude_grokking/XOR_OR_ALIGNMENT_20261027.md) are examples of such calibration. Parity rules also remind us that short symbolic descriptions need not be cheap neural implementations.

AIT models that adjust the penalty on program length provide an explicit reference for bias strength. Whether a neural amplitude or loss path implements comparable reweighting remains a matter for neural measurement. No interpretation as physical temperature is needed here. [Tadaki's extension of program weighting](https://arxiv.org/abs/0801.4194)

A precise conditional model fixes a universal prefix machine U and reweights every program producing the same object f by inverse temperature beta. The ordinary coding theorem supplies a constant c independent of f and beta such that:

$$
m_\beta(f)=\sum_{p:U(p)=f}2^{-\beta|p|},\quad \beta\geq1,
\qquad 2^{-\beta K_U(f)}\leq m_\beta(f)\leq c\,2^{-\beta K_U(f)}.
$$

Returning to the original codelength units gives:

$$
K_U(f)-\frac{\log_2c}{\beta}\leq-\frac1\beta\log_2m_\beta(f)\leq K_U(f).
$$

Increasing inverse temperature strengthens the penalty. It is the unnormalized total-mass cost rescaled by 1/beta that approaches the shortest length. Normalization introduces a common normalization term, and tied shortest implementations do not become unique. This is a substantive correspondence for the diminishing extra contribution of longer implementations under stronger penalties, not proof that BCE implements this reweighting or that neural candidate-complexity variance decreases monotonically.

The contraction coefficient remains a measure-dependent empirical quantity. Different norm spheres and their radial weighting jointly determine the aggregate curve. Independent sampling evidence exists beyond Gaussian measures; on November 1, stored particles were also reweighted by a family of norm functions, showing approximately stable coefficients beyond the Gaussian form. This connection to AIT penalty strength therefore needs no uniquely Gaussian measure, but it does not imply unchanged coefficients after changing the measure. [Radial-weight diagnostics](supplementary/updates_20261106/geometry/README_en.md)

## 4 Shared Structure Connects Rules and Memory

Algorithmic mutual information measures the description saved by describing objects jointly rather than separately. The same structure can be reused across samples, input branches, or output tasks; shared intermediate representations and computation are possible neural implementations of this saving.

For equal sample counts, comparable input protocols, and a common independent-description baseline, lower joint cost generally indicates more shareable description; less sharing means smaller savings. An absolute contraction rate does not yet give algorithmic mutual information in bits, and savings in parameters, execution cost, and description length are distinct quantities. [E08](experiments/experiment_08_shared_intermediate_pressure/RESULTS_AND_CONCLUSION.md) provides implementation-level evidence for shared computation.

Rules and noise therefore need not be treated as two categorically different kinds of information. Relative to the current model and observations, a local exception may require a separate description; after more samples arrive, it may reveal itself as part of another reusable rule. Algorithmic statistics studies such decompositions through a model description and a residual description given the model. [Algorithmic Statistics](https://arxiv.org/abs/math/0006233)

This research has used approximate linearity between random-memory contraction coefficients and sample count as an external calibration of the complexity definition. Results include approximate linearity, but also curvature across architectures and variation between training sets. Finite random data need not be devoid of exploitable structure, so strict linearity is not treated as a necessary premise of the whole framework. [E32](experiments/experiment_32_random_memory/MEMORY_EXTENSION_20261027.md)

Searching for the highest-cost dataset at fixed n, or the fewest samples achieving a given cost, offers candidate definitions of a memory frontier; these differ from being least confident on unseen points. A relative rule/noise definition can also be quantified by comparing complexity before and after adding or removing a sample or group, testing whether it is explained by current shared structure or needs an additional residual description. See [complexity contributions of added and removed samples](interpretation_update_20261106_en.md) for definitions and measurement conventions.

For example, once a constant background has substantial support, deleting an ordinary sample may barely lower cost, while deleting its only exception may lower cost considerably and the remaining data may not predict that exception. Further similar exceptions can reveal a rare but reusable complex rule. Rules/noise are therefore not arbitrary names: they concern sharing relative to current data and explanatory background. Individual attribution can depend on addition order without changing the cost of the same fixed endpoint event. Information gain is a mathematical connection under pure conditioning, not a replacement for the primary rule/residual question.

## 5 Information Gain Is Definitional Accounting

Fix a reference measure and an existing positive-mass event E, and add only an actual hard-label event B. Let p be the fraction of mass in the old ensemble supporting that label, with p>0. Then:

$$
G=-\log_2 p.
$$

This is simultaneously the label's predictive surprise, the increase in negative log event mass, and the KL information cost of the new conditional distribution relative to the old one in this setting. It is an identity once the objects are fixed, not a neural regularity awaiting experimental confirmation. The probability p includes all functions supporting the label and cannot be replaced by the mass of one complete function.

Along nested constraints under the same fixed prior, the sum of stepwise costs equals the difference between endpoint costs. The total cost of reaching the same endpoint is order-independent, although individual steps can receive different shares. [E22](experiments/experiment_22_static_free_energy_information/RESULTS_AND_CONCLUSION.md) checked this numerical accounting. It connects to [prequential coding](https://proceedings.neurips.cc/paper_files/paper/2018/file/3b712de48137572f3849aabd5666a4e3-Paper.pdf), but the prequential codelength of an arbitrary practical learner does not thereby become order-independent.

When a rule has been reliably identified, its labels on typical new samples receive high support, so their additional information cost is small. Conversely, informative samples can substantially change candidate mass. A low-mass function need not have low label support at every new input; it is the cumulative cost of fully identifying that function that directly relates to its probability at the starting point.

In an independent, uniform binary lookup-table model, every new entry supplies a fixed 1 bit, while other unseen entries remain unpredictable. Simplicity bias and shared structure allow different samples to have different marginal costs. Training and reasoning can also make existing information more usable without adding external evidence every time; active learning seeks disagreements worth querying externally.

These identities require explicit conditioning. Redefining an average-loss event after adding a sample need not produce a subset of the old event. Parameter-level information changes from continuous-precision filtering are also not identical to new label information, still less an automatic update law for SGD.

## 6 Generalization and the Sample-Size Transition

Generalization requires independent true labels or an evaluation criterion; the true rule cannot be defined after the fact from model predictions. Agreement describes consistency among candidates or repeated training runs and can be measured without a teacher, but consistent answers can still be wrong.

The agreement conjecture is not abandoned: whether high consensus statistically corresponds to reusable, interpretable rules under a fixed system and sampling protocol is an important external test of alignment between neural preferences and human description complexity. Independent symbolic audits of high-consensus endpoints in E18/E21 provide positive evidence. The revision narrows an unconditional claim to a statistical relationship requiring explicit bias strength, sample cost, finite input space and unseen coverage, rather than denying the connection.

Dominance by the complete true function is sufficient for good generalization, not necessary: many complete extensions can give the same correct answer on most test inputs. Averaging per-input correctness probabilities over test inputs equals the ensemble's mean single-model accuracy exactly. Predicting one actual training run additionally requires correspondence between static and optimizer distributions and concentration of accuracy. [Jordan's study of training variance](https://arxiv.org/abs/2304.01910)

Using unlabeled agreement to estimate external error requires another connection, such as suitable ensemble calibration conditions; it does not follow from the definition of agreement. [Jiang et al. on disagreement and calibration](https://arxiv.org/abs/2106.13799)

The single simplest function need not dominate mass either: many individually more complex functions can outweigh it collectively. If they share major structure while retaining different residuals, they may make similar numbers of errors at different locations; small cross-point error covariances can concentrate average accuracy. On a fixed training set, Rule30 measures [accuracy, between-seed agreement and shared errors under norm penalties](supplementary/updates_20261106/training_controls/README_en.md#rule30-norm-agreement). These statistics do not directly measure internal shared structure or cross-example error covariance; the [predictive-distribution account](interpretation_update_20261106_en.md) distinguishes them from Jordan's results.

The grokking transition here means a sample-size recovery threshold, not the training time required for delayed generalization. It depends on the target's cost relative to competitors and how much competing mass samples can exclude. Under comparable conditions, this can yield approximate linearity or a stable complexity ordering, but not a universal strict proportionality law for all rules. [Conditional derivation](translations_en/theory/mdl_identifiability_grokking_derivation.md), [E23](experiments/experiment_23_volume_to_data_transition/RESULTS_AND_CONCLUSION.md)

When reporting this threshold, specify how examples were drawn, at what training loss the comparison was made, and what counts as success. More examples from the same rule can exclude wrong extensions directly. With fewer examples, the correct extensions might not dominate until training reaches a lower loss. The [Rule30 trajectories](width_double_descent_en.md#post-fit-generalization-lag) illustrate the distinction: with 1024 examples all 5 seeds eventually got validation entirely right, but only after a delay following perfect training accuracy. With 4096 examples the two endpoints were almost simultaneous. The trajectories measured the delay, not the strength of the simplicity bias.

The comparison concerns total competing mass in a soft distribution, not an MDL decision that necessarily chooses the shortest description. Minimum feasible complexity, probability cost, and contraction coefficient are not interchangeable; independent and without-replacement sampling also have different survival probabilities. Statistical correlation between neighboring truth-table complexities may help explain similar competition spectra within some rule families, but requires measurement and does not follow directly from network continuity.

If two explanations are indistinguishable within the available observation range, more observations of the same kind cannot decide between them. New variables, experimental conditions, or input ranges may reveal shared structure at a broader level. Identifiability depends not only on rule complexity but also on what can be observed.

The idea discussed on 2026-10-30 of estimating algorithmic mutual information from sample-size transitions compares recovery thresholds for separate tasks with those for a joint task organized by a rule bit or output concatenation, using the savings as a proxy for shared description. It is neither newly proposed nor newly validated by this editorial update, and differs from the [existing measurement proposal](complexity_measurement_methods_en.md) that uses a transition to select query samples before running SMC. Without calibration, differences in sample requirements cannot be reported directly as algorithmic mutual information in bits.

The new four-point experiments include examples combining a relatively high full-extension cost proxy with high agreement. This limits treating the two memory criteria as synonyms, but does not directly measure low-BCE training-set cost. Constructed examples and their frequency under random sampling from a fixed generator remain distinct. The impossibility of recovering every member of a large distinguishable rule family from very few labels is a separate, family-level information constraint, not a prohibition on well-designed examples. [Consensus and cost proxies](supplementary/updates_20261106/consensus/README_en.md)

The "tree of added samples" represents dataset filtering and candidate competition: economical explanations such as constants can dominate early; incompatible evidence disperses the old consensus, and further identifying examples concentrate a new rule or behaviorally similar group. High-consensus nodes reflect implementation economy shaped by architecture and computational structure; more complex rules can also form nodes with suitable data. Non-stochastic objects offer a conceptual reference for rule/residual separability, not a guarantee of unique extension. Shared compressible structure is the central hypothesis connecting neural preferences, human readability and high consensus, still requiring independent calibration.

## 7 What This Account Adds

Basic induction can occur under a fixed bias; continuous-output constraints provide an additional mechanism for adjusting that bias. A fixed finite target in raw-output MSE differs from BCE's continuing margin-growth path, but either defines a static ensemble at a specified endpoint. The object of study is the particular joint-output region, not the loss name itself.

Nor does a sufficient fixed bias automatically mean that the original initialization prior, filtered only for correct labels, is sufficient. Sufficient MSE fitting also requires numerical outputs to approach their targets. G&C is a rejection-sampling method that can check hard correctness or finer loss conditions. With the same prior and final event, it and SMC target the same conditional distribution; computational efficiency and finite-sampling error differ.

Prior work includes not only general compression intuitions, but function posteriors, continuous-loss strata, local behavior volumes, and sample bounds for soft interpolator sampling. This research links continuous constraints, function-mass profiles, independent calibration, and training predictions in explicit finite protocols, rather than claiming that previous work ignored loss. Sampling/gradient separations under continuous MSE also limit universal equivalence claims. [Original work and comparison objects](supplementary/updates_20261106/related_work/README_en.md)

| Main connection | Evidence in this research | Retained boundary |
|---|---|---|
| Precision changes function mass | [E19](experiments/experiment_19_full_truth_rule_volume/RESULTS_AND_CONCLUSION.md), [E24](experiments/experiment_24_deep_neural_k_crossing/RESULTS_AND_CONCLUSION.md), [E31](experiments/experiment_31_loss_amplitude_grokking/README.md) | Rankings can cross; arbitrary cross-sections are not one interchangeable scale |
| Shared structure and independent calibration | [E08](experiments/experiment_08_shared_intermediate_pressure/RESULTS_AND_CONCLUSION.md), [E33](experiments/experiment_33_cross_architecture/COMPLEXITY_CALIBRATION_20261027.md) | No universal equality with classical Kolmogorov complexity has been established |
| Information cost and data recovery | [E22](experiments/experiment_22_static_free_energy_information/RESULTS_AND_CONCLUSION.md), [E23](experiments/experiment_23_volume_to_data_transition/RESULTS_AND_CONCLUSION.md) | Identity-based accounting and empirical transition prediction are distinct |
| Static distributions and training | [E36](experiments/experiment_36_peleg_g_and_c_validation/VALIDATION_EXTENSION_20261027.md) | Similar predictions do not imply identical optimizer distributions |
| Memory and reference measures | [E32](experiments/experiment_32_random_memory/MEMORY_EXTENSION_20261027.md), [E37](experiments/experiment_37_measure_robustness/README.md) | Nonlinearity, data variation, and the tested range of measures remain explicit |

**The resulting main line is: rich representations accommodate explanations, nonuniform implementation mass creates preferences, data determine compatibility, and shared structure supports generalization; continuous-output precision can additionally change bias strength and provide a route for measuring complexity.** Architecture determines which implementations are economical, while the optimizer determines whether finite resources can reach them. AIT supplies theoretical language for these connections; neural experiments test their implementation and scope.

For fuller definitions, see the [theory core](theory_core_en.md) and [strict probability language](translations_en/theory/probability_language.md). This page does not develop Ω, incompleteness, or detailed proofs of temperature corrections, and does not make those branches premises of the main account.

Description length, execution time, identification sample requirements, and optimization-search cost are different resource axes. They have not been jointly calibrated into machine-independent KC.
