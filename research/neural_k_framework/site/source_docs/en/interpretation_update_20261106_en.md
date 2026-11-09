# Distinguishing Measure, Amplitude, Complexity and Prediction

Two networks get every training example right but give different answers on new ones. Of all the rules that fit, which will a network usually learn? Consider a binary classifier that reads a positive output as 1. One network outputs 0.1, another outputs 10. They give the same **answer** but different **scores**. For label 1, binary cross-entropy (BCE), a common training loss, assigns the second network a smaller loss.

Does changing 0.1 to 10 require a new way of solving the problem? Section 3 shows why rescaling the last layer alone can do it. Even if the same solution is possible at either scale, is it equally common when we sample weights, or when we train? Sections 5 and 6 examine those two ways of obtaining weights. Separating what is possible from what is probable gives us a way to ask which rules generalize. The [experiment overview](research_updates_20261106_en.md) starts with a training comparison.

## 1 Four Steps from the Actual Problem to Prediction

For example, to teach a network addition, first decide which numbers represent which symbols. Use that same code for training and test questions. Training gives one set of weights. Run those weights on the questions to obtain a vector of real scores, then decode them using the agreed rule. **Read the scores from one set of weights together**: a new initialization can change the whole answer sheet. The distribution of those vectors over repeated draws of weights is the joint output distribution.

On a finite probe set U, let the continuous outputs be Z_U(theta) and the decoded table be H_U(theta). The output distributions induced by reference measure mu and actual training pushforward nu are, respectively:

$$
\nu_U^{\rm ref}=(Z_U)_\#\mu,\qquad
\nu_U^{\rm opt}=(Z_U)_\#\nu,\qquad H_U=\operatorname{decode}\circ Z_U.
$$

The conditioning event for a fixed training set D constrains only D. Unseen points can decode correctly at small or different amplitudes; they need not satisfy the same margins as on D. Thus fixed-D conditional prediction is not equivalent to adding all test labels to a full-target event. Whether their trends are similar is an empirical question, not something to assume by definition. [E20](translations_en/experiments/e20.md)

Consistently re-encoding training and test data can preserve task information, but does not guarantee unchanged representation, optimization or sample costs. An arbitrary fixed pixel permutation differs from a separate random permutation for each image; Swin's local windows do not automatically eliminate the influence of positional encoding either. No new image-permutation experiment is reported here, and symmetries of other architectures cannot be transferred directly to CNNs or Swin.

### 1.1 Early Encoding Experiments: Symbols, Positions and Real Values {#encoding-evidence}

The historical experiments revisited on 2026-11-07 provide concrete counterparts to these four steps. For an actual rule f, input encoder E_in and output decoder D_out, evaluation concerns decoded task behavior, not just the network's raw numerical outputs:

$$
\widehat f_\theta(x)=D_{\rm out}\!\left(F_\theta(E_{\rm in}(x))\right)\approx f(x).
$$

**Symbol and position shuffling have results recorded in the earlier paper.** Section 4.1.2 of that paper ([Chinese PDF](https://github.com/ball-lightning6/neural-sculpting-paradigm/blob/3834fb3f6555c7c58218462502635b16b9b377f9/paper_zh.pdf), [English PDF](https://github.com/ball-lightning6/neural-sculpting-paradigm/blob/3834fb3f6555c7c58218462502635b16b9b377f9/paper.pdf)) trains a Transformer to add two 8-digit hexadecimal numbers: 16 input characters, 33 binary output bits, and mean BCE. The [generator](https://github.com/ball-lightning6/neural-sculpting-paradigm/blob/3834fb3f6555c7c58218462502635b16b9b377f9/symbolic_math_logic/generate_symbol_add_shuffle_dataset.py) separately controls a fixed substitution of digit symbols with printable characters and a fixed permutation of the 16 input positions. Mappings remain identical across training and evaluation within each experiment, rather than being resampled per example; outputs remain standard binary. The table below transcribes each condition's last listed evaluation point from the table on page 11 of both PDFs.

| Symbol substitution | Position shuffle | Last listed step | Eval BCE |
|---|---|---:|---:|
| No | No | 12000 | 0.000099 |
| No | Yes | 7000 | 0.000024 |
| Yes | No | 9000 | 0.000051 |
| Yes | Yes | 25000 | 0.000095 |

All four records reach evaluation BCE below 0.0001, supporting learning without dependence on the original symbol identities or natural digit order. This is a review of historical results, not a rerun in this update. Different stopping points and multi-seed logs not rechecked here do not establish equal learning costs or statistical equivalence, and BCE is not converted directly into whole-string accuracy. The results do not establish that the network internally performs human-style digitwise addition or an explicit inverse-permutation algorithm.

**Historical code also specifies real-valued input and output encodings.** The [floating-point input addition generator](https://github.com/ball-lightning6/neural-sculpting-paradigm/blob/3834fb3f6555c7c58218462502635b16b9b377f9/symbolic_math_logic/generate_add_quaternary_float_encoding.py) groups the inputs of two 20-bit addends into pairs, producing 20 real-valued inputs with `00→0.0, 01→5.5, 10→1.3, 11→10.0`, while retaining a 21-bit binary output. The [real-valued output CA generator](https://github.com/ball-lightning6/neural-sculpting-paradigm/blob/3834fb3f6555c7c58218462502635b16b9b377f9/cellular_automata/generate_ca_to_abstract_real.py) retains a 30-bit input and encodes the output after 5 Rule110 steps into 15 real values using `00→7.6, 01→1.3, 10→5.9, 11→3.0`. These are different tasks, not a single controlled experiment re-encoding both sides. The encoding protocols were checked, but the corresponding training results have not all been located, so no new success rates or convergence values are reported. Real-valued outputs can be decoded to the nearest valid code value; this readout must be specified together with the actual training and evaluation protocol.

These experiments connect to the early claim that networks learn structure: a basic-symbol codebook and positional conversion can be reused across many examples, so changing surface form need not destroy the compressibility of task relations. Encoding nevertheless belongs to the neural reference language and can alter loss geometry, candidate mass and learning cost. The evidence concerns learnability under specific re-encodings, not invariance to arbitrary encoding or a guarantee that randomly indexing the entire input space remains equally easy.

## 2 Priors and Contraction Are Different Readings of the Same Joint Distribution

A function prior integrates over all continuous outputs within a decoded cell; equal-amplitude targets, mean BCE and pointwise BCE bounds select different subsets of that output space. They derive from the same underlying object, but are not the same integral.

$$
P_\mu(f)=\nu_U^{\rm ref}(C_f),\qquad
V_f(a,\delta)=\nu_U^{\rm ref}\{z:\|z-a s_f\|\leq\delta\},\qquad s_f\in\{-1,+1\}^{|U|}.
$$

The fact that all target rays pass through zero does not imply equal priors for all hard functions: sign decoding at zero may be ambiguous, and a ray or its small neighborhood does not cover an entire cell. Contraction curves along equal-amplitude directions alone cannot reconstruct a prior obtained by integrating over all amplitudes and nonuniform outputs.

For the same event family, over a differentiable window with positive mass, contraction rates can be accumulated exactly into a mass ratio:

$$
K_f(t)=-\log_2V_f(t),\quad \kappa_f(t)=\frac{dK_f}{dt},\qquad
K_f(t)-K_f(t_0)=\int_{t_0}^{t}\kappa_f(u)\,du.
$$

This explains the close connection between bias at a fixed section and a contraction profile; it also requires retaining the starting mass, coordinate and entire integration path. Knowing a slope over one segment does not determine the original prior, nor do identically named thresholds in different losses represent the same precision.

## 3 What the Last Linear Layer Can Scale

If the final readout is F_theta(x)=v^T phi_eta(x)+b, multiplying both v and b by a positive number a multiplies every logit by a. Existing hard classifications remain unchanged; if every training margin is strictly positive, increasing a can drive BCE to zero. This requires no change in the hidden representation. Incorrect or zero margins cannot become correct, strictly positive margins through this operation.

Thus "tightening the loss excludes shortcuts" has two different meanings: a current implementation that fits unfaithfully is excluded; or a class of implementations that still exists loses relative mass. The former does not establish that all implementations of the same hard function cease to exist. Lower BCE is also insufficient evidence of new features.

Collect the readout parameters into a q-dimensional vector v and let S_a(eta,v)=(eta,av). For a reference measure with density rho and any measurable parameter region A, a change of variables gives:

$$
\mu(S_a A)=\int_A a^q\rho(\eta,av)\,d\eta\,dv.
$$

If the readout coordinates follow a zero-mean isotropic Gaussian with covariance sigma^2 I, independently of the hidden parameters eta (using the corresponding standardized coordinates when layer scales differ), the relative change in density and volume element is:

$$
\frac{a^q\rho(\eta,av)}{\rho(\eta,v)}
=a^q\exp\!\left[-\frac{(a^2-1)\|v\|^2}{2\sigma^2}\right].
$$

This is an exact change of variables for a scaled region, not a computation of all candidate masses. It shows that a one-to-one correspondence need not preserve probability. It also shows that a small readout share of the total norm does not by itself imply a small change in Gaussian cost: what matters is the absolute increment in squared norm in whitened coordinates. Rescaling can also change an event's relative tolerance: when an MSE target is multiplied by a, the MSE tolerance must also be multiplied by a^2 to represent the same relative output error.

When the entire unconstrained network is retrained, changes in output scale can accompany changes in hidden representations, layer norms and optimization paths. But "the network must distribute amplitude evenly across layers" is not a general law. The balanced factorization favored by a minimum-norm solution and whether gradient descent finds that factorization are different questions.

For example, consider the two-layer scalar linear model yhat=uvx, a differentiable data loss depending only on the product uv, continuous gradient flow with equal learning rates and no regularization:

$$
\dot u=-v\,\partial_{uv}L,\quad \dot v=-u\,\partial_{uv}L,
\qquad \frac{d}{dt}(u^2-v^2)=0.
$$

The gradient can also be zero once an exact fit has been reached. A highly unbalanced factorization therefore does not automatically become balanced merely through gradient descent. Adding regularization, changing the optimizer, introducing nonlinearities or changing learning rates requires fresh analysis. Normalizations such as LayerNorm also introduce specific scaling equivalences, making the raw total weight norm still less suitable as a unique cost intrinsic to the function.

## 4 Amplitude Alignment Does Not Force Every Margin Upward

Mean BCE allows different samples to bear different errors. Variance regularization constrains these differences, but when the common amplitude is free, it can increase low margins, decrease high margins or shrink outputs overall. For a fixed margin vector m:

$$
\operatorname{Var}(a m)=a^2\operatorname{Var}(m).
$$

Ordinary BCE itself attenuates gradients at high-margin points; the variance term is not what first makes it attend to difficult points. Difficult points may reflect undiscovered structure or incorrect labels, and the variance term does not directly distinguish between them. Broader fitting, optimization changes and reallocation of candidate mass can occur together; no single mechanism should be presumed.

The f/g intervention shows that mean BCE, pointwise BCE constraints and variance alignment can produce different rankings; the original improvements and prospective reversal prediction remain evidence. This supports including precision allocation across samples in complexity calibration rather than treating one mean-BCE section as the only scale. A variance term does not fix the common amplitude, so a decrease in absolute variance alone does not identify the mechanism. [E31](translations_en/post_upload/e31.md)

## 5 Measure, Radius and Contraction Shape {#measure-contraction}

The central contribution in this line is the sequence of diagnostics on 2026-11-01: measuring contraction on different norm spheres, examining how Gaussian-conditioned mass moving to larger radii offsets some curvature, and then testing other norm-only weights on the same stored particles. Within the exp[-(r/a_p)^p] family with matched prior mean squared norm, p=1.5, 2.5 and 3 retain approximately 2%–3% first-to-last segment slope changes for parity4; a Gaussian is not the only weighting with approximately stable coefficients. Changing Gaussian variance to 0.9 or 1.1 of its original value also changes the window coefficient rather than merely translating K. [Current radial-reweighting diagnostic](supplementary/updates_20261106/geometry/README_en.md)

This is read-only importance reweighting of existing samples, without a new independent SMC run for each weight. The heavier-tailed p=1 case has marked weight concentration and cannot support a reliable stability claim. That evidence-level distinction does not erase the observed family of weightings; it locates the evidence accurately, with earlier independent cross-measure sampling providing a complement.

The volume-contraction law is empirical within a specified protocol and window, not exclusive to a Gaussian measure. Earlier E37 uniform-solid-ball and uniform-cube measurements, together with sphere controls in this update, also show rule-dependent contraction and approximately linear intervals; the uniform ball was additionally rechecked with an independent transition kernel. Spherical t8 curves flatten more noticeably, so stability differs across weightings. Here "stable complexity" means a relatively steady effective contraction coefficient over a window, not that K stops growing. [Cross-measure evidence](supplementary/updates_20261106/geometry/README_en.md)

**Not requiring a Gaussian does not mean independence from the measure.** Weighting affects values, slopes and curvature, not merely a normalization constant. For a spherical measure, let A_f(r,t) be the uniform angular fraction satisfying the event at radius r and rho_mu(r) its radial probability density; a Gaussian is only one choice:

$$
V_{\mu,f}(t)=\int A_f(r,t)\rho_\mu(r)\,dr,
\qquad \rho_{\mathrm G}(r)\propto r^{d-1}e^{-r^2/(2\sigma^2)}.
$$

Gaussian parameter density is largest at the origin, but radial mass includes the volume factor r^(d-1). A fixed sphere, uniform ball and Gaussian are not the same kind of uniformity. A cube also changes the angular distribution, so it cannot be treated solely as radial reweighting of the common A_f above.

For a finite radial partition with fixed prior weights p_j, define V=sum_j p_j V_j, K_j=-log2 V_j and conditional stratum weights w_j=p_j V_j/V. Within a differentiable window:

$$
K'=\mathbb E_w[K_j'],\qquad
K''=\mathbb E_w[K_j'']-\ln(2)\operatorname{Var}_w(K_j').
$$

Mass migration between strata can therefore partly offset within-stratum curvature, but does not guarantee a strictly linear curve. The [radial experiment](supplementary/updates_20261106/geometry/README_en.md) tests this accounting. Greater radii being more favorable to some complex rules is a relative finding for specified coordinates, rules and windows, not a claim that all rules become simple at large norms.

The empirical form V≈C epsilon^kappa is approximately exponential in log(1/epsilon), with both C and kappa indexed by measure, architecture, rule and window. It is not exponential in raw epsilon or output amplitude. Local slopes, secant fits and eventual asymptotic exponents are not interchangeable. The substantive observation is measurable relative-mass change and additional penalties on complex rules recurring across multiple protocols, not an unconditional, measure-independent exact power law.

SLT's local risk volumes and RLCT can motivate effective-dimensional explanations, but SLT is not restricted to MSE, nor can it be equated directly with our global BCE volume. Local error around a fixed finite output target, an event moving toward larger margins, and all implementations of one hard function are different objects. Current local probes find that the same finest-scale rank can accompany different global contraction, so effective parameter count is only part of a candidate explanation. [Geometry probes](supplementary/updates_20261106/geometry/README_en.md)

## 6 How Close Must Static Sampling and Training Be? {#prediction-concentration}

First consider groups of functions rather than only the single simplest function. Let B be a group of compatible candidates. Each Q(f) already integrates all parameter implementations of f; the group's mass must additionally sum over functions:

$$
Q(B)=\sum_{f\in B}Q(f)=|B|\,\overline Q_B.
$$

A few cheapest candidates may be outweighed collectively by many candidates that are each slightly more expensive. This has a counting logic similar to high-dimensional mass not necessarily concentrating at the point of maximum density. Changing the strength of simplicity bias can move the typical candidate group and its test performance; an ensemble is not equivalent to always selecting the unique minimum-cost explanation.

The static reference distribution Q differs from the training pushforward P. Even if their parameter distributions are not close, a particular observable can be similar; however, similar accuracy alone does not imply similar distributions. Let g(f) be single-model test accuracy. If P is absolutely continuous with respect to Q, w=dP/dQ and w has a finite second moment, then:

$$
\mathbb E_Pg-\mathbb E_Qg=\operatorname{Cov}_Q(w,g),\qquad
|\mathbb E_Pg-\mathbb E_Qg|\leq\sqrt{\operatorname{Var}_Q(w)\operatorname{Var}_Q(g)}.
$$

This supplies a conditional robustness mechanism: if g is concentrated under Q and the reweighting is not excessively biased, predictions can be insensitive to distributional differences. It does not guarantee that actual SGD meets these conditions. An optimizer may still strongly amplify regions with extremely small static mass; saying that "good functions account for 99.99%" cannot by itself exclude this possibility.

For correctness indicators Z_i at fixed test points, the following expectation and variance formulas do not require errors across samples to be independent:

$$
A(f)=\frac1m\sum_iZ_i(f),\qquad
\mathbb E_QA=\frac1m\sum_i\Pr_Q(Z_i=1),
$$

$$
\operatorname{Var}_Q(A)=\frac1{m^2}\left[\sum_i\operatorname{Var}_Q(Z_i)+2\sum_{i<j}\operatorname{Cov}_Q(Z_i,Z_j)\right].
$$

Thus "70% correct mass at every point" gives 70% mean single-model accuracy. If error covariances across test points are sufficiently small, averaging over many points can also concentrate individual-run accuracy. Different models can make similar numbers of errors at different locations. If errors are perfectly synchronized, individual accuracy can instead be only 0 or 1; concentration depends on joint error structure, not pointwise means alone.

Jordan repeated training 60000 times in a fixed CIFAR-10 setting. At 64 epochs, accuracy on the fixed test set had standard deviation 0.149 percentage points; the analysis found approximately uncorrelated error indicators across different test examples as the seed varied, while allowing different error probabilities per example. This is not independence of two fixed models' error events over random inputs. [Jordan, Section 3.2](https://arxiv.org/pdf/2304.01910v4#page=5)

The Jiang et al. paper cited by Jordan further connects disagreement and error: for independent models h and h' from the same training-induced distribution, the stated calibration conditions on hard-prediction vote probabilities imply equality of expected disagreement and mean error. This is not calibration of one model's softmax and not an exact equality for every model pair. [Jiang, Section 4](https://arxiv.org/pdf/2106.13799v2#page=7)

Our mechanistic account is that a candidate group can reuse similar shared structure while retaining different training-specific residuals. If candidates have similar total error, or small cross-point error covariances, concentrated accuracy can coexist with differing error locations. A finite-strength simplicity bias can also make a moderately costly but numerous function group dominant. This is an explanatory direction for the statistics, not a claim that the two papers measured the complexity of that group or its internal residual structure.

The Rule30 norm intervention supplies a related observation. At 600 inputs, below the observed recovery range, all groups sufficiently fit the same training set, but mean whole-string accuracy is 68.5181%, 79.6570% and 39.5203% for baseline, high-norm penalty and low-norm repulsion. The hard-label-compatible set is unchanged; regularization and actual training selection change. This is compatible with altered relative candidate-group mass or accessibility. Differences shrink again as data become sufficient over the 750 to 900 input levels. [Training controls](supplementary/updates_20261106/training_controls/README_en.md)

The [endpoint predictions of the same models](supplementary/updates_20261106/training_controls/README_en.md#rule30-norm-agreement) also give whole-ring agreement. Even when both baseline seeds in a pair got the same input wrong, their wrong strings differed in 71.05% of such cases. This measures whether seeds give the same answer, not Jordan's error covariance between different inputs or the complexity of static candidate functions.

An ensemble across seeds need not match a time ensemble along one trajectory. Slowly changing late-stage loss and stable predictions may indicate that the trajectory stays within a narrow predictive region, or that finite observations cannot detect change; they do not prove ergodicity. In the new wide-CNN check, time agreement is much higher than agreement across seeds, reinforcing the need for this distinction. [Consensus check](supplementary/updates_20261106/consensus/README_en.md)

## 7 Agreement, Rule Nodes and Memory {#agreement-conjecture}

The core agreement conjecture is retained: do high-consensus neural preferences often correspond to human-interpretable, reusable structure with low description cost? E18 found signed-threshold rules at all 46 strict complete-consensus endpoints confirmed after screening 8192 random datasets; E21 also found more complex but readable quadratic-threshold rules. These independent symbolic checks are positive evidence linking neural preferences to human complexity, not merely "consensus means consensus." The revision adds conditions concerning architecture, data generation, sample budget, bias strength and unseen coverage to this statistical relationship rather than abandoning it. [E18](translations_en/experiments/e18.md), [E21](translations_en/experiments/e21.md)

For pointwise predictions q_x(y) on an unconstrained probe set T, pointwise agreement between two independent draws and the complete-function collision probability are, respectively:

$$
\operatorname{Agr}_T(Q)=\frac1{|T|}\sum_{x\in T}\sum_yq_x(y)^2,
\qquad \operatorname{Coll}_U(Q)=\sum_f Q(f)^2.
$$

These are not the same quantity. Many functions differing at only a few points can still yield high pointwise agreement. High agreement therefore guarantees neither a unique complete function nor a unique internal implementation. True labels must be specified by an independent generator or evaluation distribution; "generalization" cannot be defined after the fact from the predictions.

Treating added hard labels as a refinement of events, the current conditional distribution can be written as a mixture over subsequent mutually exclusive and exhaustive branches B_j:

$$
Q_E=\sum_j Q_E(B_j)\,Q_E(\cdot\mid B_j).
$$

A tree of added samples therefore permits dispersion, concentration and renewed dispersion. A few same-label examples may first favor a constant; an incompatible sample weakens the old explanation, and further evidence for an AND-like rule can concentrate a new group. Simple rules often gain an advantage earlier, while more complex rules can also form high-consensus nodes given sufficiently identifying data. Different orders may reach the same endpoint; redefining the mean-loss event still requires checking whether it constitutes refinement.

A "rule node" is not an additional mysterious entity. It is a state in which an economical behavior or a behaviorally similar group dominates the constrained region. This remains informative: which nodes form readily reflects implementation economy determined by architecture, computational structure and bias strength. A finite input space limits independent samples and remaining validation space; once coverage is complete, training consistency is no longer evidence about unseen extensions.

Training-set cost, confidence in unseen predictions and correctness relative to the true generator are related but distinct axes. High full-extension cost and high agreement coexist in the four-point XOR example, while XOR remains human-readable; this does not directly refute the link between consensus and interpretable structure. High-cost frontiers at fixed n remain relevant to memory, but this proxy is not direct K_D. Non-stochastic objects and the separability of rules from residuals also offer a conceptual connection, without equating them to high agreement or requiring a unique shortest program.

### 7.1 Rules and Noise: Complexity Contributions of Added and Removed Samples {#rule-noise-contributions}

The primary question is whether a sample belongs to a shared explanation of the current dataset or requires an additional residual description. Fix the protocol, precision requirement and per-point event E_z, let E_D be their intersection, and define C(D)=-log2 mu(E_D). For positive masses, addition and deletion contributions are:

$$
\Delta^+(z\mid D)=C(D\cup\{z\})-C(D),\qquad
\Delta^-(z\mid D)=C(D)-C(D\setminus\{z\}).
$$

When many constant-label samples support a stable background, deleting an ordinary sample may barely change cost; deleting the only exception may lower cost substantially, and the remaining samples may not predict that exception. This makes rules/noise relative to the current background quantitatively testable. A large contribution from one sample is not a certificate of objective noise: the first observations of a rare complex branch may look the same, with later examples revealing reusable structure.

The same additions and deletions can be applied to sample groups to expose shared explanations within them; pointwise attribution depends on background and order. The main object here is the rule/residual distinction, with information gain only a related identity under pure conditioning. Renormalizing mean BCE can break nesting, and a local contraction-slope difference is not automatically a negative-log-mass difference, so the measurement convention must remain fixed. See the [rule/noise quantification in the theoretical core](theory_core_en.md) for the full definition.

## 8 Independent Complexity Calibration Remains Essential

A neural network can serve as a resource-bounded representation language. Different architectures provide different low-cost operations and forms of reuse, so some ranking changes are reasonable. Compiler constants between universal languages are not the same as finite-network execution time, required width or implementation-mass overhead, nor do they guarantee similar rankings on real tasks.

For an effectively encodable discrete generative protocol, probabilistic coding gives a one-way relation:

$$
K(f)\leq-\log_2P_\Pi(f)+K(P_\Pi)+O(1).
$$

High generation probability can therefore yield a short universal description, but a short description does not guarantee large implementation mass in a specified network. Still less can a contraction slope be substituted directly into the right-hand side. Independent logical constructions, shared structure, shortest descriptions within a specified language and external predictions remain necessary calibrations.

"Preserving the original computation while adding irreducible extra work" is a useful structural comparison, but does not currently constitute a strict monotonicity theorem across all architectures and output protocols. An extra branch may change how often the original computation occurs or its required precision, as the f/g intervention already illustrates. For functions without a structural inclusion relation, interpretable architecture-dependent ranking reversals should be allowed, rather than retrospectively relabeling every winner as simple.

Back to the two classifiers: an output of 10 does not establish that its network learned a more advanced feature than the one that outputs 0.1. We can instead ask which rule becomes more common as the training scores must become more decisive, then see whether repeated training changes predictions on new examples in the same direction. A match gives us a testable prediction. Calling the winning rule simpler requires one more check against its logic or another independent standard.
