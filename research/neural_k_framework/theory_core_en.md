# The Neural K Theory Core

> This page presents only the final theoretical framework and does not retell the research process. Experiment IDs serve only as evidence anchors.

## One-sentence version

Once the network architecture, encoding, parameter measure, and loss are fixed, a neural network is inherently more likely to represent certain simple functions; continuing to lower the training loss then causes different functions and their internal implementations to lose parameter volume at different rates, usually giving an advantage to descriptions with lower neural-network-relative complexity and higher reusability. Therefore, on structured data, pushing the training loss lower is often equivalent to requiring the network to describe the training set with a more economical program. The data determine which programs remain compatible with the samples, the network architecture determines what counts as "economical", and the optimizer determines whether these low-loss programs can actually be reached from initialization.

## 1. Two kinds of simplicity are two scales of one principle

### 1.1 Prior simplicity

When network parameters are drawn at random, the probabilities of different functions are highly nonuniform. A function that can be implemented by more parameter configurations has greater mass in the function prior induced by the network. A large body of existing research and this project's own prior sampling both show that structurally simpler, more architecture-aligned functions typically have greater prior mass.

This does not mean that every function that looks simple to humans is easier. Parity is the classic counterexample: it is short in the human XOR language, but can be expensive in the representational language of an ordinary tanh MLP.

### 1.2 Loss-resolved simplicity

The cleanest definition of, and the direct evidence for, the loss-resolved simplicity used here is: **use all input-output samples of each candidate function**, then measure that function's full-target loss-volume profile separately. For a finite Boolean space this means using the complete truth table; for spaces that cannot be enumerated, a sufficiently complete target set, applied consistently to all candidates, must be specified in advance. It does not start from a small subset of training samples and then automatically treat the data generator known to the researcher as the only target function.

Under a full target, parameters can still sit at different loss depths even after they already produce all correct hard outputs.

> **Core conclusion: as the raw loss is tightened further, the usable parameter volume of each function does not shrink in equal proportion. Implementations that are more complex, rely on many non-reusable exceptions, or are misaligned with the architecture often contract faster at deep loss; more economical implementations gain a relative advantage.**

When only a partial training set $D$ is given, what is measured is "among the low-loss parameters that satisfy $D$, how much mass each complete extension occupies". This fixed-$D$ candidate mass is strictly related to the full-target volume, but it is not the same quantity; [E20](experiments/e20.html) established the measure-theoretic link between the two through conditional probabilities and a margin bridge. Which function dominates under a partial training set is also affected by sample coverage, imbalance, and other compatible extensions; the volume ranking under the full target cannot be carried over directly without bridging.

This empirical regularity cannot be written as "some pre-specified simple function rises monotonically over all loss intervals". [E12](experiments/e12.html), [E14](experiments/e14.html), and [E24](experiments/e24.html) have shown that the winner and the local contraction rates can swap order as the loss changes. The truly stable theoretical object is an entire complexity curve, not a permanent scalar.

### 1.2.1 Constraint elimination and reallocation among the feasible candidates: why the complexity distribution can narrow

Lowering training error involves at least two different processes. Adding hard-label constraints within a fixed parent ensemble, or tightening the loss threshold for a fixed training set, can eliminate shortcuts and extensions that no longer satisfy the requirements; among functions that remain feasible, the continuous-precision requirements of the joint outputs reallocate parameter mass. These effects can coexist, and the latter cannot be replaced by the former. When adding samples redefines the average-loss event, its nesting must be checked separately.

Let the remaining function mass be

$$
Q_t(f)=\frac{\Omega_t(f)}{\sum_g\Omega_t(g)},
\qquad
s_t(f)=-\partial_t\ln\Omega_t(f),
$$

where `t` is a pre-fixed loss-depth coordinate. Within a window where positive-mass branches are differentiable and summation and differentiation can be interchanged, the following identity holds for an external complexity scale `kappa(f)` that does not change with `t`:

$$
\frac{\mathrm d}{\mathrm dt}E_{Q_t}[\kappa]
=-\operatorname{Cov}_{Q_t}(\kappa,s_t).
$$

If contraction rates correlate positively with fixed costs within the current window, average complexity decreases. Whether the variance decreases is a separate question. Let $m_t=E_{Q_t}[\kappa]$. Under the same differentiability conditions:

$$
\frac{\mathrm d}{\mathrm dt}\operatorname{Var}_{Q_t}(\kappa)
=-\operatorname{Cov}_{Q_t}\bigl((\kappa-m_t)^2,s_t\bigr).
$$

A nonpositive right-hand side guarantees nonincreasing variance in that window. Fixed support or the absence of new modes alone does not guarantee narrowing; a smaller cost variance does not mean that a single complete function dominates.

This provides a layered description of the earlier observations: shortcuts insufficient for the training requirements can exit; in protocols with informative data, the target rule can become competitive among the remaining candidates; as loss is pushed lower, relatively faster contraction of complex implementations can lower the mean fixed reference cost. Further narrowing or concentration on a few functions must be measured separately. Even when the complete hard function is fixed, its continuous logit implementations can still be filtered, without changing that hard function's fixed reference cost.

Therefore, the empirical observation that "the complexity distribution keeps narrowing" should be stated as a conditional mechanism supported by the output joint distribution and the loss cross-sections, not as "lower loss necessarily implies a monotonic decrease in complexity".

This also corrects the earlier BCE-centered wording by placing the joint output distribution at the center: the architecture, encoding, and parameter reference measure first determine the continuous joint distribution over outputs on multiple examples, while BCE, MSE, and amplitude regularization cut different regions from it. The specific loss genuinely changes function odds, ordering, and contraction rates through the geometry of its cross-section, but the name of no particular loss is a universal mechanism. The theoretical claim concerns the reallocation of function mass under a specified joint-output event.

### 1.3 Relation to "compression is intelligence"

Ordinary training code explicitly optimizes only the training loss; there is no second term for machine-independent program complexity. Compression is a consequence produced by the network architecture, the parameter measure, and the loss geometry: if reusing features, intermediate computations, and rules can satisfy more samples simultaneously with fewer independent degrees of freedom, it is usually easier for such a solution to keep reaching deep loss.

Therefore, in clean, structured tasks with sufficient data, lowering the loss can basically be understood as continually searching for a more economical description of the training set. This connects directly to "compression is intelligence": intelligent behavior comes from discovering reusable regularities, not from storing an independent answer for every input. [E08](experiments/e08.html) gives causal evidence for shared intermediate computation, and the early hundred-plus rule experiments provide breadth evidence.

But this is not an unconditional equality. Early in training, when the task is learned from constant outputs, the absolute program complexity can rise; with random labels or noise, continuing to lower the loss can also increase the memorization of exceptions.

> **Core boundary: the loss is the explicit objective; compression is a commonly taken economical path under structured data and a finite network language.**

## 2. Complexity is a network-relative profile

Combining the architecture and parameterization, the input-output encoding, the reference parameter measure, and the per-sample loss into one object, we call it the neural reference protocol:

$$
\Pi=(\mathcal A,\varphi,\mu,\ell).
$$

For a complete target function $f$, define the parameter volume below the loss threshold $\epsilon$:

$$
V^{\mathrm{full}}_\Pi(f;\epsilon)
=
\mu_\Pi\{\theta:L_{D_{\mathrm{full}}(f)}(\theta)\le\epsilon\}.
$$

The Neural K-profile is defined as:

$$
K^N_\Pi(f;\epsilon)
=
-\log_2V^{\mathrm{full}}_\Pi(f;\epsilon).
$$

It is not machine-independent Kolmogorov complexity, but "how rare it is for the current neural language to implement the function $f$ to accuracy $\epsilon$". Changing the architecture, activation, width, encoding, initialization scale, or loss all changes this curve.

Complexity is first of all a profile: the volume at a fixed loss is one cross-section, and the local contraction rate is a slope. The hard-function prior is a separate baseline, not the high-threshold limit of the full-target curve; for finite loss, $\epsilon\to\infty$ gives $V^{\mathrm{full}}_\Pi(f;\epsilon)\to1$. In contrast, the fixed-D branch mass intersected with the target cell tends to its hard-prior mass in that limit. Different full-target curves can cross; even when their respective output signs are fixed, continuous margin and logit implementations remain subject to filtering. [E24](experiments/e24.html) provides corresponding deep-tail sampling evidence, with sampling uncertainty limiting the precise crossing location.

### An independent operational quantity: identification and recovery transition from random training sets

The Neural K-profile is a static geometric object of the full target. Actual training also lets one define a directly measurable sample complexity that **must not be surreptitiously swapped with the Neural K-profile**. To do so, fix the neural reference protocol $\Pi$, the training-set sampling distribution $q$, and then fix the complete training-audit protocol $\mathcal T$: the network capacity, initialization, optimizer, training budget, number of seeds, full probe range, and recovery thresholds must all be written down in advance.

For a complete target function $f$ and a sample size $n$, draw multiple training sets $D_j$ of size $n$ uniformly at random from $D_{\mathrm{full}}(f)$, and train multiple independent seeds on each. Define the dataset-level recovery criterion $\mathcal R_{\Pi,\mathcal T}(D_j,f)$; it must not look only at validation accuracy, but must simultaneously require:

1. the vast majority of seeds have fit the training set;
2. among the fitted seeds, the target complete function occupies sufficiently high mass;
3. the target function is the modal hard function;
4. the complete-function collision / agreement has exceeded the preregistered threshold.

The recovery rate at sample size $n$ is:

$$
\widehat r_{\Pi,\mathcal T}(f,n)
=
\frac{1}{M}
\sum_{j=1}^{M}
\mathbf 1\!\left[
\mathcal R_{\Pi,\mathcal T}(D_j,f)=1
\right].
$$

After a monotone estimate of the recovery rate, define:

$$
n_q^{\mathrm{id}}(f;\Pi,q,\mathcal T)
=
\min\left\{
n\in\mathcal G,\ n<N:
\widehat r_{\Pi,\mathcal T}^{\,\uparrow}(f,n)\ge q
\right\},
\qquad q\in\{0.5,0.9\}.
$$

Here $N=|D_{\mathrm{full}}(f)|$, and $\mathcal G$ is the preregistered grid of non-complete sample sizes. `n50` and `n90` are respectively the first crossings on the grid at which random training sets stably recover the complete rule with 50% and 90% probability. A point estimate `n_q=n_k` actually only localizes the continuous transition to the interval $(n_{k-1},n_k]$; a bootstrap interval should be reported alongside. If no crossing has occurred up to the largest non-complete sample size $N-1$, record it as `n_q>N-1` or right-censored; the complete truth-table fit at $n=N$ must not be treated as a generalization transition.

What it gives is therefore an **interval** for the grokking / rule-recovery transition on the data-amount axis. A tie between grid-adjacent rules with overlapping confidence intervals, or a one-grid-step inversion, is not enough to refute the statistical relationship. In particular, in a space like 4-bit with only 16 states, one sample already accounts for 6.25%, the holdout position and symmetries matter a great deal, and the hardest rules run directly into the full-space upper bound; in that case `n50/n90` is often unidentifiable, and one can only rely on the full profile together with local leave-one-out side evidence.

A more accurate name for this quantity is the **protocol-relative identification/recovery sample complexity**. It is measured in "number of samples" and directly answers "under the sampling distribution $q$, how many random constraints does the current training system need before it stably selects this complete rule". It contains at least four components at once: the Neural K-profile of the complete target itself, the denominator of competing extensions under a fixed partial training set, the identifiability caused by sample coverage and shortcuts, and the accessibility of the optimizer. It can therefore serve as an external criterion and empirical anchor for Neural K, but it cannot be defined as the same quantity without additional assumptions.

More explicitly, the numerator of a fixed-D static target probability is $\mu_\Pi(C_f\cap A_D)$, not the full-target volume without a bridge. Only when the full-target event is contained in this intersection can its mass be written as the branch mass times a within-cell continuous-precision acceptance rate. The random-training-set transition also involves target-dependent competitor denominators and optimizer transport. Only when these additional factors are controlled across tasks, or independent evidence supports a monotone correspondence, can the `n50/n90` ranking serve as a proxy for the full-target Neural K ranking.

### How these experiments revised the original conclusion

We initially had a very natural conjecture: **the more complex a rule the network considers it to be, the more training samples are needed for it to be stably identified.** The preregistered experiments on parity1 through parity4 matched this ordering exactly, showing that the Neural K of complete rules does have prospective predictive power, and is not just a renaming applied after seeing the results.

Subsequently, parity2 and MUX3 produced a counterexample that could not be ignored. Judged by complete-rule volume, MUX3 is easier; but when trained with ordinary random samples, it instead needs more data. This forced us to admit: knowing only how hard the target rule itself is, is not enough.

The key missing term is **which alternative explanations the current training set still allows**. For MUX3, a large number of samples are consistent both with the true selector rule and with the shortcut of "copy one bit, then memorize a few exceptions". Although such samples are numerous, they do not genuinely help the network distinguish between these functions.

The first supplementary experiment changed only the sampling method. When we added exactly those samples that distinguish the true MUX3 from the copy shortcuts, MUX3 immediately recovered earlier than parity2. Neither the target rule nor the network changed; what changed was only whether the samples hit the positions where the competing functions genuinely disagree.

The second supplementary experiment removed the optimizer entirely and directly observed the low-loss functions allowed by the same batch of training samples. The result was still the same: ordinary samples leave behind many wrong shortcuts, while conflict samples systematically eliminate them, letting the true MUX3 become dominant. This shows that the phenomenon above comes first from the static function competition shaped by the training set itself, and that the optimizer mainly affects the arrival speed and the specific path.

> **Revised conclusion: how much data a rule needs depends not only on how hard the rule itself is, but also on which functions are currently competing with it, and on whether the sampled data can effectively exclude those competing functions. The Neural K of the complete rule is still an important first-order predictor, but it does not equal the grokking transition point.**

The detailed numbers, the three supplementary experiments, and the numerical boundaries are all in [E23](experiments/e23.html); the theory core keeps only this revised causal relationship.

[E24](experiments/e24.html) adds another case: even when two complete mappings are already fixed, their relative volumes can still swap order as the loss continues to decrease. This shows that "rule difficulty" itself should also be read from the entire loss profile, not from a single shallow cross-section.

### Relation to MDL and algorithmic information theory

Standard MDL writes learning as a two-part codelength:

$$
L_{\mathrm{MDL}}(H,D)=L(H)+L(D\mid H).
$$

$L(H)$ is the cost of describing the hypothesis or program, and $L(D\mid H)$ is the cost of using that hypothesis to encode the not-yet-explained residual. Algorithmic probability and Kolmogorov complexity express the same principle from the other direction: short programs should receive greater prior mass.

Neural K rewrites this language into a measurable version that depends on the neural protocol. The reference measure $\mu_\Pi$ assigns mass to implementations; $-\log_2V^{\mathrm{full}}_\Pi(f;\epsilon)$ is the event information cost of reaching the specified full-target precision. The Neural K-profile can therefore be understood as a precision-dependent description-cost scale. Interpreting it further as a prefix codelength distinguishing all functions requires disjoint events or a separately constructed valid coding distribution; general high-threshold full-target events can overlap, so the coding probability budget cannot be applied directly.

This does not mean that the training program explicitly minimizes "loss plus program length" at the same time. Training code usually only lowers the empirical loss; MDL-style selection comes from the mass differences jointly produced by the architecture's reference measure and the loss sublevel geometry. Nor is it machine-independent Kolmogorov complexity, because changing the network, encoding, parameterization, or reference measure changes the codelength. There is currently no neural version of a coding theorem proving the two equivalent.

What is falsifiable here is not just a new name for volume: [E22](experiments/e22.html) strictly accumulates per-sample generalized surprise into an order-independent endpoint codelength, and [E23](experiments/e23.html) used a pre-frozen full-target profile to predict the `n50/n90` of independent random training sets. This connects parameter mass, predictive coding, and identification sample complexity into one empirical chain.

## 3. Static geometry and optimization dynamics must be separated

Fixing a training set $D$ and a loss threshold $\epsilon$, the static mass of a complete function is:

$$
\Omega_{\Pi,D}(f;\epsilon)
=
\mu_\Pi\{\theta:L_D(\theta)\le\epsilon,\ h_\theta=f\}.
$$

After normalization we obtain the static function distribution:

$$
Q^{\mathrm{static}}_{\Pi,D,\epsilon}(f)
=
\frac{\Omega_{\Pi,D}(f;\epsilon)}
{\sum_g\Omega_{\Pi,D}(g;\epsilon)}.
$$

Let $F_\theta$ denote the continuous forward map, $h_\theta$ the hard readout, and $H_U(\theta)=(h_\theta(x))_{x\in U}$ the complete hard table on a fixed measurement domain. Write the actual initialization distribution as $\mu_{\rm init}$, without assuming it equals the reference measure $\mu_\Pi$. The training map $T_{\Pi,D,t}$ first produces a parameter distribution, which the hard readout then maps to a function distribution:

$$
\nu_{\Pi,D,t}=(T_{\Pi,D,t})_\#\mu_{\rm init},
\qquad
Q^{\mathrm{opt}}_{\Pi,U,D,t}=(H_U)_\#\nu_{\Pi,D,t}.
$$

Training randomness such as random mini-batches must also be included in the protocol; it can be combined with initialization into a joint random input before taking the pushforward.

> **Key separation of layers: the static volume determines which candidate regions are large and which regions are more stable as the loss changes, and is an important first-order influence on actual training; the optimizer additionally determines the entrances, flow tubes, speeds, connectivity, numerical noise, and history. The static volume therefore cannot be called the SGD posterior directly.**

The parity scaffold shows that a low-loss endpoint can exist and be locally stable, yet be unreachable from random initialization. [E17](experiments/e17.html) shows in turn that with the same network, training set, and similar loss, SMC and AdamW can still select markedly different functions.

[E28](experiments/e28.html) is a real-task comparison with a different target distribution: on full 50k MNIST with the same small CNN, likelihood-weighted posterior HMC and plain Adam reach 99.03% and 98.95% respectively, at similar performance levels, yet their function diversity and time-averaging gains differ. The HMC target here is a Gaussian prior multiplied by the full training-data likelihood, not a conditional measure at a fixed loss threshold. Weighting changes relative mass inside the allowed region, so proximity to the hard-threshold ensemble cannot be assumed. E28 does not directly validate hard-threshold volume or the contraction law.

[E29](experiments/e29.html), in contrast, gives a matched-loss verdict for hard-threshold static SMC in a dead-bit counterfactual: the five-function average difference between decay-free Adam and static SMC is only 0.0010, while L2/MAP forms a different transport outcome by actively compressing the unconstrained weights. The experiments must be interpreted separately according to their respective target distributions; "close prediction" and "equal distribution" remain different, and the posterior-weighted comparison must not be counted as hard-threshold volume evidence.

[E30](experiments/e30.html) serves only as a supplementary clarification: grokking does not require weight decay; the raw-loss landscape without regularization can itself already support rule takeover. Weight decay written into the loss merely strengthens the rule preference within the same static landscape, and is nothing mysterious or special.

For any task, three things should be checked separately: whether the network can represent the target, whether the data are sufficient to identify the target, and whether the optimizer can reach the target.

### Decoupling, scaffolds, and optimization accessibility are not complexity

The original objective of an end-to-end task can be written as $L_{\mathrm{end}}(\theta)$. Decoupled training, intermediate supervision, or a scaffold usually adds an auxiliary objective:

$$
L_{\lambda}(\theta)
=
L_{\mathrm{end}}(\theta)
+
\lambda L_{\mathrm{aux}}(\theta),
$$

and later in training reduces $\lambda$ to zero, or removes the auxiliary inputs and intermediate labels entirely. What this operation changes first is the gradient vector field and the paths into the low-loss region:

$$
-\nabla L_{\lambda}
=
-\nabla L_{\mathrm{end}}
-
\lambda\nabla L_{\mathrm{aux}}.
$$

The original endpoint loss may give almost no useful direction near a random initialization, or the gradients of different computational stages may cancel each other out. The auxiliary objective breaks a long-range, composite, or symmetric computation into locally learnable steps, so that the network first forms intermediate representations and then gradually internalizes them into an end-to-end computation. This shows that the search path has been improved; it does not mean the Neural K of the original target function has necessarily decreased.

If, after the scaffold is removed, the network can still maintain the solution using only $L_{\mathrm{end}}$ and recover from perturbations, this shows that the original endpoint task does have a stable low-loss region supported by its own loss; the initial failure was mainly an entrance or gradient-accessibility problem. The parity scaffold—removal—perturbation-recovery of [E16](experiments/e16.html) is exactly this kind of criterion. The early decoupling experiments on mod 3, multiplication, recursion/search, and the Tower of Hanoi also provide phenomena in the same direction.

Conversely, the success of auxiliary supervision cannot by itself prove that the original function is simple; end-to-end failure cannot by itself prove that the function is complex. Four objects must be distinguished:

1. **Representational capacity**: whether parameters implementing the target exist in the network;
2. **Data identifiability**: whether the training samples are sufficient to exclude competing extensions;
3. **Protocol-relative complexity**: the target's Neural K-profile under the original neural reference protocol;
4. **Optimization accessibility**: whether, starting from the specified initialization and optimizer, the gradient can enter that region.

If the auxiliary variables, modules, or inputs are kept permanently in the final system, then the neural reference protocol $\Pi$ has already changed, and the target may indeed be simpler in the new language. If the auxiliary information is used only during training and removed entirely at the end, then it is mainly a continuation / curriculum path; the final complexity should still be measured under the original endpoint protocol. When comparing decoupled and end-to-end results, one must first declare which case applies.

## 4. The language of statistical physics

### 4.1 Correspondence table

| Statistical physics | Object in the neural network |
|---|---|
| Microstate | A specific set of parameters $\theta$ |
| Macrostate | A complete hard function, or a further subdivided function-margin/representation cell |
| Energy | The total loss on the training set |
| Density of states | How many parameter microstates there are near a given energy |
| Entropy | The logarithm of the density of states or of the cumulative volume |
| Inverse temperature $\beta$ | A Gibbs reweighting parameter; neither a physical temperature nor the hard loss threshold |
| Free energy | The combined cost between the energy requirement and the parameter entropy |
| Phase transition / change of phase | The advantage of different function macrostates changes hands with the amount of data, temperature, or loss depth |

To preserve additivity over samples, write the total energy as the sum of per-sample losses:

$$
E_D(\theta)=\sum_{i=1}^{n}\ell(F_\theta(x_i),y_i)=nL_D(\theta).
$$

First define an energy measure for function $f$, retaining any atoms:

$$
\Xi_{D,f}(B)
=\mu_\Pi\{\theta:H_U(\theta)=f,\ E_D(\theta)\in B\}.
$$

The cumulative-threshold view retains mass below a specified average loss:

$$
\Omega_{D,f}(\epsilon)
=\Xi_{D,f}([0,n\epsilon]).
$$

Only when this energy measure has an ordinary density do we write $d\Xi_{D,f}(E)=\rho_{D,f}(E)dE$ and the corresponding integral. A cumulative-threshold ensemble is also distinct from a fixed-energy thin shell.

Its logarithm can be regarded as the cumulative parameter entropy. A function that has more implementations at the same loss has higher parameter entropy and greater static mass.

### 4.2 Canonical ensemble, partition function, and free energy

Given an inverse temperature $\beta$:

$$
Z_D(\beta)
=
\int e^{-\beta E_D(\theta)}d\mu_\Pi(\theta),
$$

the partition function restricted to a function is:

$$
Z_{D,f}(\beta)
=
\int \mathbf 1[h_\theta=f]e^{-\beta E_D(\theta)}d\mu_\Pi(\theta).
$$

The Gibbs mass of a function macrostate is then:

$$
Q_\beta(f\mid D)=\frac{Z_{D,f}(\beta)}{Z_D(\beta)}.
$$

The free energy is defined as:

$$
F_D(\beta)=-\frac1\beta\log Z_D(\beta),
\qquad
F_{D,f}(\beta)=-\frac1\beta\log Z_{D,f}(\beta).
$$

The relative odds of two functions are determined by the difference of their restricted free energies:

$$
\log\frac{Q_\beta(f\mid D)}{Q_\beta(g\mid D)}
=
-\beta\bigl(F_{D,f}-F_{D,g}\bigr).
$$

Larger $\beta$ emphasizes lower energy, but whether a function loses relative mass depends on differences in restricted free energies. Under differentiability conditions, the relative odds satisfy

$$
\partial_\beta\ln\frac{Q_\beta(f\mid D)}{Q_\beta(g\mid D)}
=-\mathbb E_{\beta,f}E_D+\mathbb E_{\beta,g}E_D.
$$

Here $\mathbb E_{\beta,f}$ denotes a Gibbs expectation restricted to cell $f$. The label "complex" alone does not determine the sign, and relative disadvantage must not be described as an individual restricted free energy increasing with $\beta$.

Cumulative loss cross-sections and canonical weighting derive from the same energy measure; the latter is its Laplace-type integral. This project has measured both formulations, but a crossing of cumulative curves does not automatically guarantee a Gibbs crossing at a corresponding $\beta$.

### 4.3 What is not being claimed physically here

Training is not a physical system that automatically reaches thermal equilibrium, and $\beta$ is not the real temperature of the GPU. Unless additional Langevin or equilibrium assumptions are made, SGD/AdamW is not a Gibbs sampler. Statistical physics here is first of all a mathematical language for describing the static density of states, the energy-entropy competition, and the changing hands of macrostates.

Grokking can appear as a rule macrostate taking over at deeper loss or larger data volume, but a smooth crossover in a finite network need not be a strict thermodynamic singularity. This page has not yet analytically derived the shape of $\rho_{D,f}(E)$ from the network structure; that is exactly the deeper question of "why the two kinds of simplicity hold", and it is currently outside the scope of this research.

## 5. How to use volume to predict an unseen sample

Given a training set $D$, an unseen input $x$, and a candidate label $y$, construct the extended training set:

$$
D_y=D\cup\{(x,y)\}.
$$

In the microcanonical formulation, the label-branch mass is:

$$
P_{\mathrm{hard}}(y\mid x,D,\epsilon)
=
\frac{\mu_\Pi\{\theta\in A_D(\epsilon):h_\theta(x)=y\}}
{\mu_\Pi(A_D(\epsilon))}.
$$

Intuitively, among all parameters that have already brought the training set to the current loss, look at how many would predict the new input as label $y$. The branch with greater mass is the current static prediction.

One can also compute the partition function of the dataset extended with each hypothesized label separately:

$$
\widetilde P_\beta(y\mid x,D)
=
\frac{Z_{D_y}(\beta)}{Z_D(\beta)},
$$

and then normalize over all candidate labels:

$$
P_\beta(y\mid x,D)
=
\frac{Z_{D_y}(\beta)}{\sum_{y'}Z_{D_{y'}}(\beta)}.
$$

When the per-sample loss is the standard negative log-likelihood and $\beta=1$, this is the ordinary posterior predictive evidence. More generally, it is still a protocol-relative Gibbs branch score.

> **Prediction principle: if the volume corresponding to a label contracts more slowly as the loss deepens, it means that it retains more compatible implementations under increasingly strict training accuracy, and therefore the prediction tendency strengthens.**

The volume curves of different labels can still cross, so a prediction must declare which loss depth or $\beta$ it uses. MNIST [E25](experiments/e25.html) directly tested this kind of static branch prediction.

[E26](experiments/e26.html) pushes the same principle to the level of label combinations: given only one anchor and the 5:5 ratio, the natural MNIST 0/1 split rises in both blinded panels from last place at shallow loss to top-1 at deep loss, and occupies about 99.97% of the normalized volume mass at loss 0.3. It supports the claim that "volume can predict unseen label combinations", but it does not amount to fully unsupervised clustering with the class-ratio constraint removed.

## 6. Surprise, information gain, and sample-order invariants

### 6.1 Surprise of an observed label

For an already observed label $y$, its predictive surprise is:

$$
s(y\mid x,D)=-\log_2P(y\mid x,D).
$$

Define the dataset state cost:

$$
C(D)=-\log_2Z(D).
$$

The generalized cost increment of adding one sample is defined as the endpoint cost difference:

$$
\Delta C_t=C(D_t)-C(D_{t-1})
=
-\log_2\frac{Z(D_t)}{Z(D_{t-1})}.
$$

Summing along any sample order:

$$
\sum_{t=1}^{m}\Delta C_t
=
C(D_m)-C(D_0).
$$

The intermediate terms telescope away, so the total cost depends only on the starting point and the final training set, not on sample order. **Each cost increment still depends on order; it is the sum that is invariant.** [E22](experiments/e22.html) numerically checked closure for 256 rules and all 40,320 orders per rule. The ratio directly gives the corresponding label predictive probability only under coherent hard conditioning, or standard negative log likelihood with $\beta=1$ on total loss. A general Gibbs increment is not actual predictive surprise without accounting for label normalization.

This invariant belongs to the same static partition function; it does not represent conservation along SGD trajectories.

### 6.2 Under hard conditioning: the actual surprise is exactly equal to the information gain of the function distribution

The probabilities here do not require the samples themselves to come from a random generator. Let the current complete function be the random variable

$$
F\sim Q_D(f),
$$

and the label at a candidate input $x$ is just a projection of this function random variable:

$$
Y_x=F(x),
\qquad
P_D(y\mid x)
=
\sum_fQ_D(f)\mathbf 1[f(x)=y].
$$

After observing the label $y$ and performing pure hard conditioning, the new function distribution is:

$$
Q_{D'}(f)
=
\frac{Q_D(f)\mathbf 1[f(x)=y]}
{P_D(y\mid x)},
\qquad
D'=D\cup\{(x,y)\}.
$$

Therefore, when the KL is computed in base 2, there is the exact identity:

$$
D_{\mathrm{KL},2}\!\left(Q_{D'}\Vert Q_D\right)
=
-\log_2P_D(y\mid x)
=
s(y\mid x,D).
$$

> **Exact hard-conditioning theorem: the more improbable the true label is under the current function distribution, the more candidate-function mass it deletes; the predictive surprise of an observed label is exactly equal to the information gain of the function posterior update.**

This is the concretization of [Itti and Baldi's Bayesian surprise](https://pmc.ncbi.nlm.nih.gov/articles/PMC2782645/) in the space of complete functions: what changes is not the "sample distribution", but our belief distribution over which complete function can still hold. The identity requires a coherent hard update on the same static $Q_D$; retraining a multi-seed optimizer endpoint distribution does not automatically satisfy this Bayes formula.

### 6.3 The actual surprise is not equal to the expected information gain when the label is unknown

Surprise is the actual coding cost incurred after the label has been observed. When choosing the next unlabeled input, the more relevant quantity is the expected information gain: after observing its label, how much the function or parameter distribution shrinks on average.

Under a coherent observation model and Bayesian update, the expected information gain is defined as follows; KL and entropy below use base two:

$$
\mathrm{IG}(x\mid D)
=
\mathbb E_{y\sim P(y\mid x,D)}
\left[
D_{\mathrm{KL},2}\bigl(q(\theta\mid D,x,y)\,\|\,q(\theta\mid D)\bigr)
\right]
=
I(\Theta;Y_x\mid D).
$$

The equivalent [BALD](https://arxiv.org/abs/1112.5745) form is:

$$
I(\Theta;Y_x\mid D)
=
H[Y_x\mid D]
-
\mathbb E_{\theta\sim q(\theta\mid D)}H[Y_x\mid\theta].
$$

If every hard function gives a deterministic label for $x$, the second term is zero, and the expected information gain equals the predictive entropy of the candidate-label distribution. The input closest to 50:50, with the lowest agreement, is then on average the most informative. This is the theoretical basis of the active disagreement experiment [E21](experiments/e21.html).

For binary classification, if $p=P_D(Y_x=1)$, then

$$
A(x\mid D)=p^2+(1-p)^2.
$$

So while the label is still unknown, low agreement corresponds monotonically to high predictive entropy and hence to high **expected** information gain. But once the label is observed, the reading becomes $-\log_2P_D(y\mid x)$: a label that was originally close to 50:50 is very informative on average; a label with extremely high agreement that goes against the consensus can instead produce a larger **actual** surprise. The two sampling intuitions do not conflict: they answer respectively "how much can be learned on average before the query" and "how many functions this already-observed label actually excluded".

One practical boundary must be kept: "all models answer correctly" in the hard-function posterior does not guarantee that the sample has no effect at all upon retraining. Its soft margin can still differ, and adding the sample also changes the optimization path starting from initialization.

In a soft Gibbs update, this boundary can be written more explicitly. If

$$
q'_{D}(\theta)
\propto
q_D(\theta)e^{-\beta\ell(\theta;x,y)},
$$

then the generalized cost increment in bits is

$$
\Delta C
=
-\log_2\mathbb E_{q_D}
\left[e^{-\beta\ell(\theta;x,y)}\right],
$$

and the posterior KL is

$$
D_{\mathrm{KL},2}(q'_D\Vert q_D)
=
\Delta C
-
\frac{\beta}{\ln2}\,\mathbb E_{q'_D}\ell.
$$

Here $\ell$ uses natural logarithms, requiring division by $\ln2$. Under a soft loss, the generalized cost increment, posterior KL, and single-network BCE are related but unequal; pure conditioning with hard indicator weights recovers the exact identity of the previous subsection.

### 6.4 After adding a sample: convergence at the query point does not imply monotone global agreement

After hard conditioning, all surviving functions give the label $y$ at the added query point $x$, so the local agreement at that point immediately becomes 1. But at another probe point $x'$, the agreement can go up or down: the deleted function cluster may originally have sustained some global consensus, or it may have been exactly the source of the disagreement. Consequently there is no per-sample monotonicity theorem for the global average agreement, the complete-function collision, or the target-function mass.

This gives a direct explanation of "convergence—divergence—reconvergence": a high-surprise label may first destroy the previously dominant function cluster, making the remaining clusters temporarily more balanced and lowering the global agreement; subsequent informative samples then continue to delete competing clusters, and only then does the function distribution condense onto the new rule. The local KL information gain is always nonnegative; this does not mean the global agreement must rise.

The MUX3 cell/conflict experiment turned this distinction into a visible mechanism. On ordinary cells with $x_1=x_2$, the true MUX3, `copy x1`, and `copy x2` give the same label, so these samples can hardly distinguish the three; only the selector-conflict cells with $x_1\ne x_2$ separate the shortcuts from the true rule. Uniform fixed-$D$ SMC has a posterior target accuracy of 0.993 on ordinary cells and only 0.777 on conflict cells; after raising the fraction of conflict samples to 75%, the conflict-cell accuracy rises to 0.995, and the complete-MUX3 target mass rises from 0.000214 to 0.782. What this intervention increases is not the abstract "number of samples", but the efficiency with which each unit of sample cuts down the current competing-function denominator.

## 7. The agreement conjecture

For an input $x$, if the candidate-label probabilities are $P(y\mid x,D)$, the pairwise agreement — the probability that two independent function draws give the same label — is:

$$
A(x\mid D)=\sum_yP(y\mid x,D)^2.
$$

For the complete-function distribution $Q(f\mid D)$, the complete-function collision is:

$$
\operatorname{Coll}(D)=\sum_fQ(f\mid D)^2.
$$

High average per-point agreement does not guarantee high complete-function collision; many functions can disagree at only a few distinct positions. In applications one must therefore report, at the same time, the per-point agreement on representative probes, the modal mass of the complete function, and the collision or Hamming-ball mass.

> **The key advantage of agreement is that it does not require knowing the external generating rule. Per-point agreement measures consistency at the corresponding inputs; only complete-function collision or modal mass close to one supports stable selection of a complete mapping under the current protocol. In the early rule experiments that motivated the conjecture, insufficient samples produced greater disagreement, while sufficient samples with generalization produced less disagreement. These observations motivated the conjecture that stable condensation may indicate an extractable rule, but per-point agreement alone proves neither complete-function condensation nor data sufficiency.**

[E27](experiments/e27.html) directly checked the temporal ordering that this intuition most easily confuses. At the four data levels of mod97, the unseen-input agreement at hard fit was only 0.027--0.033; it did not approach 1 just because the training set had been memorized. After entering grokking, agreement rose mainly as different seeds jointly recovered the correct target, and the remaining errors were still mostly seed-specific. The current evidence therefore does not support "a shared complete wrong function had already formed before grokking", nor was a second agreement jump independent of accuracy observed.

High agreement does not guarantee that the network has recovered the external generating rule presupposed by the researcher. A single sample can make the network converge unanimously to some constant extension; but a constant function is itself an extremely short and human-readable rule, so this is not a counterexample to the conjecture that "high agreement corresponds to a readable rule". It only shows that the conjecture must be teacher-free: agreement points to the rule jointly selected by the training set and the network protocol, and does not guarantee that this rule equals the generator the researcher hid behind the data. Our conjecture is more specific:

> In a sufficiently large problem space, if a nontrivial training set can make the **complete-function distribution** converge to close to 1 under a strict fresh-seed audit, then that training set usually corresponds to a logical rule that can be extracted into a short, human-readable form. Discovering the rule itself can still be very difficult.

Needing more samples to reach high agreement usually means that the identifiable rule is more complex, or that the competing extensions are harder to exclude. In this direction, the number of samples needed for agreement convergence can be used as another operational measure of protocol-relative complexity.

The conjecture has already passed two rounds of preliminary stress tests: the strict high-consensus endpoints of [E18](experiments/e18.html) all fell into the signed threshold family; after [E21](experiments/e21.html) pushed up the required sample size with anti-consensus prefixes, the endpoints expanded from linear thresholds to more complex but still readable quadratic polynomial thresholds. Both results support the conjecture, but they have not yet proved that "all high-agreement functions are necessarily human-readable".

This conjecture also has a deeper implication: if high agreement still systematically corresponds to human-readable short rules across more tasks, architectures, and larger problem spaces, then the complexity ordering induced by neural networks and the complexity ordering in human intuition are not unrelated to each other. The two may both be exploiting the same batch of compressible structures, such as locality, symmetry, compositionality, low-order interactions, and computational reuse. The early hundred-plus deterministic-rule experiments also provide breadth side-evidence in the same direction: when the target has an exact, reusable generating rule, and the data and optimization accessibility are sufficient, the network can often simultaneously press both the unseen-sample loss and the cross-seed function disagreement very low; underconstrained or noisy conditions do not automatically produce this kind of complete condensation.

> **If this direction continues to withstand stress tests, what it suggests is not a unique, absolute "universal encoding", but a class of robust compression structures that are relatively economical in many effective representational languages. Neural networks and humans may partly converge to similar complexity judgments because they face the same structures and finite computational resources.**

This is still a strong conjecture, not a conclusion. The architecture dependence and counterexamples such as parity have already shown that the two kinds of complexity do not coincide completely; nor did all of the hundred-plus rule experiments adopt the strict fresh-seed symbolic audit of E18/E21, so they can only serve as supporting phenomena and cannot replace cross-architecture preregistered tests.

Potential applications include rule discovery and data-sufficiency diagnosis. When complete-function collision or modal mass is close to one and passes fresh-seed and independent-input audits, one can attempt to extract the rule stably selected by that protocol. This does not guarantee that it is the unique extension logically allowed by the data or the external true rule. When only per-point agreement is high, residual disagreement must be checked before claiming a unique complete rule.

## 8. Why symbolic concepts may emerge

The OOD rule-composition experiments show that the network can interpret a rule code or role bit as reusable semantics and keep executing on combinations that never appeared in training. The shared-intermediate-computation experiment [E08](experiments/e08.html) further shows that when multiple tasks need the same expensive intermediate state, reusing the representation can reach lower loss with less capacity and fewer samples.

Symbols and concepts may therefore not be things imposed from outside a connectionist system, but economical macro-representations under multi-task compression pressure. A stable symbol can bind different roles, compose different rules, and avoid storing an entire computation anew for every task.

The current evidence is mainly behavioral and computational-economy evidence. It does not directly prove that there exist unique, discrete variables in the hidden layers corresponding one-to-one with human symbols, nor does it prove that all semantics are formed by the same mechanism.

## 9. Relation to existing AGI and deep-learning theories

### 9.1 A common mathematical core: the Gibbs variational principle

Let $\mu_\Pi$ be the architecture-induced reference measure and $E_D(\theta)$ the training-data energy; then:

$$
\Phi_{\Pi,\beta}(D)
=
-\log Z_{\Pi,\beta}(D)
=
\min_q
\left[
\beta\,\mathbb E_qE_D(\theta)
+
D_{\mathrm{KL}}(q\Vert\mu_\Pi)
\right].
$$

The first term requires explaining the data and lowering the loss; the second term measures the complexity that a candidate distribution $q$ must pay for deviating from the architecture's reference measure. This "data fit + description deviation" structure is the mathematical core genuinely shared by this framework with several AGI, cognitive, and statistical learning theories.

Decomposing the parameter space further by the complete function $f$ gives the function-restricted partition functions and free energies. This step is the resolution this project cares about most: not only asking about the overall evidence, but asking how each complete-function macrostate competes as the loss, the samples, and the architecture vary.

### 9.2 Solomonoff induction and AIXI: the resource-bounded counterpart of the universal prior

[Solomonoff induction](https://mlanthology.org/misc/1964/solomonoff1964misc-formal/) gives short programs higher prior weight by program length. [Hutter's AIXI](https://arxiv.org/abs/cs/0701125) further combines universal sequence induction with action and reward maximization, forming an uncomputable ideal universal agent.

The counterpart in this framework is:

$$
P_\Pi(f)\propto e^{-C_\Pi(f)},
\qquad
C_\Pi(f)=-\log P_\Pi(f),
$$

where $\Pi$ is not a universal Turing machine, but a concrete, finite, resource-bounded neural-network protocol. It can therefore be measured through sampling and training experiments, but it does not have the universal optimality of the Solomonoff prior or AIXI.

The most accurate connection is: a neural network can be regarded as a **computable, resource-bounded, architecture-relative inducer**. The current theory only describes the selection of functions from data; it has not yet incorporated state, action, future history, and reward, so it is not yet an agent theory in the AIXI sense.

### 9.3 Schmidhuber's compression progress: from static compression to curiosity

[Schmidhuber's compression progress theory](https://people.idsia.ch/~juergen/driven2008.pdf) holds that what genuinely generates the curiosity reward is not how compressible an object already is, but how much progress the compressor has made in its ability to compress the data. An interesting experience should contain both surprise and new regularities that can be learned.

This framework can already measure the free-energy increment brought by one sample:

$$
\Delta\Phi(z\mid D)
=
\Phi(D\cup\{z\})-\Phi(D).
$$

If the sample was already predicted by the current function ensemble, the increment is small; if it excludes a large number of candidate functions, the increment is large; if learning it also makes a large number of future samples no longer surprising, genuine compression progress is produced.

E21's disagreement queries and active sample supplementation can be regarded as a passive-learning version of this idea: first find the samples currently most promising for changing the function distribution. To become a curiosity agent, the system would also need to actively choose environmental actions and take future compression progress as an intrinsic reward.

### 9.4 Free Energy Principle, predictive coding, and active inference

[Friston and Kiebel's (2009) predictive coding / Free Energy Principle](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/) writes perception as approximate Bayesian inversion of the hidden causes of the world, and gives a computable objective via the variational free energy. The two frameworks share the "prediction error/energy + posterior deviation/complexity" structure, but the meanings of the variables differ:

| This framework | Free Energy Principle / Active Inference |
|---|---|
| $q(\theta)$ is a distribution over parameter or program implementations | $q(z)$ is an approximate posterior over hidden environmental causes |
| The data are training samples | The data are sensory observations |
| The function free energy measures the quality of rule implementations | The variational free energy measures how well the internal generative model explains observations |
| disagreement finds high-information training samples | prediction error drives perceptual updating |
| There is currently no action loop | active inference changes future observations through action |

This project's $-\log Z$ is the evidence under a given reference measure; FEP usually optimizes a variational upper bound of the evidence. After introducing an approximate distribution $q$, the two are connected by the Gibbs variational formula above.

This is only a mathematical and functional-level connection; it does not prove the biological version of the Free Energy Principle, nor does it prove that the brain works with the function macrostates measured on this site.

### 9.5 PAC-Bayes and the Gibbs posterior

[Catoni (2007)](https://arxiv.org/abs/0712.0248) directly connects the empirical risk with generalization bounds using a temperature, the Gibbs posterior, and relative entropy. The standard form shared with this framework is:

$$
q_\beta(\theta\mid D)
\propto
\mu_\Pi(\theta)e^{-\beta E_D(\theta)}.
$$

PAC-Bayes mainly asks how, given the posterior, the KL complexity and the empirical risk control the test risk. This project further decomposes $q$ by the complete function $f$, measures each function's $Z_f(\beta)$, ranking inversions, and loss profiles, and compares them directly with real optimizer distributions.

The two therefore share the Gibbs mathematics, but the resolution and goals of the research differ. PAC-Bayes bounds cannot automatically yield a particular Neural K-profile, nor can they prove that SGD samples the Gibbs posterior without bias.

### 9.6 Singular learning theory

[Watanabe's singular learning theory](https://doi.org/10.1017/CBO9780511800474) replaces ordinary parameter counting with the RLCT to describe the Bayesian evidence and free-energy asymptotics of singular models such as neural networks. [WBIC](https://jmlr.csail.mit.edu/papers/volume14/watanabe13a/watanabe13a.pdf) and the later [Local Learning Coefficient](https://proceedings.mlr.press/v258/lau25a.html) bring this kind of complexity closer to an estimable object.

The most direct candidate correspondence is:

$$
\text{RLCT / LLC}
\quad\longleftrightarrow\quad
\text{local or asymptotic contraction exponent of the low-loss volume}.
$$

But SLT mainly studies large-sample Bayesian asymptotics and local singular structure; this project measures finite data, finite networks, specific complete-function macrostates, profile crossings, and optimizer accessibility. SLT may become a tool for rigorously deriving volume slopes in the future; at present it cannot directly replace these experiments.

### 9.7 Classical statistical mechanics of learning, flat minima, and local entropy

[Seung, Sompolinsky, and Tishby (1992)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.6056) already used the Gibbs ensemble to study sample size, temperature, generalization curves, and learning phase transitions; the classical work focused mainly on the teacher–student perceptron, large-system averages, and a small number of analytically tractable order parameters.

[Flat Minima](https://doi.org/10.1162/neco.1997.9.1.1) and [Entropy-SGD](https://arxiv.org/abs/1611.01838) focus on the width or local entropy around a single parameter solution. This framework instead merges the parameter space into macrostates by complete function, asking how many implementations of the same function there are in the entire parameter space, and how its total volume contracts with the loss.

Function volume is therefore more coarse-grained than a single minimum, and closer to program selection; but it still depends on the reference measure and parameterization, and is not a coordinate-independent absolute complexity.

### 9.8 Information bottleneck and MDL probing

[Information Bottleneck](https://arxiv.org/abs/physics/0004057) compresses information about the input that is irrelevant in a representation variable while retaining the information relevant to the target. It is adjacent to the "compression is intelligence" language, but what it compresses is the mutual information of random variables, not the parameter volume of complete functions; [Saxe et al. (2018)](https://openreview.net/forum?id=ry_WPG-A-) and [Kolchinsky, Tracey, and Van Kuyk (2018)](https://arxiv.org/abs/1808.07593) further pointed out the degeneracy of the deep-net "compression phase" and the measurement boundaries in deterministic tasks.

[Voita and Titov's (2020) MDL probing](https://aclanthology.org/2020.emnlp-main.14/) asks not only whether a layer can linearly predict the labels, but how much online codelength is needed to learn the labels given the representation. This is highly adjacent to the idea of using hidden-layer representations as inputs and measuring the free energy of the remaining label mapping.

Both can become auxiliary measurements at the representation level, but one cannot draw a direct equality sign among probe codelength, mutual information, parameter norm, Neural K, and machine-independent Kolmogorov complexity.

### 9.9 Why this is still not a complete AGI theory

> **AGI positioning: what the current framework explains is how, given data and a neural protocol, a learning system induces in function space; it is currently a theory of learning and induction, not a complete AGI theory.**

A complete agent theory also needs at least:

$$
\text{state}
+
\text{action}
+
\text{future prediction}
+
\text{reward}.
$$

A possible extension, after specifying an action and observation model and a coherent Bayesian update, is to let action selection consider expected information gain and external reward:

$$
a^*
=
\arg\max_a
\left\{
I(\Theta;Y_a\mid D,a)
+\lambda\,\mathbb E[R(a,Y_a)\mid D,a]
\right\}.
$$

Here $Y_a$ is the unknown observation associated with action $a$, and mutual information is defined by the expected posterior KL. For a fixed model, $\Phi(D)-\Phi(D\cup z_a)$ is a negative information cost, not information gain; compression progress additionally requires comparing predictive or coding performance before and after learning. This candidate objective is adjacent to active learning and information acquisition with rewards, but is not the full objective of AIXI or active inference. No experiment has established this action loop; it remains an extension direction.

### 9.10 The connections this framework may genuinely add

One cannot claim to have reinvented free energy, the Gibbs posterior, MDL, AIXI, or the statistical mechanics of learning. The combination more likely attributable to this project is:

1. decomposing the parameter space by complete-function macrostates;
2. measuring each function's entire low-loss density-of-states / Neural K-profile;
3. strictly connecting per-sample surprise to the same endpoint free energy;
4. using the full-rule profile to prospectively predict the sample transition of random datasets;
5. separately treating the optimizer's deviation from the static function volume as nonequilibrium transport;
6. using agreement to measure function condensation without knowing the teacher, and testing its symbolic readability.

It is therefore more like an experimentable common coordinate system:

<div class="theory-link-chain" aria-label="Common theoretical coordinate system">
  <span>Solomonoff / MDL</span>
  <span>Bayesian evidence / PAC-Bayes</span>
  <span>statistical-physics free energy</span>
  <span>predictive coding</span>
  <span>neural-network function volume</span>
</div>

## 10. Why this is not a circular definition

Taking

$$
K^N_\Pi(f;\epsilon)=-\log_2V^{\mathrm{full}}_\Pi(f;\epsilon)
$$

and **defining** it as the protocol-relative volume complexity is of course not falsifiable by itself; it merely names a measurement result. What is genuinely falsifiable is whether this definition can predict quantities outside itself.

There are currently the following independent anchors:

1. [E10](experiments/e10.html) constructed strictly simple/complex function pairs using linear separability before the experiment, then looked at the low-loss odds, avoiding renaming the winner after the fact.
2. [E23](experiments/e23.html) first measured the complete-rule volume score and froze the hash, then predicted the `n50/n90` of independent random training sets; the parity-family ordering was hit exactly.
3. [E25](experiments/e25.html) used static branches to predict unseen MNIST labels and the NLL turning point, rather than only explaining the training objective itself.
4. [E18](experiments/e18.html) and [E21](experiments/e21.html) used an independent symbolic-complexity audit to check whether the high-agreement endpoints are human-readable.
5. [The f/g extension of E31](experiments/e31.html#update-20261027) first used logical structure to judge that g should be more complex than f; after observing the counterintuitive raw-BCE ordering, it prospectively predicted reversal under amplitude alignment. New direct sampling, increased-budget runs, and bridging support that direction.
6. [E33 external calibration](experiments/e33.html#update-20261027) checks Neural K against independent logical costs and complexity proxies, retaining shared orderings as well as robust local reversals; no single gate library is declared a universal complexity ruler for all architectures.

For `f=a XOR b` and `g=(a XOR b) OR c`, an independent strict logical criterion is available: in a fixed gate library containing XOR and OR, with at most two inputs per gate and free wiring, f has minimum cost 1 gate; g has three essential inputs, so one gate cannot implement it, while a two-gate construction exists, giving minimum cost 2 gates. This formal explanation does not depend on Neural K and is not presented as a formal preregistration of that gate library at the time. The logical ordering comes from outside the measurement, which gives the subsequent intervention its non-circular calibration role.

The reversal also exposes a limitation of mean BCE as a universal complexity ruler: it can let an easy branch lower its loss first while a difficult branch retains a looser precision allowance. The original volume remains a valid measurement, but is not thereby identical to independent logical cost. Further work may need to revise or supplement the constraint family on the joint output distribution; amplitude alignment has not been proved the uniquely correct definition.

Counterexamples have also genuinely changed the theory: the AND shortcut refuted the presupposed winner, the weighted rule-bit refuted the whole-course monotonicity of a single function, the shallow misordering of random/parity refuted the single-cross-section scalar, the deep crossing refuted the sufficiency of the hard ID, and MUX3/parity2 further refuted "the difficulty of the complete rule alone determines the sample transition". Precisely because the theory is forced to revise by these results, it is not the circular explanation that "whatever the network selects is called simple".

## 11. Data volume, loss depth, and phase regions

The same rule can exhibit four typical regions at different data volumes:

1. **Underconstrained region.** Too few samples; multiple huge extensions satisfy the training set; agreement can be very low, or it can be wrongly high on a constant or a shortcut.
2. **Critical/grokking region.** The data already allow rule identification, but the rule dominates only at deep loss; training first fits, then the rule takes over as the loss keeps decreasing.
3. **Sufficient-data/direct-learning region.** The rule overwhelms the alternative functions already at relatively shallow loss, and train and validation decrease in sync from early on; most of the early hundred-plus successful experiments belong here.
4. **Finite-precision and noise region.** Any finite training set supports only finite extrapolation accuracy; continuing to press the loss may emphasize training-set-specific residuals or wrong labels, causing NLL, accuracy, or external-rule performance to separate.

Under pure hard conditioning, adding a sample consistent with the target rule does not delete the target cell, but it does delete some competing cells, so the target's normalized hard mass does not decrease. For a fixed raw-BCE cross-section there is no equally unconditional monotonicity theorem, because a new sample changes both the average loss and the margin constraints at the same time.

Data volume, network capacity, and training time play different roles: capacity determines whether the target can be represented, data determine whether it can be identified, and training dynamics determine whether it can be reached. Mixing the three into a single "task difficulty" would conceal the real differences among parity, mod 3, grokking, and overfitting.

## 12. Current theoretical boundaries

The following questions currently have no explanation or rigorous derivation:

- why a broad range of network architectures produces a prior simplicity bias;
- why the low-loss density of states of a specific function has its current shape;
- whether the complete Neural K-profile can be analytically predicted from the network structure;
- whether the profiles of an arbitrary function pair eventually stabilize, and whether the number of crossings is finite;
- whether high complete-function agreement universally implies a human-readable rule;
- how the static volume forms a quantitative closed-form relationship with concrete SGD/AdamW transport.

This theory is therefore currently a static-dynamic layered framework supported by controlled experiments, not a completed unified law of deep learning.

> **Current empirical claim: in the protocols tested, the continuous decrease of training loss has function-selection meaning; this selection can be measured with protocol-relative function volume, free energy, prediction branches, and sample transitions.**

### 12.1 Six unresolved questions of calibration and applicability {#open-questions-20261027}

Quantities under a specified protocol can be rigorously defined, but a rigorous definition does not establish a unique ruler, a complete explanation, or universal applicability. After the October 2026 evidence update, the following questions remain open.

| Question | What can already be specified | What has not been established |
|---|---|---|
| Loss and amplitude | Specifying a joint-output event, precision coordinate, and reference measure defines its mass and finite-window slope; different positions may have different amplitudes. | Which of mean BCE, pointwise bounds, or equal-amplitude conditions provides a more general complexity ruler. |
| Reference measure | Gaussian measures can be replaced by explicitly specified spherical or other probability measures. | Independence from every measure, or identical values, orderings, and asymptotic formulas under all natural measures. |
| Memory calibration | Growth and curvature as random sample count increases can be measured within an architecture. | Necessary strict proportionality of the contraction coefficient to sample count, or equality to a universal additive information quantity; additional exploration of this question is deferred. |
| Architecture and scale | Evidence covers several finite architectures and larger parameter counts with small training sets. | The same law for every architecture, and validity when both the network and dataset become large. |
| Optimizer correspondence | Accuracy, pointwise frequencies, and other observables can be compared under specified protocols. | One static distribution simultaneously equaling distinct SGD, Adam, and STP distributions; wider-CNN discrepancies remain a boundary. |
| Function resolution | A hard function is a measurable coarse-graining of continuous outputs and can be refined using intervals or distance tubes. | A uniquely natural resolution, or invariant complexity when the resolution changes. |

STP tests whether the distinctive gradient updates of SGD/Adam are indispensable. Similar generalization and correspondence with static predictions in tested conditions support the importance of shared low-loss structure, but do not prove optimization bias irrelevant. The new memory findings likewise do not invalidate the volume definition; they limit interpreting the contraction slope directly as a bit counter with a fixed contribution per label.

### 12.2 The unified picture remains a testable working hypothesis

The retained core picture is that, for a specified architecture, reference measure, and data constraints, some continuous output-precision paths can eliminate insufficiently fitting shortcuts and increase the relative competitiveness of compatible, simpler implementations within certain windows. Existing experiments test several such paths and their predictive power for actual function selection.

The claim that every neural network has such a direction and ultimately finds an implementation that fits the data as simply as possible is a stronger universality conjecture, not a completed proof. A direction should be specifiable in advance and reusable across a class of tasks, and the complexity criterion should have independent content; selecting the desired function region after the fact does not prove such a direction exists. Static mass preference also does not guarantee optimization accessibility, a globally simplest implementation, or recovery of the true rule.

These questions can remain unresolved, but should be disclosed alongside the positive evidence. For the specific extensions and unreached conditions, see [E31](experiments/e31.html#update-20261027), [E32](experiments/e32.html#update-20261027), [E33](experiments/e33.html#update-20261027), [E36](experiments/e36.html#update-20261027), and [E37](experiments/e37.html#update-20261027).
