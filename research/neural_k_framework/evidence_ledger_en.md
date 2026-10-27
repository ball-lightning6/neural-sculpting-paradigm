# Beyond Static Conditioning: Loss-Resolved Function Volume Connects Generalization, Grokking, and Overfitting in Neural Networks

> **Document status: internal evidence ledger, not the current main manuscript.** This file preserves the complete experimental numbers, counterexamples, caveats, and earlier theory versions; for the readable research narrative see [short_paper_zh.md](short_paper_zh.md).

> Working draft, 2026-08-25. This document's first priority is to state the problem, the evidence, and the theoretical boundaries clearly; formatting, the English translation, and the submission template are deferred until the Chinese scientific content is frozen.
>
> The stable identifiers `[E01a]`, `[E01b]`, etc. in the main text link to [`experiments/README.md`](experiments/README.md). Each identifier corresponds to an uploadable experiment description, frozen scripts, raw result bundles, and conclusion boundaries.

## Abstract

Overparameterized neural networks can fit the same finite training set with a large number of different functions, yet real training usually favors only a very small subset of them. Existing function-prior theory has revealed that randomly initialized networks carry a strong simplicity bias, and has proposed that the post-training function distribution can be approximated by conditioning the initialization prior on hard consistency with the training set.

Through complete function enumeration, gradient-free constrained sequential Monte Carlo (SMC), and real training from many initializations, this document studies the continuous loss dimension that this picture omits.

We distinguish the initialization function prior, the static loss-conditioned function mass under a fixed training set, the optimizer-induced distribution, and the single-trajectory time distribution, and we define the protocol-relative loss-resolved function volume

$$
\Omega_{\Pi,D}(f;\epsilon)
=
\mu_\Pi\{\theta:L_D(\theta)\le\epsilon, h_\theta=f\}.
$$

The experiments first show that after the training set is completely fit, the complete function distribution can still undergo systematic migration far beyond sampling noise, so training cannot in general be reduced to a single hard conditioning of a fixed initialization prior. We then find that different functions, and different continuous realizations inside the same hard function, contract at different and scale-dependent rates as the loss decreases; the relative volume ranking can even reverse predictably in the deep low-loss region.

We call the complete curve

$$
K^\mathrm{N}_{\Pi}(f;\epsilon)
=
-\log_2 V^{\mathrm{full}}_\Pi(f;\epsilon)
$$

the Neural K-profile. An 8-bit volume ranking frozen before training strictly predicted the `n50/n90` ordering of the parity family on independent random training sets.

Furthermore, using only two MNIST training images per class, the static volume branch reaches about 93%--96% accuracy on unseen `0/1` images and about 77%--78% on `3/8`; its soft free-energy curve, calibrated and then frozen, also reproduces the U-shaped turn of validation negative log-likelihood in real training.

With the same network and the same four samples, the low-loss volume disadvantage of `3/8` relative to `0/1` is amplified from less than one order of magnitude in the shallow region to about 55 decimal orders of magnitude in the deep tail, in the same direction as the operational minimal sufficient sample sizes `4 vs 512`. Another fully exhaustive experiment writes per-sample generalized cost increments as differences of the bit-valued state function $J=-\log_2Z$, and verifies across all sample orders of the 256 rules that their cumulative value is strictly equal to the complete-rule endpoint cost; Section 3.13 gives the free-energy conversion and the normalization conditions for predictive surprise.

These results support a layered picture: the architecture and the parameter measure provide a nonuniform function prior, the data and the continuous loss define a function-resolved static geometry, and the optimizer applies path-dependent transport on top of that geometry. Static volume has strong first-order predictive power, but it is not identical to the real optimizer distribution.

This document does not claim to have computed machine-independent Kolmogorov complexity, nor does it attribute all generalization to a single static mechanism; Neural K is a complexity object under a fixed neural reference protocol — measurable, but dependent on architecture, encoding, prior, and loss resolution.

## 1. Introduction

A training set usually cannot uniquely determine its complete extension on unseen inputs. For a neural network with sufficient capacity, many mutually different functions can all achieve zero training classification error, and can even drive the training loss arbitrarily small. What truly needs explaining is therefore not only whether the network can fit the training data, but:

> Among the large number of training-set-consistent functions, why does the neural network choose the current extension? Does this choice stop after interpolation? What roles do data size, loss depth, and optimization path each play?

This document adopts a training-set-centered starting point. In standard empirical-risk training, the explicit label signal the network receives is the training objective; the external generation script, the validation set, the "true rule" in the researcher's mind, semantic invariances, and generalization accuracy do not automatically enter the optimization objective.

More precisely, optimizer updates are determined by the training objective and its regularization terms, with the architecture and initialization imposing implicit biases on top; so-called generalization is the result that these low-training-loss solutions happen to perform well on unseen distributions, not the result of the network having a separate "pursue generalization" objective written into the program. This document therefore takes the training-loss geometry as the primitive object, and then asks when and why it aligns with an external rule.

Along the same line, this document does not call the data generator known to the researcher a "true function" endogenous to the training set. Even if the samples were indeed generated by a program `g*`, a finite training set only defines a set of compatible extensions; when there are very few samples, the pattern coverage is unbalanced, or the complete space is very small (for example `4-bit -> 1-bit`), another constant, threshold, shortcut, or exception function may be the explanation the network prefers under the current data.

`g*` can serve as an external evaluation benchmark, but it must not be secretly granted prior privilege when the training mechanism is analyzed. This document therefore uses "generator-aligned function" for the extension consistent with the generator, and calls the other compatible functions competing extensions, rather than calling them spurious rules or wrong rules in advance.

Random parameters induce a highly nonuniform function prior through the network architecture. [Dingle et al. (2018)](https://www.nature.com/articles/s41467-018-03101-6), [Valle-Pérez et al. (2019)](https://arxiv.org/abs/1805.08522), [Mingard et al. (2019)](https://arxiv.org/abs/1909.11522), and [Mingard et al. (2021)](https://www.jmlr.org/papers/v22/20-676.html) systematically showed that simple functions typically occupy a larger share of the initialization parameter mass, and that once the initialization prior is conditioned on the training data, it can predict many first-order statistics of the real trained-function distribution and of generalization error. This result provides an important static foundation for understanding the simplicity bias of neural networks.

However, hard-consistency conditioning places all parameters that satisfy the training labels into a single fixed event. Once training classification error reaches zero, as long as the training set no longer changes, the model itself no longer provides a new time direction. This stands in direct tension with grokking, post-interpolation complete-function migration, continued margin growth, and ongoing representation reorganization.

Even if the final function distribution is highly correlated with some static prior, this cannot identify whether real training samples from a static set or systematically transports parameter mass into new function regions through gradient flow.

The broad rule experiments add a more direct quantitative pressure: training samples cover only a tiny fraction of a huge input space, yet the network often converges stably across initializations to the generating rule. A purely static explanation must hold that the **total conditional prior mass** of all extensions that equally satisfy the training set but deviate from the generator is still far smaller than that of the target rule. This requirement looks extreme, but it cannot be ruled out by intuition alone: a single exception function can be very rare, while the number of candidate functions can grow exponentially; the total mass of the two must actually be measured.

This document therefore does not take "severe underconstraint yet generalization" itself as a logical counterexample to the static theory, but turns it into testable predictions function by function and loss by loss.

This document starts from this experimentally decidable conflict. We do not try to directly compute the uncomputable absolute Kolmogorov complexity; instead we lower the object of study to function mass measurable on finite networks: with the architecture, parameterization, input encoding, parameter measure, and loss fixed, we progressively tighten the training loss and observe how the parameter mass of complete functions and of their internal realizations changes. This perspective leads to three questions:

1. Is the post-training function distribution merely a hard conditioning of the initialization prior?
2. Does the low-loss mass of different functions contract proportionally by one common factor?
3. Can static volume curves predict independent sample complexity, unseen-sample labels, and overfitting turns?

This document gives four main groups of results.

**First, the strong hard-conditioning picture fails.** In enumerable tiny Boolean spaces, the real trained-function distribution keeps moving away from the fixed conditional prior after the first fit, with function-rank reversals and training-history dependence; in the 7-bit Boolean setting used by [Mingard et al. (2025)](https://www.nature.com/articles/s41467-024-54813-x), systematic migration of the complete 128-bit function distribution is also observed once their first-zero-error stopping point is extended.

**Second, loss depth carries function-selection meaning.** Full-target SMC and fixed-training-set hard-cell experiments jointly show that function mass cannot in general be written as `function constant × common loss factor`. Different functions have different local contraction rates that vary with scale; parity4 and a single-exception variant undergo a deep-tail absolute-volume reversal long after their hard functions have been fixed, and the crossing location can be extrapolated in advance from the shallower curves.

**Third, static volume can predict data transitions, but it is not a single-variable definition of the transition point.** For 8-bit parity1--4, the full-rule volume ordering frozen before reading any training results strictly predicted `n50=24/64/96/160` and `n90=48/80/112/160` in independent random-training-set experiments. Deeper profiles resolved part of the shallow-region misordering; MUX3/parity2 further proved that, when crossing function families, one must also account for fixed-$D$ competing extensions and whether samples can hit the positions where they disagree.

**Fourth, the measurement generalizes to real images.** In MNIST binary classification, the static hard-cell/soft-likelihood branches induced by the same fixed training set can predict hundreds of unseen images. As the training-loss section goes deeper, the predicted distribution keeps concentrating; on the hard task, after accuracy approaches a plateau, the wrong samples still become more confident, thereby forming a validation NLL U-shape approximately aligned with the real optimizer.

These results do not support mystifying SGD as the sole source detached from static geometry, nor do they support reducing it to a transparent sampler of a static posterior. The layered description that fits the data better is: the initialization prior determines the zeroth-order function mass, the continuous loss defines a static complexity flow, and the optimizer determines the entry, the path, and the actual transport.

## 2. Problem Setup and Measurable Objects

### 2.1 Neural Reference Protocol

Denote the complete experimental protocol by

$$
\Pi=(\mathcal A,\varphi,\mu,\ell),
$$

where `A` is the network architecture and parameterization, `varphi` is the input-output encoding, `mu` is the reference probability measure on parameter space, and `ell` is the per-sample loss. Write the continuous forward output as $F_\theta(x)$ and the hard output under a fixed readout rule as $h_\theta(x)=R(F_\theta(x))$. Given a training set

$$
D=\{(x_i,y_i)\}_{i=1}^{n},
\qquad
L_D(\theta)=\frac1n\sum_i\ell(F_\theta(x_i),y_i),
$$

define the loss sublevel set

$$
A_D(\epsilon)=\{\theta:L_D(\theta)\le\epsilon\}.
$$

This definition deliberately excludes validation or test error. If the training protocol also includes explicit regularization such as weight decay, the corresponding terms should be written into the complete training objective; but in any case, the network does not know the researcher-assumed generator semantics out of thin air. All "correct rule", "generalization", and "OOD invariance" statements in this document are evaluation quantities outside the training objective, and are not smuggled into the static volume definition.

Formally, if only hard consistency is considered, what the training set defines is an equivalence class

$$
\mathcal F(D)=\{f:f(x_i)=y_i,\ \forall(x_i,y_i)\in D\},
$$

rather than a unique member of it. The generator `g*` only guarantees `g* in F(D)`. Empirical sample frequencies also change the weights of different constraint directions through the averaged loss, so sample imbalance not only reduces information but also directly changes the loss geometry. `generator alignment(f,g*)` is defined only when comparing on an external probe or data distribution; it cannot replace `Omega_D(f;epsilon)` itself.

All "volumes" in this document are relative to an explicit `mu`; they are not coordinate-independent Euclidean volumes. Changing the network width or depth, the activation, the input encoding, the initialization scale, or the parameter prior all change the reference protocol and the measured complexity.

### 2.2 Function-Resolved Mass Under a Fixed Training Set

On a finite probe/complete input domain $U$, let $H_U(\theta)=(h_\theta(x))_{x\in U}$, identify function $f$ by this hard output table, and write $C_f=\{\theta:H_U(\theta)=f\}$. Below, $h_\theta=f$ always means equality of hard tables. Under a fixed training set, the joint parameter mass of a candidate function is

$$
\Omega_{\Pi,D}(f;\epsilon)
=
\mu_\Pi\{\theta:L_D(\theta)\le\epsilon, h_\theta=f\}.
$$

After normalization this gives the static loss-conditioned function distribution

$$
Q^{\mathrm{static}}_{\Pi,D,\epsilon}(f)
=
\frac{\Omega_{\Pi,D}(f;\epsilon)}
{\sum_g\Omega_{\Pi,D}(g;\epsilon)}.
$$

If there exists a common factor independent of the function,

$$
\Omega_{\Pi,D}(f;\epsilon)=A_{\Pi,D}(f)B_{\Pi,D}(\epsilon),
$$

then the relative odds of any pair of functions are independent of `epsilon`, and continuing to lower the loss after interpolation carries no additional function-selection meaning. The core static question of this document is to test this separability.

### 2.3 Full-Target Volume and the Neural K-profile

For a rule with a finite full target set `D_full(f)`, define

$$
V^{\mathrm{full}}_\Pi(f;\epsilon)
=
\mu_\Pi\{\theta:L_{D_{\mathrm{full}}(f)}(\theta)\le\epsilon\}.
$$

Further define the protocol-relative

$$
K^\mathrm N_\Pi(f;\epsilon)
=
-\log_2V^{\mathrm{full}}_\Pi(f;\epsilon).
$$

We call the entire curve the Neural K-profile; its first derivative with respect to `s=-log epsilon` describes the local volume contraction rate, and its second derivative describes how the contraction rate changes with scale. This definition has a direct formal connection to algorithmic probability and description length, but it is not the machine-independent `K_U(f)`, and there is not yet a neural coding theorem.

`V_full` and `Omega_D` under a fixed partial training set are not the same quantity. Write the full-target event as $A_{\mathrm{full}}(f;\epsilon_{\mathrm{full}})$. Only when

$$
A_{\mathrm{full}}(f;\epsilon_{\mathrm{full}})
\subseteq A_D(\epsilon_D)\cap C_f
$$

and the conditioning event has positive mass do we have

$$
V^{\mathrm{full}}_\Pi(f;\epsilon_{\mathrm{full}})
=\Omega_{\Pi,D}(f;\epsilon_D)
\Pr\!\left(A_{\mathrm{full}}(f;\epsilon_{\mathrm{full}})
\mid A_D(\epsilon_D)\cap C_f\right).
$$

For nonnegative per-sample losses and $D\subseteq D_{\mathrm{full}}(f)$, an equal total-loss budget $n\epsilon_D=N\epsilon_{\mathrm{full}}$ guarantees containment in the parent training event, where $N$ is the full-target sample count; containment in the target hard cell must be established separately. Hard-exactness alone is insufficient. Without nesting, the intersection or mass outside the parent event must be retained, and full-target mass cannot directly replace the fixed-$D$ numerator. This document checks these set relations and numerical closure through leave-one-out, the AND four-candidate conditional event, and the hard-margin bridge.

The hard prior $p_0(f)=\mu_\Pi(C_f)$ is a separate baseline. For a positive-mass cell, $\Omega_{\Pi,D}(f;\epsilon)=p_0(f)\Pr(A_D(\epsilon)\mid C_f)$; for an almost-everywhere finite nonnegative loss, full-target mass tends to 1 in the high-threshold limit, not generally to $p_0(f)$. Full-target events can also overlap, so negative log mass is first an event information cost; a function prefix-code-length interpretation requires a disjoint partition, suitable normalization, or an event family satisfying the corresponding conditions.

### 2.4 The Optimizer-Induced Distribution Is Not Static Volume

Given an initialization distribution $\mu_{\mathrm{init}}$ and a training algorithm, real training at time `t` first induces a parameter distribution and then a function distribution:

$$
\nu_{\Pi,D,t}=(T_{\Pi,D,t})_{\#}\mu_{\mathrm{init}},
\qquad
Q^{\mathrm{opt}}_{\Pi,D,t}=(H_U)_{\#}\nu_{\Pi,D,t},
$$

Here `T` outputs trained parameters, `H_U` reads the hard function, and `#` denotes pushforward. If training is stochastic, include random seeds and other randomness in the initial joint distribution and training protocol; $\mu_{\mathrm{init}}$ need not equal the static reference measure $\mu_\Pi$. `Qopt` can be strongly influenced by the static mass, but it also depends on entry, connectivity, curvature, optimizer state, batch noise, and history. This document always measures `Qstatic` and `Qopt` separately.

### 2.5 Volume Branches of Per-Sample Labels

For a fixed training set, loss section, and unseen input `x`, the hard label decomposes the parent event exactly as

$$
P_{\mathrm{hard}}(y\mid x,D,\epsilon)
=
\frac{\mu_\Pi\{\theta\in A_D(\epsilon):h_\theta(x)=y\}}
{\mu_\Pi(A_D(\epsilon))}.
$$

For a binary-classification logit, one can also define the soft branch

$$
P_{\mathrm{soft}}(y\mid x,D,\epsilon)
=
\mathbb E_{\theta\sim\mu_\Pi(\cdot\mid A_D(\epsilon))}
[p_\theta(y\mid x)].
$$

Its predictive surprise is $-\log_2P_{\mathrm{soft}}(y)$ bits, or $-\ln P_{\mathrm{soft}}(y)$ nats. If the new sample enters as a standard likelihood factor at $\beta=1$, the latter also equals the corresponding free-energy increment; general Gibbs weights require scale conversion and normalization across labels. This document evaluates both candidate labels on the same parent particles and predicts the branch with higher probability and lower predictive cost.

## 3. The Experimental Verdict Chain

Every experimental unit is recorded under the same logic: question, competing explanations, experimental design, observations, theoretical correction, and retained boundaries. The experiments are ordered by dependence; concepts formed later are not back-filled into the preregistered motivations of earlier experiments.

### 3.1 Static Hard Conditioning Receives Its First Direct Verdict

#### 3.1.1 [E01a] Function-by-Function Verdict in a Tiny Boolean Space

In the complete 256-function space of 3-bit input, 1-bit output, every network can be evaluated on all 8 inputs to obtain its exact hard function ID.

The experiment fixes a `3 -> 64 x 10 -> 1 tanh` network, `sigma_w=1`, `sigma_b=0.2`, and full-batch Adam; it uses 1,048,576 untrained networks to estimate the initialization prior, and trains 8,192 independent initializations under each of five nested training conditions. The training conditions consist of two complementary single-sample constraints and the first 2, 3, and 4 samples of the Rule30 truth table.

For each condition, the experiment saves the complete function at the first time 100% training hard accuracy is reached, and 100 and 1,000 steps afterward.

Reproduction script: [`experiment_static_prior_vs_sgd_posterior.py`](experiments/experiment_01_static_conditioning_failure/experiment_static_prior_vs_sgd_posterior.py).

The hard-conditioned prior is obtained by renormalizing, among identically constructed initialized networks, the functions that already satisfy the training labels:

$$
Q^{\mathrm{hard}}_{0,D}(f)
=
\frac{Q_0(f)\mathbf 1[f\models D]}
{\sum_g Q_0(g)\mathbf 1[g\models D]}.
$$

The difference between the real trained distribution and this static prediction is far larger than finite sampling noise. When the model first fits exactly, the function-by-function TV distance is about 0.10--0.31; after 100 further optimization steps with the training set unchanged, the distance rises to about 0.54--0.64, while the TV noise baseline from repeated sampling of the same distribution is only about 0.01--0.04. The deviation is therefore not empirical-histogram jitter caused by a huge function space or too few seeds.

Nor is the difference merely that "the trained distribution is sharper than the prior". In `rule30_k3`, the static conditional prior most prefers function 254 with mass 43.61%, while function 238 has only 17.32%; after Adam training, function 238 instead obtains 79.32% and function 254 falls to 16.53%. The optimizer changed the relative ranking of specific functions, rather than merely amplifying all simple functions in their original proportions.

The experiment also fixes the same batch of initializations and exactly the same final three-sample training set, and compares direct training, forward sample-by-sample addition, and reverse sample-by-sample addition. After reaching the final training set and continuing for another 2,000 steps, the three paths still have not fully converged: 28.20%, 19.51%, and 27.08% of the paired models in the direct/forward, direct/reverse, and forward/reverse comparisons, respectively, end with different complete functions.

The static hard-conditioned posterior depends only on the final training set and cannot express this finite-time history dependence.

What this experiment refutes is the strong proposition

$$
Q^{\mathrm{opt}}(f\mid D)
\equiv
Q^{\mathrm{prior}}(f\mid h_f\text{ fits }D).
$$

It does not refute that the initialization prior has predictive power, nor that some coarse-grained statistics near first interpolation can be approximated by the conditional prior. What is actually used here is full-batch Adam, so this document states the result as a counterexample to "this Adam protocol is merely sampling from the hard-conditioned prior", rather than inflating the old script name `SGD posterior` into an unconditional theorem about all optimizers; the path experiment likewise only proves that history still has predictive power within finite training time, and does not claim that the paths never merge after infinite time.

#### 3.1.2 [E01b] Single-Sample Function-Mass Transport

The previous experiment showed a significant distribution-level deviation, but with many dominant functions it could still be misread as a complex statistical difference. To obtain the most direct test case, we return to the early tiny Boolean configuration: `3 -> 1024 -> 1024 -> 1024 -> 1`, GELU+LayerNorm, Adam, no dropout and no weight decay.

The initialization prior is estimated from 4,096 networks; the four training conditions are `000 -> 0/1` and `111 -> 0/1`, each using the same set of 128 paired initializations. The experiment does not write "must become the constant function" into the stopping condition; it only records the first fit of the single training sample and 1 and 2 steps afterward.

Reproduction script: [`experiment_single_sample_prior_dynamics_1024.py`](experiments/experiment_01_static_conditioning_failure/experiment_single_sample_prior_dynamics_1024.py).

The initialized networks cover 186 of the 256 Boolean functions: constant 0 takes 14.01%, constant 1 takes 13.72%, and the remaining 72.27% are nonconstant functions. Given a single-sample label, the target-constant mass predicted by static hard conditioning is only 27.49%--27.99%; about 72% of the compatible mass still belongs to functions that are not constant on the other seven inputs.

Real training behaves completely differently. Under all four conditions, all 128/128 networks fit the training sample at the first optimization step, and at that same step all become the corresponding target constant function; continuing for two more steps produces no hard-function reversal. The minimum absolute logit margin over all unseen inputs remains 7.41--7.67, so this is not a discretization artifact of many logits barely crossing the zero threshold.

A stricter paired control keeps only the networks that already answered the training sample correctly at initialization but whose complete function was still nonconstant. There are 45 and 47 such seeds for targets 0 and 1, and in the end all of them, 45/45 and 47/47, migrate to the target constant. Optimization does not merely repair the training points previously answered wrong; it changes complete extensions that already satisfied all training constraints.

If the 128 runs were really independent samples from the static conditional prior, the probability that a single group all land on the target constant would be about `10^-71`. The TV distance between the actual trained distribution and the static conditional prior is about 0.72, and the JS divergence is about 0.52 bit; the hard-function entropy drops from about 5 bit to 0. This constitutes an explicit function-mass transport:

$$
\text{compatible broad function prior}
\xrightarrow{\text{one Adam step on one sample}}
\text{target-constant point mass}.
$$

One can therefore rule out the strong proposition that "all of the simplicity bias of a neural network after training already exists in the initialization function prior, and the optimizer only deletes functions that do not fit the training labels". The experiment does not rule out a more general Bayesian description: one could add a cross-entropy likelihood, a margin, or explicit training dynamics on the continuous logits; but then the object that really explains the mass rearrangement already goes beyond a one-shot static screening on hard functions.

At the same time, the hard function being stable after one step does not mean training has stopped: the logits at age 0--2 continue to evolve toward larger margins.

#### 3.1.3 First-Stage Verdict

The two groups of experiments jointly support a bounded but definite conclusion: the simplicity bias of the initialization function prior is a real and important zeroth-order term, but real optimization can transport a large amount of function mass that already satisfies the training set onto another complete function, and can make the outcome depend on training age and presentation history. Subsequent theory needs to keep both the static mass and the optimizer pushforward, and must not equate the two.

### 3.2 [E02] The Rule Bit Reveals the Separation Between Training Loss and Semantic Invariance

The first-stage experiments show that the optimizer actively reshapes the compatible-function distribution; immediately a boundary appears that is unfavorable to a strong version of "dynamic compression". The task input consists of one rule bit and 30 ECA state bits, and the output is one step of Rule30 or Rule110 evolution. When the rule bit is held fixed throughout training, external semantics says the output should not depend on it; but the training set only requires the fixed branch to be correct and gives no counterfactual-invariance constraint.

The experiment fixes a `31 -> 1024 x 3 -> 30` GELU+LayerNorm MLP, 2,048 training states, 65,536 probes, 8 seeds, no dropout and no weight decay.

Stage 1 trains only on the fixed rule bit for 8,000 steps; Stage 2 forks from that checkpoint, adds `k=0...512` nested counterfactual samples, and trains for another 5,000 steps. The cold control trains on the final data directly from the same initialization for 13,000 steps. The experiment compares both binary `0/1` and centered `-1/+1` encodings, and both directions in which the fixed value is 0 or 1.

Reproduction scripts and full description: [`E02`](experiments/experiment_02_rule_bit_invariance/README.md).

In Stage 1, all original branches reach exact accuracy 1, but behavior after flipping the control bit depends heavily on the encoding and the value. Under the binary encoding, after training with the bit fixed at 0 and flipping to 1, Rule30 and Rule110 both keep exact 1 on 8/8 seeds; after training with the bit fixed at 1 and flipping to 0, the median exact of Rule30 is about 0.9856, while Rule110 is only 0.00151.

The flip under the centered encoding is more drastic: Rule30 only about 0.011--0.015, and Rule110 is 0.

This asymmetry can be explained directly from the first layer:

$$
h=W_xx+w_zz+b.
$$

If `z=1` during training, the data can only constrain `w_z+b`: the constant feature and the bias obtain parallel gradient pathways, and the network can fully use `w_z` to carry the thresholds and gating needed to lower the training loss. At test time, `1 -> 0` amounts to suddenly deleting part of the learned bias. If `z=0`, that column receives no gradient in Stage 1; the centered flip then applies the original contribution with the sign reversed.

An "irrelevant variable" in human semantics therefore does not equal "will not participate in the computation" in the neural parameterization.

A small amount of counterfactual evidence can write the invariance into the loss. A warm model trained binary-fixed-0 needs only `k=1` (about 0.049% of the original training set) to make both rules strictly invariant on 8/8 seeds; but binary-fixed-1 succeeds strictly on only 7/8 even at `k=512`, and most centered conditions need tens to hundreds of samples.

Warm has no uniform advantage over cold either, so this experiment cannot be summarized as "an already-learned representation can always complete an invariance from a single example".

What this group of results refutes is the idealized strong proposition: optimization automatically recovers the purest program in human semantics and deletes all variables the true rule does not need. The more robust conclusion is: the network lowers the training loss in the language of the concrete architecture; counterfactual directions not identified by the data can become computational resources, retain priors, or exhibit path dependence. Only after counterfactual samples enter the training set does semantic invariance become a constraint that optimization must satisfy.

This boundary does not refute the later loss-resolved compression main axis; on the contrary, it clarifies its conditions: complexity and function mass must be defined relative to the training data and the network protocol, and one cannot use external generator semantics to designate a unique canonical implementation on the network's behalf.

### 3.3 [E03] Abandoning the Circular K Ruler and Measuring Function-Ensemble Dynamics Directly

E01 had already refuted the passive-screening version of the hard prior, but there was still no independent criterion for "whether the migration heads toward lower K complexity". The original plan tried to construct a hand-written DSL shortest length for Boolean functions, and was then actively withdrawn: a DSL only defines an artificial reference language; if one then defines neural complexity by the post-training function surprise and uses it to explain the same training distribution, the reasoning becomes circular.

E03 therefore deliberately does not measure K at all; it only saves complete function IDs and logits, first establishing the empirical facts that any complexity theory must explain.

E03a uses a `3 -> 1024 x 3 -> 1` GELU+LayerNorm network and Adam. The three training sets contain 1, 2, and 4 samples, leaving 128, 64, and 16 hard-compatible functions. Each group trains 128 ordinary initializations together with 512 prior-consistent models that already satisfy all labels at initialization, and tracks them to 5,000 steps post-fit.

Reproduction code and complete results: [`E03`](experiments/experiment_03_function_posterior_dynamics/README.md).

The prior-consistent models change their complete functions on a large scale after only one update: 72.7%, 99.0%, and 97.9% under the single-, two-, and four-sample conditions; the maximum function mass changes from 27.3% to 100%, from 7.2% to 77.3%, and from 17.4% to 39.3%.

The migration has clear structure: with two samples it concentrates mainly on one symmetric maximum-margin threshold function, and with four samples mainly on `x2 AND (not x1)` and `x2 OR (not x1)`; the one-variable solutions that abound in the initialization conditional prior are instead squeezed out. The selection criterion therefore equals neither the hard-prior proportions nor the surface length of a human formula.

After the ordinary models fit for the first time, continuing for 5,000 steps, 32.8%, 50.8%, and 47.7% of the seeds still change their complete function under the single-, two-, and four-sample conditions. The overall four-sample distribution changes by only 0.125 in TV, yet nearly half of the models have switched functions, showing that transport in opposite directions can cancel in the marginal histogram; "the overall distribution looks stable" cannot replace paired trajectories.

E03b then scans widths 16/32/64/128 of `3 -> width x 2 -> 1 tanh`. The main posterior differences between width 64 and 128 are within bootstrap noise, while width 16 retains the qualitative migration but changes the concentration speed and the ranking of near-tied functions.

The initialization-prior TV between the whole tanh family and the 1024x3 GELU+LayerNorm network is already about 0.29, and the specific attracting functions also differ systematically; yet all widths retain large-scale post-fit migration, and the 353-parameter width 16 is no exception. The network protocol determines both the prior and the reference language of the dynamics.

At this point the object of study was temporarily rewritten as the training-set-indexed family of function distributions

$$
\mu_{\Pi,D,t}(f)
=
P_{\theta_0,\xi}[h_{\theta_t}=f\mid D],
$$

where `xi` denotes the training randomness. The prior, the single-sample constant state, the underconstrained multimodal state, the narrow-rule state after grokking, and random-label function clouds can all be regarded as different members of this family. The term "training-induced function ensemble" is used here rather than Bayesian posterior, to avoid presupposing that it satisfies Bayesian conditioning. Whether a unique stable limit exists, and whether the path is eventually forgotten, still had to be decided by later experiments.

E03 also draws a boundary for agreement: low agreement only says that pointwise prediction is not concentrated; it cannot independently prove that the complexity distribution is broad, because two complementary functions each taking half the mass can produce the same marginal. Complete-function entropy, target-function mass, and paired migration must be measured separately.

### 3.4 [E04] The Low-Loss Limit Target and Post-Fit Function Drift

After E03, an intermediate picture closer to the final theory appeared. A fixed training set and network protocol do not simply specify a Bayesian hard posterior; together they define a family of parameter masses and function distributions that vary with the loss threshold. In the notation formed later, this can be tentatively written as

$$
Q_{\Pi,D,\epsilon}(f)
\propto
\mu_\Pi\{\theta:h_\theta=f,\ L_D(\theta)\leq\epsilon\}.
$$

At the time, the limit geometry pointed to by $\epsilon\to0$ was called the "low-loss limit target" jointly determined by the training set and the network. "Target" here does not mean a unique weight point, and does not even presuppose a unique hard function; it refers to the dominant function mass in the low-loss limit. SGD can keep approaching this target, yet may be limited by accessibility, finite time, and path bias, so it is neither guaranteed to arrive exactly nor guaranteed to sample unbiasedly.

This picture also corrected the idea that "as long as training is long enough, small data will eventually grok". If $D$ is insufficient to identify the generator-consistent extension, then even if the training loss is pressed very low, multiple candidate functions can still coexist; optimization time cannot replace missing data constraints. Conversely, the harder a rule is to describe and the more competing extensions it has, the more random samples are usually needed to identify it.

This corresponds naturally to the identification idea in MDL, and at the time it formed a prediction awaiting test: **the location of the grokking data transition can become an operational measurement of protocol-relative rule difficulty.** Later text will formalize it as the Neural K-profile and as `n50/n90` on random training sets, but at this experimental stage it was still only a forward-looking hypothesis derived from MDL.

To judge whether first interpolation has already arrived at this low-loss target, we conducted, on [the Boolean protocol of Mingard et al. (2025)](https://www.nature.com/articles/s41467-024-54813-x), the experiment [E04](experiments/experiment_04_mingard_postfit_drift/README.md).

The experiment uses the 128-state 7-bit complete truth table, a `7 -> 40 x 10 -> 1 tanh` network, 64 training samples, three target complexities, and `\sigma_w\in\{1,8\}`; parameters are updated by Adam, and sample selection uses the paper's advSGD hard-example sampling. Each condition trains 2,048 models, for 12,288 trajectories in total.

Unlike the original protocol, which stops at the first zero training classification error, E04 continues optimizing for 100 and 1,000 steps, and saves each trajectory's complete 128-bit function, logits, and training loss.

The preregistered competing predictions are direct: if the first-zero-error snapshot already adequately represents the static-conditioning endpoint, then, conditional on the training-set labels remaining all correct, the complete function should drift only by a sampling-level amount; if training is still transporting function mass along the continuous loss geometry, then the post-fit stage should show systematic changes on unseen inputs.

The results clearly support the latter case:

| Initialization scale | Function-change rate of the three targets at age 1,000 | Mean Hamming distance/128 | Mean per-input marginal drift |
|---|---:|---:|---:|
| `sigma_w=1` | 59.96%--97.66% | 1.16--4.03 | 0.00407--0.00529 |
| `sigma_w=8` | 99.66%--99.80% | 14.77--15.37 | 0.01001--0.01467 |

The permutation tests of per-input marginal drift in all six conditions give $p\approx0.00498$; keeping only the models that maintain zero training classification error at both endpoints, first fit and age 1,000, does not change the conclusion. The direction of migration, however, varies by protocol: for `sigma_w=8` the whole-space error of the three targets decreases on average by about 1.81, 1.53, and 0.80, while for `sigma_w=1` the changes are about +0.46, +0.11, and -0.03.

What E04 proves is therefore that the post-fit function distribution is still migrating systematically, not that "continued training monotonically improves along some artificial complexity proxy in every setting". The paper retains the LZ statistic for alignment with the original work, but its direction is not uniform, and this project does not treat LZ as the primary measurement of Kolmogorov complexity.

E04 thereby rules out the strong proposition that "first interpolation is the endpoint of function selection", and also concretizes the early limit-target picture: after the classification hard constraint is satisfied, the continuous training loss still provides a large amount of additional structure, and the function distribution keeps moving toward deeper low-loss regions. It does not yet prove, however, that the limit distribution is unique, that SGD eventually mixes sufficiently, or that extending training can recover the generator rule when the data are underconstrained; these boundaries were later addressed by static volume, optimizer-gap, and sample-identification experiments respectively.

### 3.5 [E05] A Sufficient Fixed NTK Still Does Not Prevent Feature Learning

Reading [Göring et al. (2025), *Feature Learning Is Decoupled from Generalization in High Capacity Neural Networks*](https://arxiv.org/abs/2507.19680) forced a necessary conceptual split: large changes in the kernel, the hidden representation, or the weights only prove that feature learning has occurred in the network; they do not prove that these changes necessarily improve generalization.

Random labels can also cause representation changes; conversely, a fixed kernel well suited to the task may also achieve generalization without changing features.

Our framework explains this separation naturally. Feature learning is a mechanism by which the optimizer transports parameters and representations along the training objective; generalization depends on whether the low-loss mass defined by the training set and the network protocol concentrates on a common extension on unseen inputs. A network facing underconstrained or random-label data can still reorganize its representation without having enough constraints to form a generator-consistent rule; a fixed kernel already aligned with the task can also generalize directly.

The two are therefore neither mutually sufficient nor mutually necessary.

This immediately raises a sharper experimental question: if [the fixed NTK of Jacot, Gabriel, and Hongler (2018)](https://arxiv.org/abs/1806.07572) is already sufficient to solve the task, will a finite-width network still actively change its features? [E05](experiments/experiment_05_ntk_sufficient_feature_learning/README.md) uses 30-bit Rule 110 single-step updates, 8,000 training states, and 20,000 unseen states.

The network is a bias-free `30 -> 1024 x 3 ReLU -> 30`; three seeds are each trained with BCE and MSE for 30,000 steps, with weight decay 0.

The experiment strictly distinguishes three objects: the analytic infinite-width NTK of the matched parameterization, the finite-width empirical NTK of the three concrete initializations, and the multi-output block NTK measured on 128 fixed probes during end-to-end training. The closed-form empirical NTK was first checked against the per-parameter autograd Jacobian Gram, with a maximum absolute error of `3.55e-15`.

The analytic NTK then achieves zero error on 20,000 unseen states, 600,000 output bits. The width-1,024 empirical NTK averages 99.8778% bit accuracy on three representative output bits, with three-bit joint exact accuracy of 99.575%--99.750%; it is very close, but it has not been passed off as the infinite-width zero-error result.

All six real network trajectories eventually reach zero error on both training and test complete 30-bit outputs. From initialization to 30,000 steps, the structural changes are as follows:

| loss | Hidden representation `1-CKA` | block NTK `1-CKA` | ReLU gating flips | Hidden-representation target alignment |
|---|---:|---:|---:|---:|
| BCE | 0.2400 | 0.01556 | 37.85% | 0.1305 -> 0.2472 |
| MSE | 0.0920 | 0.00728 | 25.76% | 0.1305 -> 0.1666 |

The MSE control still shows significant CKA change, about a quarter of gating flips, and an increase in task alignment, ruling out attributing the whole phenomenon to BCE continuing to inflate the logit scale after zero error. After the network is first simultaneously all-correct on the training and test sets, five trajectories cross the preregistered CKA and gating double thresholds; the sixth, `mse_seed2`, still has 1.145% gating change, but both normalized kernel changes are below the `0.001` threshold.

Because there is no snapshot between steps 3,000 and 10,000, the result is conservatively reported as "five pass, one does not".

E05 therefore supports a precise proposition: **the existence of a fixed-kernel solution does not force the end-to-end network to stay in the fixed-kernel mechanism; nor does feature learning occur only when the NTK cannot generalize.** It does not prove that feature learning is a necessary condition for generalization, and it does not equate CKA change directly with compression or a drop in program complexity. Its role is to formally split "whether the representation changes" and "whether the unseen function is identified by the data" into two layers.

### 3.6 [E06] Continuous Loss Becomes a Static Measurement Coordinate for the First Time

The discussion of agreement first clarified a fact that later became very important. The same partial training set is usually embedded in many different complete rules; continuing to add samples along the generator can make the function distribution converge toward the generator, while adding samples along another set of compatible labels may first make the distribution fork and then converge onto another readable rule. Agreement therefore only expresses the degree of function condensation under the current data and protocol; it does not grant truth privilege to some external generator.

This idea later developed directly into the anti-consensus completion and the "does high agreement correspond to a human-readable program" experiments; in this time period it was still only an explicit experimental-design prediction.

The key turn came from re-examining grokking. When training hard accuracy is already 100%, BCE is still nonzero, and the network keeps lowering the training loss. The initial conjecture was: lower loss pushes the network toward regions of lower program complexity, so that grokking becomes continued compression after the hard constraint is satisfied. An immediate objection is: for the same finite training set, both simple and complex functions can have arbitrarily low-loss realizations.

What really needs comparing is therefore not whether low-loss solutions exist, but how much parameter mass the low-loss realizations of different complete functions each occupy.

This led to the embryo of what later became the project's most central static object: as the training-loss threshold is tightened, does the relative mass of complex functions disappear faster than that of simple functions? At the time it was also conjectured that complex regions might escape more easily, but the latter belongs to dynamics and cannot be verified by static sampling. [E06](experiments/experiment_06_loss_conditioned_prior_annealing/README.md) tests only the former part.

The experiment fixes a `3 -> 1024 x 3 -> 1` GELU+LayerNorm network, samples 4,194,304 untrained networks, and completely identifies the 256 hard functions on the 8 inputs.

For the single-, two-, and four-sample training sets, the parameters whose hard predictions are all correct are kept first, and the complete function distribution is then reconstructed from nested quantile sublevel sets of training-set raw BCE, together with `exp(-beta L_D)` annealing weights. RMS-normalized loss and fixed logit-scale subgroups are used to separate pure scale effects.

The four-sample hard-exact cohort still has 365,248 networks, and the lowest 0.1% tail has 366, enough for a tail verdict.

The result is not a simple monotone law. In the single-sample raw lowest 0.1%, constant 0 rises from 28.1% to 86.4%, and the function entropy falls from 5.021 to 1.098 bit; but the effect disappears or reverses after scale control, showing that the overall logit scale contributes a great deal.

The two-sample case gives a genuinely robust result: the symmetric threshold function ID 113 rises from 1.57% at hard-exact to 23.0% at the raw lowest 0.1% and 50.7% at the normalized lowest 0.1%, and the fixed-scale lowest 0.5% also has 44.1%. This same function is precisely the main attractor chosen by real SGD in E03.

The four-sample case then becomes a necessary counterexample. The static total mass of the SGD attractors ID 48/243 rises from 21.4% to about 29.0% in the intermediate region, then falls back to 23.0% at the raw lowest 0.1%; the normalized deep tail even turns toward other functions, and the function entropy does not decrease. Real SGD, by contrast, pushes the two to about 77.2% after one step.

This rules out two strong propositions at once: that all compatible functions contract proportionally by a common loss factor, and that real SGD equals the static annealed distribution at the same loss.

The lasting conclusion left by E06 is not "the lower the loss, the more every predetermined simple function monotonically increases", but rather: **the continuous training loss rearranges static parameter mass in a function-specific, architecture-dependent way; this geometry can strongly influence the multi-seed training distribution, yet it is not the optimizer distribution itself.** Different seeds are samples of the optimizer-induced pushforward, not samples of a Bayesian posterior; the two may be close because they share the same low-loss geometry, and may also deviate systematically because of entry, inflow rates, escape rates, and path history.

At this stage there was not yet an independent definition of architecture-relative complexity, so "lower loss favors simpler functions" could only be retained as a partially supported working hypothesis. The later full-target volume, measure closure, and Neural K-profile rewrote it into a measurable formal proposition that permits scale crossings.

### 3.7 Function Wandering, Matched Loss, and Shared Computation

#### 3.7.1 [E07] The Same Loss Does Not Determine the Same Function Distribution

E06 still only observed static parameter samples. To directly see how the optimizer transports function mass, [E07](experiments/experiment_07_function_wandering_matched_loss/README.md) switches to a fully trackable `4-bit -> 1-bit` system: 6 balanced training points are fixed, leaving 10 unseen points and 1,024 compatible hard functions;

1,024 `4 -> 16 x 3 -> 1 tanh` networks are trained with full-batch Adam, aligned at the moment each first fits the training set completely, and recorded for another 20,000 steps.

1,019/1,024 trajectories change their complete function at least once after the first fit, with 4,379 transitions in total, 90.9% of which flip only one unseen-input bit. The function distribution contracts from entropy 5.217 bit, top mass 12.1%, and agreement 0.683 at first fit to entropy 1.291 bit, top mass 79.8%, and agreement 0.945 at age 10,000.

The migration occurs mainly along adjacent function cells, not as an unstructured random walk over the compatible functions.

The training labels turned out afterwards to satisfy exactly `y=x1`, whose ID 666 is the most common at first fit; the long-run dominant function, however, becomes another linear threshold, ID 722. Only 81 of all compatible functions are linearly separable, yet after age 10, 100% of the models lie in that family. This shows that the network first strongly selects a narrow architecture-preferred family, and then redistributes mass within the family by margin and dynamics; `x1`, the human-shortest formula, does not automatically become the optimizer's endpoint.

4,194,304 untrained networks were then sampled on the same network and training set. raw BCE quickly becomes unmatchable: the lowest loss of static hard-exact samples is 0.6061, while SGD is already down to 0.6006 at age 2.

After independently recomputing the normalized BCE from the saved logits, the mean losses of static `beta=128` and SGD age 20,000 are 0.2885 and 0.2903, almost identical; yet the function distributions are:

| Metric | Static same-loss annealing | SGD age 20,000 |
|---|---:|---:|
| Function entropy | 3.139 bit | 1.450 bit |
| ID 666 mass | 30.6% | 2.9% |
| ID 722 mass | 23.9% | 77.1% |
| agreement | 0.853 | 0.936 |

The static geometry had already amplified ID 722 from 0.94% to 23.9%, so it did identify the candidate direction that real training would exploit; the JS divergence between the two function distributions is still 0.378 bit, and SGD further reverses the ID 666/722 odds and forms history dependence.

E07 thereby gives a decomposition retained all the way to the final theory: **static loss mass provides the candidate structure, while the optimizer determines the actual inflow, residence, and path transport; the same scalar loss is insufficient to determine the function distribution.**

The fine dynamics of this experiment come only from width 16. The entropy re-expansion appearing after age 17,000 occurs at loss about `1e-8`, close to the Adam epsilon and the float32-sensitive region, and is not used as evidence of a universal phase transition; the automatic normalized-match column in the result bundle also used a wrong matching target, and the numbers in the main text come from the independent recomputation on the saved logits.

#### 3.7.2 MDL Is Not a Second Objective in the Training Code

This stage also corrected the statement that "training minimizes loss and program complexity at the same time". Standard MDL is written as

$$
L(H)+L(D\mid H),
$$

but ordinary neural-network training code usually only explicitly minimizes the empirical loss plus explicit regularization. The program does not contain a separately written machine-independent `K(H)`; continuing to lower BCE after the hard fit is still just continuing to optimize the training NLL. The compression, reuse, and enrichment of simple functions observed in this document should be interpreted as **results** of the architecture, the parameter measure, and the loss geometry, not as secretly giving SGD a second purpose.

MDL still provides an important upper-level explanation. If, in a family of candidate programs of length about `K` bit, every wrong candidate has error rate at least `eta` on the data distribution, then the probability that one wrong candidate passes `n` independent samples is about `(1-eta)^n`. Taking a union bound over about `2^K` candidates, the sample size required for identification satisfies the rough relation

$$
n
\gtrsim
\frac{K+\log_2(1/\delta)}{-\log_2(1-\eta)}.
$$

The longer the description, the more competing programs, or the harder they are to tell apart, the more data random samples usually need to identify it. At this stage this was still the MDL intuition for interpreting the grokking transition point; only later did the Neural K-profile's prospective prediction of `n50/n90` turn it into direct evidence within the project.

#### 3.7.3 [E08] Why Shared Intermediate Computation Helps Reach Deeper Loss

The remaining gap is: why does a more economical implementation gain an advantage in the low-loss region? The corresponding phenomenological mechanism is computation reuse under finite capacity. If two outputs both need an expensive intermediate state, sharing the computation only needs to form that state once; without reuse, two sets of capacity must be occupied. The sole explicit objective of lowering the loss therefore puts pressure on implementations that can share features, computation, and intermediate representations.

[E08](experiments/experiment_08_shared_intermediate_pressure/README.md) uses CA to construct a four-cell causal control. Let `F_k=Rule30^k` and `G_k=Rule110^k`, and let `A` and `B` apply AND/OR to the two halves of the 30-bit state, respectively.

The Shared condition is `[A(F),B(F)]` or `[A(G),B(G)]`, and the Separate condition crosses them into `[A(F),B(G)]` and `[A(G),B(F)]`; the four atomic tasks thus each appear once in both groups. The Joint network allows the two outputs to share three hidden layers, while the Split network uses two parameter-matched, fully isolated branches.

If Shared were merely easier in label statistics, it should show the same advantage in Split.

The experiment scans prefix depths `k=0..3`, Joint widths 256/512/1024, and 3 seeds, for 288 trajectories in total. The Shared/Separate loss gap in Split is about `1e-8` in every cell, proving that the atomic difficulties are balanced.

The fixed learning rate caused 25 trajectories to collapse late after reaching extremely low loss, so the preregistered conclusions use the best loss within budget and the first time a loss threshold is reached, not the mean of the final checkpoints.

The core interaction shows a clear dose relationship: `k=0` has no sharing benefit, `k=1` lies in the crossover zone; at `k=2/3`, Shared best loss is lower than Separate in all 18 seed pairs. At `k=2` the Separate/Shared best-loss ratio is 8.06, 4.59, and 3.58 across widths; at `k=3` it is 399.4, 74.8, and 6.07.

The wider the network, the easier it is to hold two sets of computation at once, and the weaker the sharing advantage. The online sample size needed to reach a fixed low loss gives results in the same direction: for example, at `k=2,W=256`, reaching BCE `1e-3` requires 768,000 versus 1,984,000 samples.

E08 therefore provides a bounded but important causal bridge: when the common intermediate computation is expensive enough, allowing reuse reaches the same deep loss with smaller capacity and fewer samples. This supports the phenomenological explanation that "lower loss rewards reusable representations", and it does not depend on inferring the internal implementation from the hard function ID. It does not yet prove through intermediate-layer probes that the network explicitly forms the researcher-named `F_k/G_k` variables, nor that all layers or neurons share the computational pressure uniformly.

#### 3.7.4 An Interim Account of OOD Semantics and Grokking

Shared computation also provides a candidate macro-level explanation for the earlier OOD rule-bit generalization: when many tasks can be reused by the same interpreter and symbol bindings, forming stable semantics may be more economical than keeping multiple sets of hard-wired circuits. This is a theoretical extrapolation connecting the existing OOD behavior with E08, not E08 alone proving human concept formation; the human-cognition analogy is kept only in the long-paper discussion.

The same picture also makes grokking smoother. When data are insufficient, multiple shortcuts, hybrid extensions, and local reuse schemes can all satisfy the training constraints, and continued training does not guarantee that the generator function eventually dominates. Near the transition, the data already permit rule identification, but the optimizer may first enter other fitting regions and only later, at deeper loss, turn toward the shared rule.

When data are sufficient, newly added consistent samples keep eliminating competing extensions, so the rule direction dominates already at relatively high loss; train/validation then decrease in sync, and no macroscopic memorization plateau is seen.

What can be strictly guaranteed here is hard conditioning: adding a new sample consistent with the target function keeps the target cell and deletes incompatible cells, so its normalized hard mass does not decrease. For the continuous volume at a fixed raw-BCE threshold, a new sample also deletes part of the target weight by margin, so one cannot claim without additional conditions that every loss level strictly increases. The later loss-stage reordering experiment is exactly a further stress test of this strong monotonicity claim.

### 3.8 Noise, Training-Set Semantics, and the Raw-BCE Proposition

#### 3.8.1 [E09] 80% Hidden Label Noise Films Overfitting in Slow Motion

E09 first adjudicates a widely circulated claim: even when 80% of the MNIST training labels are wrong, the student network can still reach about 91% test accuracy. The real ambiguity is whether that number comes from the interpolation end state in which the network has already memorized the noise, or from an early window in which the shared digit regularities have been learned first and the noise has not yet been memorized.

[E09](experiments/experiment_09_mnist_hidden_noise_long_training/README.md) fixes the corruption of 40,000 labels among 50,000 training samples, without giving the input any noise identity;

an overparameterized CNN and a small CNN close to the original setting each use 3 seeds, the noise80 and clean controls share elementwise-identical initialization and data order, and training continues for 300 epochs. The experiment strictly separates the clean-validation best point, the first interpolation point of noisy train 99.99%, and the final long-training point.

The early results reproduce the about-91% phenomenon almost exactly: the overparameterized CNN reaches 91.89% test at epoch 4, and the small CNN reaches 93.89% at epoch 5; at that time noisy train accuracy is only about 19.5%, the corrupted subset has only about 1% accuracy on the wrong labels, and still about 91%--93% on the hidden truth.

The model has already found the shared digit structure, yet has barely begun to memorize the fixed wrong labels.

After the noisy training loss is further lowered, the function evolves in the opposite direction:

| Architecture | Final noisy train | Final clean test | Clean control final test |
|---|---:|---:|---:|
| overparam CNN | 62.51% | 22.79% | about 99.29% |
| small CNN | 33.46% | 32.42% | about 99.25% |

By the endpoint, 59.42% of the overparameterized model's corrupted subset agrees with the wrong labels and only 11.47% with the truth; from epoch 5 onward, the correlation between noisy train accuracy and clean test accuracy is `-0.941`, and `-0.980` for the small network. The clean control does not collapse under long training, showing that the destruction comes from satisfying the wrong training labels, not from time itself.

Neither noise model reaches 99.99% interpolation within 300 epochs, so E09 cannot claim to have measured "the final performance after completely memorizing 80% noise". What it can robustly prove is the more direct fact: **long before noise memorization is complete, the external rule has already collapsed from about 92%--94% to 23%--32%, with no late recovery.** This magnifies overfitting into a two-stage process: "first exploit shared structure to lower the loss, then keep lowering the loss with sample-level residuals and rewrite the overall function".

#### 3.8.2 The "A is A" Clarification of the Training Objective

E09 forces the theory to withdraw any globally monotone

$$
L_D\downarrow\Rightarrow K(f)\downarrow
$$

claim. When the network starts from a random initialization and forms task-relevant structure, the task program and `I(theta;D)` rise in the first place; after the structured signal is exhausted, continuing to memorize non-reusable residuals further increases sample-level complexity. The only thing written into the optimization program is the training objective; compression, feature reuse, rule learning, and noise memorization are just different implementation paths for lowering that objective in different data regions:

$$
\boxed{\text{loss is the objective; compression is a conditional path; generalization is an external result.}}
$$

This looks like a tautology, yet it rules out a frequently committed conceptual smuggling. The data-generation script is not handed to the network as the "correct rule"; wrong labels are not marked as noise to be ignored; test accuracy and human semantics are not in the training objective. That clean validation can select an early model is precisely the researcher re-injecting the semantics they care about from outside training.

#### 3.8.3 [E10] From a Failed Broad Scan to Strict Function Pairs

The stress test of the working hypothesis left by E06 then began. The first version of [E10](experiments/experiment_10_raw_bce_rule_pair_competition/README.md) scanned projections, majorities, parities, and random balanced constraints in the complete 4-bit space, using 67,108,864 `tanh16x2` and 100,663,296 `GELU-LN 1024x3` prior networks.

Judged by external complexity panels such as ANF, DNF, and decision trees, only 56.8% of the 44 valid conditions become simpler overall under raw BCE, and only 29.5% and 33.3% under normalized and fixed-scale. The strong proposition "under any finite constraint, lower loss monotonically simplifies in an external Boolean language" is refuted.

The failed conditions then exposed two confounders. First, severely underconstrained data do not tell the network the researcher's presumed generating rule; two diagonal samples do not uniquely imply majority even to a human. Second, many functions the panel calls complex are actually linear thresholds that an MLP expresses naturally.

The criterion was therefore changed to directly competing function pairs: simple is a projection, AND, or OR, and complex adds two exceptions at unseen points; the two have the same training outputs and the same complete Hamming weight, but simple can be expressed by a single linear-threshold unit, while complex is proven not linearly separable by linear programming and is also strictly more complex under other logical proxies.

Training samples are taken only from positions where the two agree, with `k=10,12,14`.

The main criterion is fixed at the raw BCE the network actually corresponds to:

$$
R(q)=
\frac{P_q(f_s)/P_q(f_c)}{P_{hard}(f_s)/P_{hard}(f_c)}.
$$

normalized and fixed-scale change the loss or the conditional sample group, and serve only as mechanism diagnostics; they cannot participate in judging the truth of the original proposition. The old automatic summary also treated 0/1 hits in the complex tail as reliable results; the final re-audit requires that, under the odds-unchanged null hypothesis, the expected tail counts of both simple and complex be at least 20, and then selects the deepest quantile and performs the Fisher test.

The final raw result is 12/12 odds increases, all 12/12 significantly favoring simple. The nine GELU groups average about 10.6x odds enrichment, and the three tanh groups about 5.1x. The strongest projection `x0` results by data size are:

| Sample size | simple odds enrichment | Fisher p |
|---:|---:|---:|
| 10 | 13.59x | `5.88e-8` |
| 12 | 26.71x | `2.00e-16` |
| 14 | 81.99x | `1.96e-18` |

GELU AND enriches 9.96x at `k=12`, OR enriches 8.49x at `k=10`, and tanh AND enriches 6.00x at `k=12`. This result directly separates "complex functions have equally low-loss solutions" from "complex functions have equally low-loss mass": on these independent ordered function pairs, lower raw BCE significantly increases the relative mass of the simple compatible function.

E10 is still not a globally monotone law. It only compares the one deep slice of each group that meets the final reliability standard; the failure of the broad scan remains valid, and static odds do not equal the optimizer distribution. The later AND shortcut and deep-tail crossing will further show that the dominant function and the volume ranking can reorder with the loss scale.

#### 3.8.4 Erratum to "With Sufficient Data, Indefinitely Long Training Is Always Healthy"

At the time, E10 was used to further infer: for clean synthetic rules, as long as the samples exceed the grokking transition point, continuing to lower the loss will no longer overfit. Under the later loss-resolved picture this statement is still too strong. Any finite training set constrains unseen extensions only within a finite resolution; exceeding the transition point means the generator-aligned function already dominates at some operational threshold, not that it dominates permanently for arbitrary $\epsilon\to0$.

As training enters deeper loss, the margin configuration among the finite samples and the still-uneliminated competing extensions may re-emerge, and train and validation can eventually still separate. Adding random samples pushes this separation point toward deeper loss; adding the samples with currently lowest agreement or most uncertain prediction is usually more efficient. Only a complete truth table, or a training set that already suffices to uniquely identify the target under explicit assumptions, can promote "infinitely long training is safe" to a stronger conclusion.

Early stopping is therefore meaningful not only in the presence of label noise. The more accurate statement is: the more sufficient the data and the cleaner the structure, the deeper the training loss to which the rule can be supported; the training budget and validation still correspond to the external accuracy level required by the researcher, not to metaphysical insurance.

#### 3.8.5 [E11] Data Size Determines How Deep the Rule-Aligned Descent Channel Extends

At 17:35 a more local measurable question was raised: when data are sufficient, why do train and validation loss decrease almost in sync? This does not require the high-loss region to already be filled with the correct hard function. For one small update along the negative raw training gradient,

$$
\Delta L_{val}
\approx
-\eta\nabla L_{val}\cdot\nabla L_{train},
$$

the angle between the training and validation gradients gives a first-order prediction of validation-loss change along that direction, not a direct measurement of an actual Adam update. [E11](experiments/experiment_11_rule30_gradient_alignment/README.md) fixes one step of 30-bit Rule30, a `1024x3` GELU+LayerNorm network, and 5 paired seeds, using strictly nested `n=256...4096` training sets and 4,096 independent validation states.

The experiment compares not only equal step counts, but also selects the nearest checkpoint by each trajectory's `log(raw BCE)` distance, retaining only points with `max(actual_loss / target_loss, target_loss / actual_loss) <= 1.35`. The frozen implementation does not interpolate: these are diagnostic points near target losses, not exact common sections.

The archived results show that at target BCE 0.2, 0.1, 0.05, 0.02, 0.01, and 0.003, each of the five seeds individually shows train/validation gradient cosine strictly increasing with sample size under nearest-checkpoint matching. At target BCE 0.3, only 4/5 seeds preserve that complete strict ordering; it must not be reported as 5/5 throughout the range.

Under nearest-checkpoint matching at target BCE 0.1, the mean cosine over five seeds rises in sequence from -0.167 at `n=256` to 0.927 at `n=4096`, and the mean validation-train gap falls from 0.482 to 0.019; at target BCE 0.01 the mean endpoint cosines are -0.237 and 0.950, with gaps of about 0.693 and 0.00137. Actual BCE values lie within the tolerance above and do not all equal the target; a negative group mean also does not imply negativity for every seed.

The behavioral endpoints form a continuous spectrum: 256/512 samples fit the training set quickly but end with only 74.84%/92.96% validation bit accuracy; 768 samples fit first, then generalize smoothly with delay to 99.9958%; with 1024 samples all five seeds end all-correct; with 4096 samples both training and validation exact accuracy reach 100% at about step 84.

What the data size determines is therefore not whether there is any rule direction at high loss, but how deep that direction can extend: with low data, cheap shared structure is exploited first, and the training gradient then rotates into sample-specific directions; with high data, overall rule coherence is still preserved at approximately matched deep loss. This provides trajectory-level local gradient-geometry evidence for the erratum in §3.8.4, and places ordinary overfitting, delayed generalization, and synchronous descent in one continuous picture; it does not measure the actual Adam update direction or validation-loss increment.

Attempts were then made to find, in Rule30, the original grokking shape of "validation loss first falls, then rises, then falls again", but the script hit float32 gradient extinction at about `2.5e-8` BCE and formed no valid verdict; Mod97 only reproduced a known curve, and the Mod3 "small data memorizes first, then groks" conjecture failed. All three remain in the ledger and do not establish short-paper experiment numbers.

### 3.9 Discrimination Cycles, the Hard-ID Boundary, and AND Stage Reordering

The norm and width/depth capacity scans, by the final verdict, do not enter the short paper. They refute several one-dimensional proxies without providing a stable definition for Neural K. The turn that really enters the main line begins with an epistemological problem: if whatever function the network selects can afterwards be called "simpler under the current architecture", then the simplicity hypothesis cannot be falsified.

#### 3.9.1 Two Discrimination Strategies That Constrain "Simple" in Advance

To avoid circular explanation, the experiments froze two strategies:

1. choose simple functions such as copy, NOT, AND, and OR that are nearly free of architecture controversy, and give them relatively sufficient training patterns;
2. for the same training set, directly compare simple with its complex variant after adding explicit exceptions, requiring the same training outputs and the same complete Hamming weight, while complex is strictly harder on linear separability and multiple logical proxies.

The multi-architecture 4-bit pilot verified that architecture differences do exist, but formed no independent claim because the inputs are too small and structures such as CNNs are too artificial. Three subsequent rounds of 30-bit ECA prior sampling then found that directly random weights can barely reach any meaningful low-loss region, with the lowest tail still close to random BCE. These method calibrations establish no experiment numbers; they pushed the research back to the enumerable 4-bit space, to solve the problem with falsifiable function pairs and data interventions.

The same stage also identified the granularity limit of the hard function ID. An ID only fixes the hard outputs on the 16 symbolic inputs; inside it there can still be different logits, margins, hidden representations, and continuous-input behavior, so "which hard function dominates" cannot exhaust program complexity. Furthermore, the training loss is the average of per-sample losses:

$$
L_D(\theta)=\frac1n\sum_i\ell_i(\theta).
$$

The section fixing `L_D=epsilon` actually integrates all sample-level margin configurations satisfying `sum_i ell_i=n epsilon`. A newly added sample changes this joint geometry, rather than simply superimposing a static label onto the old terrain. This observation later developed into the per-sample free-energy decomposition.

#### 3.9.2 [E12] AND Gives a Clear Counterexample to Any Globally Fixed Ranking

[E12](experiments/experiment_12_and_shortcut_static_geometry/README.md) first scans copy, NOT, AND, OR, and XOR on 33,554,432 `4 -> 64 x 3 -> 1` GELU+LayerNorm prior networks, with training sample sizes 10, 12, and 14.

copy/NOT/AND/OR give 12 adjudicable conditions in total; from the complete prior to the lowest 0.01% raw-loss tail, the target-function mass increases strictly monotonically in all of them, with enrichment reaching tens to hundreds of times; XOR is hit only 17 times and does not participate in the verdict.

But after fixing the hard-exact cohort, AND at n=10 becomes a highly significant counterexample. AND falls from 8.75% to 6.81% at the lowest 10%, while the structurally more complex F550 rises from 6.02% to 10.17%; the two odds-cross at about the lowest 50%. F500 also keeps growing in deeper regions.

The original training set omits several off-diagonal negative examples, so a shortcut relying on `x4` remains consistent with all 10 points and can form larger margins on the seen points: the mean signed margins of AND/F500/F550 are 0.378/0.404/0.444.

Targeted hole-filling established the causal chain. Closing only one branch eliminates the old shortcut but may expose a new `copy x2`; after covering the four key negative examples at the same time, AND rises from 40.2% at hard-exact to 54.9% at the lowest 1%. Of six random negative-example subsets, five enrich and one decreases. These controls preserve sample count and the positive-to-negative label ratio of 4:6, supporting an effect of concrete coverage on static selection that cannot be explained by sample count or label ratio alone.

A separate minimal balancing intervention replaces the positive example `1101 -> 1` with the negative example `1000 -> 0`, changing coverage of the four `(x1,x2)` patterns from `3/2/1/4` to `3/2/2/3` while also changing the positive-to-negative label ratio from 4:6 to 3:7. It does not change only negative-example positions and cannot isolate the coverage effect. Static AND rises from 16.65% to 28.10% at the lowest 0.2%, the mean number of exceptions falls from 1.733 to 1.242, and the entropy falls from 4.331 to 3.303 bit; but the AND/F050 odds first decrease and then overtake in the deep loss.

The static section itself already shows the stage reordering of "intermediate loss strengthens the shortcut, deeper loss strengthens AND".

#### 3.9.3 [E13] SGD Reproduces the Reordering, and a Single Sample Rewrites the Long-Run Distribution

[E13](experiments/experiment_13_and_shortcut_sgd_intervention/README.md) trains directly on the same network with 2,048 paired seeds, full-batch AdamW, and zero weight decay.

On the original gap, everything is already hard-exact at step 20, when AND is only 0.83% and F510/F550 are 26.71%/19.43%; the exceptions then gradually decrease, and at step 20,000 AND rises to 21.19%, but F100 still leads with 26.12%.

After the original condition is extended to 127,458 steps, it stabilizes on an underdetermined plateau: AND 22.51%, F100 25.29%, mean exceptions 1.771, unseen accuracy about 70.5%. From 100k to the endpoint, 98.3% of the hard functions no longer change, and the training loss still falls to `2.06e-9`, yet the unique AND is not automatically recovered. Optimization time cannot replace missing constraints.

The balanced condition, which replaces only one sample, produces a completely different trajectory; it uses the condition above that changes both coverage and the positive-to-negative label ratio, not a single-factor coverage intervention. At step 20, F050 is 42.58% and AND is 21.58%; at step 50,288, F050 falls to 5.62% and AND rises to 64.94%, with mean exceptions down to 0.450 and unseen accuracy 92.5%. Of the final AND models, 72.1% are formed by post-interpolation migration from the shortcut.

Compared with the original condition at the same seeds, 69.97% of the models reduce their exceptions, by 1.337 wrong points on average; the initialization seed did not fix the endpoint in advance.

The static geometry correctly foretold the stage directions, but did not reproduce the SGD strength: in the balanced static lowest 0.2%, AND is 28.10%, while real SGD ends at 64.94%. This once again confirms that static function mass and optimizer transport are related but not equal.

#### 3.9.4 Interim Theoretical Correction

E12--E13 clearly refute the stronger conjecture: there is no hard function fixed in advance that necessarily gains weight monotonically as the loss decreases in every loss interval. The dominant function can first fork, reorder, and then re-converge; when the data are underdetermined, it can also stop at a stable mixed plateau. Defining every actual winner afterwards as "simpler" would only re-create the circular argument.

The safe version is therefore changed to: the static loss terrain pre-exists, real training is strongly influenced by it but is not unbiased sampling; the concrete dominant function must be actually measured relative to the data, the architecture, and the loss scale. The winner in the low-loss limit can serve as an operational quantity, but one cannot prove an independent complexity ordering merely from "it won".

Only later, by adopting the complete Neural K-profile, using the pre-training volume ranking to predict independent `n50/n90`, and then externally validating with high-agreement symbol audits, was this circle truly bypassed.

### 3.10 Loss Stages, Accessibility, and the Static/Dynamic Layering

This stage first confirmed that naive prior sampling cannot enter the truly deep loss tail: the samples needed to crash from BCE 0.3 to 0.01 can grow astronomically, so one must switch to rare-event SMC and reference-measure-preserving MCMC. The difference from the existing volume literature also gradually became clear: existing work mostly starts from external generalization or a presupposed target function; this document takes the fixed training set, the continuous loss, and all compatible mappings as the primitive objects, without granting the generator function a special status in advance.

#### 3.10.1 [E14] A Single Hard Function Can Also Rise First and Fall Later

To directly break the globally monotone proposition, [E14](experiments/experiment_14_weighted_rulebit_function_switch/README.md) constructs a rule-bit-controlled Rule110/Rule30 task, with training weights set from 1:1 to 10000:1.

Each ratio uses 512 paired seeds, plain full-batch SGD, zero momentum/weight decay, and comparison at the same weighted raw BCE first crossing.

The fixed function `Rule110-both=0x6E6E` clearly first rises and then falls with loss: the peaks at 100:1, 1000:1, and 10000:1 are about 52.93%, 54.69%, and 51.76%, after which the same seeds gradually migrate to the complete composite function. Most branches fix Rule110 first; a minority of branches then correct bit by bit along `0x6E->{0x4E,0x2E}->0x0E->0x1E` according to the three conflicting neighborhoods.

After dividing the loss by the minority-branch weight, the curves of different extreme ratios approximately collapse, and the final takeover occurs near the theoretical loss floor of Rule110-both.

The 10000:1 condition even actively pushes the Rule30-branch BCE from 0.728 up to 1.297 while the total loss is falling rapidly, and only returns to the complete mapping at extremely deep loss. Training has no human-specified "composite true rule" objective; it only faithfully optimizes the weighted data distribution. E14 therefore constitutes an explicit counterexample to the monotonicity of a fixed hard function's probability.

#### 3.10.2 [E15] A Broad Complete-Function Distribution Does Not Imply a Lack of Marginal Condensation

[E15](experiments/experiment_15_mod97_function_distribution/README.md) uses 60%, 70%, 80%, and 90% nested data in the Mod97 division task, 32 seeds per level, tracking the complete 9,312-point function.

At 60% and 70%, not a single seed ends with the complete target; at 80%, only 1/32; at 90%, they start appearing from step 12,500, and eventually 18/32 become the complete target. In the 90% condition, when the matched train CE falls from `3e-7` to `3e-11`, the target mass rises from 3.125% to 56.25%.

Yet the coordinate-wise modal function recovers the target far earlier than any single seed: at 80% it has already become the complete target at step 18,000, while the first target seed appears only at 78,000; the corresponding steps at 90% are 2,000 and 12,500. At 70%, there is no target seed in the end, yet the modal accuracy still reaches 95.41%.

This shows that before grokking, not all seeds share the same complete shortcut; rather, they share a target-related skeleton while each keeps different off-table residuals.

Complete-function mass, hash entropy, agreement, and the coordinate-wise modal function must therefore be reported separately. Low agreement or zero exact-target mass is insufficient to conclude that the pointwise predictions are not condensing around the rule.

#### 3.10.3 [E16] Parity Separates Good Endpoints from Global Entry

Parity initially looked like an extreme counterexample of "leave out one point and it can never be completed", but [E16](experiments/experiment_16_parity_reachability/README.md) found that the leave-one-out protocol itself has strong position and symmetry effects.

At 4-bit it stably chooses the single-point flip, at 6-bit it is mixed, at 8--12-bit the complete parity has already become the dominant completion with about 92.7% on internal holdout; at 14--16-bit, before the completion preference can be reliably measured, the accessibility from random initialization collapses first.

At 14-bit, random, label-balanced half-spaces let all 64/64 models escape the `ln2` plateau and fit the 8,192 training points, with test mean 99.935%, but 0/64 become the complete parity. Precisely revealing the mean 6.22 wrong points of each parent model lets 23/64 recover the complete function; the same number of random correct points gives 0/64. Overly strong replay, although it clears the old errors, creates almost the same number of new errors at other unconstrained positions, demonstrating that constraint information and training weight must not be conflated.

The final 16-bit width-256 verdict is even more direct: random initialization endpoint-only is 0/8; after prefix scaffold, 5/8 exact, and after withdrawing all auxiliary supervision and continuing pure endpoint training, 7/8 exact.

When a relative L2 perturbation of 0.1 or 0.2 is applied to an exact anchor, all 28/28 hard functions are destroyed; after recovering with endpoint BCE alone for 5,000 steps, 23/28 and 21/28 are exact again, respectively.

Under the current protocol, the main obstacle for parity is therefore global entry, not insufficient capacity, not the exact endpoint being unsupported by the original objective, and not the absence of a local basin. A low program description and a favorable low-loss endpoint do not guarantee that ordinary gradients can see it; the main theory must treat optimizer accessibility as an independent boundary, rather than inferring from end-to-end failure that the target function is complex.

#### 3.10.4 [E17] Static Volume Is Strong, but the Optimizer Can Dominate

[E17](experiments/experiment_17_static_vs_optimizer/README.md) returns to the balanced AND n=10, `4 -> 16 x 2 -> 1 tanh` reference machine.

33,554,432 brute-force priors together with AdamW, SGD, and momentum crossings first show a strong first-order relation: at BCE 0.60, the static top-100 functions cover 98.7%--100% of the three optimizers' mass, with the same dominant functions; but JSD/TV all exceed sampling noise, so the optimizer is not unbiased sampling.

Constrained SMC has full 65,536-dimensional function-distribution JSD of only 0.00677/0.00297/0.00144 against the brute-force prior at 0.68/0.65/0.60, and then advances reliably into the deep tail.

The static dominant state undergoes multiple reorderings with loss — constant zero, 0x5000 shortcut, D000/D040, F040/D440, and so on; after all particles are already hard-exact, the soft BCE continues to strongly rearrange the unseen extensions. By 0.07, D440 is about 50.99% and AND is 0%.

A 32,768-seed AdamW run with the same network, same data, and same initialization measure, however, obtains AND 76.52% and F040 23.48% at step 11,444. The same-loss distribution at about BCE 0.10 already has JSD 0.301 against SMC; the final comparison has JSD 0.760 and TV 0.765. The optimizer's reweighting is by no means always a gentle second-order term.

Taking 4,096 real weight points from the SMC deep tail D440 and continuing AdamW, the loss decreases by about four million times, yet the endpoints are still D040/F040 with AND 0%; the paired group projected back to the initialization box keeps D440 at 100%. In the random-initialization path, F040 flows to AND, but the same F040 hard ID in the SMC path flows to D040.

The continuous state inside a hard ID and the entry history determine the future; parameter points inside one cell cannot be regarded as exchangeable samples.

Finally, the 8-bit experiment refutes the single-trajectory ergodic interpretation. The 64 late checkpoints of a single trajectory usually cover only 5--14 neighboring functions, while an IID-64 seed sample covers 28--64; surface-level agreement proximity cannot recover the margins and covariance either. Multiple seeds are an optimizer-induced ensemble; single-trajectory time points are not independent Bayesian posterior samples.

#### 3.10.5 The Layered Conclusion After 21 Days

At one point the research tried to split the volume hypothesis into two layers: first prove the static simplicity enrichment of lower loss, then attribute real generalization mainly to that static layer and demote SGD to a second-order term. E17 preserves only the strong first-order influence of the first half, and directly refutes the universal version of the second half. The more accurate empirical bookkeeping is

$$
Q_O(f\mid D,L)
\propto
V_\mu(f;D,L)R_O(f;D,L),
$$

where `V_mu` is the static function mass and `R_O` summarizes entry, flow tubes, speed, and history reweighting. This is not a known multiplicative law; it only reminds us that the two layers cannot cancel each other out. When the data are very sufficient and some rule already occupies overwhelming static mass at the relevant loss section, optimizer differences may look small at the hard-function level; in underconstrained, rare-event, or entry-difficult regions, `R_O` can dominate the observations.

### 3.11 Consensus Symbolicity, Full-Target Volume, and Measure Closure

#### 3.11.1 `Qopt proportional to V times R` Is Only a Ledger, Not a Theory

External review first pointed out:

$$
Q_O(f\mid D,L)
\propto
V_\mu(f;D,L)R_O(f;D,L)
$$

If `R_O` is only defined after the fact as `Q_O/V_mu`, the equation always holds yet has no independent predictive power. Its role is to prevent conflating static mass with optimizer transport, not to explain why `R_O` has a particular shape. This document therefore calls it a layered ledger: the real empirical content comes from measuring `V` and `Q` separately and finding when they are close and when they deviate; the microscopic law of `R` remains an open problem.

#### 3.11.2 [E18] Does High Agreement Point to Human-Readable Rules

E18 raises a teacher-independent conjecture: if a partial training set makes the complete functions of different initializations agree at nearly 1, does it usually correspond to an extractable short symbolic rule? This is not a trivial rewriting of MDL, because the network-relative complexity could in principle prefer a program that is Kolmogorov-short yet hard for humans to interpret; whether neural complexity and human symbolic complexity align must be measured experimentally.

[E18](experiments/experiment_18_consensus_symbolicity/README.md) first scans 540 8-bit data sets, 64 seeds each. Of the four strictly high-consensus data sets, three come from hidden teachers; the only teacher-free random n=12 training set makes 61/64 seeds converge to the same complete function, with collision 0.908.

The main mode depends on only 4 variables and can be written as a short 2-CNF/signed-threshold rule; the training set has no compatible 0--3-variable junta, so it attains the lowest essential-variable count in the version space.

Across all 512 random training sets, agreement is significantly negatively correlated with ROBDD, essential variables, ANF, decision tree, and influence.

Cross-capacity interventions show that the symbolic skeleton is stable while the exact function mass is protocol-dependent. Adding one high-disagreement sample consistent with the canonical rule makes five tanh networks almost all converge to the canonical function; giving the same point a conflicting label instead eliminates the original rule and forms a broad function cloud.

Re-adding to training a low-disagreement consistent sample — one that all seeds of a base end state already hard-predict correctly — can still significantly change the final distribution, further refuting the training model of "first obtain the final hard posterior, then apply zero-one filtering".

A large-scale scan of 8,192 random n=12 training sets, through 64-seed screening and 4,096 fresh-seed confirmation, yields 46 strict high consensuses. The preregistered Tier is 4/31/11/0; a post-hoc LP audit finds that 46/46 are equal-absolute-value-weight signed thresholds, and all 120/120 loose candidates are also linearly separable.

No high-consensus complex counterexample is observed, but the unified signed-threshold identification is a post-hoc finding and cannot be passed off as a preregistered conclusion.

E18 supports a one-directional empirical proposition: in this 8-bit MLP protocol, high complete-function consensus systematically corresponds to human-readable low symbolic complexity among the candidates compatible with the training set. It does not prove that all networks are like this, and it does not support the converse that "simple rules necessarily produce high agreement".

#### 3.11.3 [E19] Measuring the Rule's Own Volume Curve with the Full Truth Table

To avoid the rule ambiguity induced by partial training sets, [E19](experiments/experiment_19_full_truth_rule_volume/README.md) directly specifies parity1--4, majority3, and one preregistered random balanced function with the complete 16 points each, and runs 8-replica constrained SMC under the `4 -> 16 x 2 -> 1 tanh` uniform parameter measure.

The four parities strictly satisfy `V_p1>V_p2>V_p3>V_p4` over the jointly completed BCE 0.68--0.55, with non-overlapping replica intervals; each additional participating bit approximately adds a constant cost to `-log V` at fixed loss.

The stronger contrast comes from majority/random: the two are almost identical over 0.68--0.55, majority overtakes after 0.50, and by 0.40 they differ by about `5.6e3` times, by 0.30 about `1.5e18` times. Lower loss pulls structure targets that are approximately equal in the shallow region apart by a gigantic volume difference, directly refuting the separable common-loss-factor picture.

The result also produces a necessary correction: parity2, which looks extremely short to humans, in this tanh reference machine has a smaller prior and target-loss volume than the preregistered random function. Machine-relative complexity includes spectral order and architecture alignment, and cannot be directly equated with gate-circuit or ordinary Boolean Kolmogorov length.

Lineage degeneracy also occurs in the complex-parity deep tail, so E19 only confirms the ordering over the common interval and the majority/random mid-range separation, and does not pretend to have completed absolute volumes below 0.04.

#### 3.11.4 A Complexity Flow Still Exists Inside a Hard Function

A subsequent attempt to observe internal grokking on 100% modular-addition data exposed the fundamental boundary of the hard ID: when the complete mapping is entirely the training set, all hard-exact networks are completely identical at the ID level, yet the representation, margin, Fourier structure, and stability to perturbation can still continue to change.

That round of SMC sampling did not reliably enter the target deep tail, and the internal probes did not form cross-seed confirmatory results, so no experiment number is established; the methodological conclusion it provides is formally carried by the later hard-margin bridge and the parity4 deep-tail crossing.

At the time, the theory once contracted to two principles: the initialization measure prefers certain simple functions; under training constraints, complex realizations contract faster with loss. E19 provides strong evidence for the second, but the cross-family reversal and the later volume crossing show that "faster" must be understood as the complete loss-resolved profile, and cannot be frozen into one global scalar ordering.

#### 3.11.5 [E20] Full-Rule and Fixed-D Are Joined by a Strict Measure Bridge

The next question: why can the full-target self-loss volume and the candidate-function cell mass under a fixed partial training set simultaneously exhibit hundred-order-of-magnitude gaps and significant competition? [E20](experiments/experiment_20_measure_closure/README.md) no longer relies on analogy, but uses strict set identities.

Constant leave-one-out first compares 15 constant-0 training points, the complete constant 0, and the single-point exception rule. Taking `delta=16 epsilon/15`, the complete loss event must be a subset of the subset event, so the volume ratio of the independent full SMC must equal the conditional crossing-event ratio of the subset particles. At the 7 thresholds where both sides are measurable, the maximum log-ratio residual is about 0.071, closing the three-way relation.

The balanced AND n=10 repeats the experiment for D440/F040/D040/F440, using `delta=16 epsilon/10`, with a maximum closure residual of about 0.049 in the measurable region. This confirms that `V_full` and `Omega_D` are not the same quantity, yet belong to the same parameter measure and satisfy a conditional-probability decomposition; the two observations are not contradictory.

Finally, from the same `L10<=0.065` parent distribution, four branches are split by hard ID; keeping the training loss and the hard cell unchanged, only the maximum BCE margin on the 6 heldout points is tightened. The `tau=ln2` starting point reproduces the parent hard posterior with TV about 0.0086; then the absolute core volumes inside the four cells separate continuously, and D440 gains an overwhelming advantage.

Tens to hundreds of orders of magnitude of difference can therefore mainly form inside the continuous logit/margin cores within the same hard function.

E20 upgrades full-rule volume from an uncalibrated proxy to a measurement with a strict bridging basis; it also retains the boundary: it has not proven that an independently defined complex function necessarily contracts faster under any fixed D, and the complexity direction still needs external criteria from the subsequent Neural K-profile and independent sample-transition predictions.

### 3.12 Agreement Becomes an Instrument for Intervening on Function Condensation

#### 3.12.1 From "Amount of Disagreement" to an Operational Identifiability Threshold

The teacher-free result of E18 gives agreement a clearer status. For data of unknown meaning, if different initializations give highly consistent complete predictions on an independent probe, one can judge without an external teacher: the current training set and protocol have already compressed the function distribution into a narrow region. Conversely, low agreement only says that the candidate extensions have not yet condensed, which may come from insufficient data, label noise, a multimodal terrain, or optimization inaccessibility; it cannot diagnose the cause by itself.

"Data is enough" must therefore be operational: relative to a fixed architecture, training budget, seed count, probe range, and agreement threshold, the predicted distribution is narrow enough. It does not guarantee external correctness, exactness over the whole input space, or that the distribution will never fork again after the loss is pressed deeper. The grokking transition point likewise depends on a threshold and can be written as

$$
n_c(\tau)
=
\min\{n:\mathbb E_D[A_{\Pi,D}]\ge\tau\},
$$

where the training-loss depth and the random-training-set average should also be specified. The later `n50/n90` are exactly such relative definitions.

This is consistent with the picture that finite data support only finite loss precision. Crossing some transition threshold only means the rule direction already dominates at the current scale; going deeper in loss may still expose extensions not distinguished by D. Adding random samples, especially samples with currently high prediction entropy or low agreement, pushes the train/validation separation point toward deeper loss.

#### 3.12.2 [E21] Using Label Choice to Control Condensation or Maintain Divergence

[E21](experiments/experiment_21_agreement_complexity_frontier/README.md) starts from random balanced 8-bit, n=8 training sets.

Each round selects the unseen input whose committee prediction is closest to 50:50, and retrains with paired initializations for labels 0/1 respectively; anti-consensus chooses the branch with lower final agreement, pro chooses the higher branch, and random chooses at random. Label decisions do not read symbolic complexity.

After 16 starting points are completed to n=24, the 512 fresh-seed results are: unseen agreement of anti/pro/random is 0.96977/0.99904/0.97599, and the exact modal median is 0.0156/0.9990/0.0459. Anti is lower than pro in 16/16, and lower than random in 12/16.

Under the narrow-posterior gate, anti is 0/16, pro 15/16, and random 0/16. Agreement can therefore not only be observed, but also causally controlled by the path of newly added labels.

The 15 stable pro functions are all general linear thresholds under post-run LP audit, of which 7 are equal-weight and 8 are small-integer weighted thresholds. The only one that fails is also exactly the only path that did not truly converge, so one cannot use the pointwise majority function of a broad function cloud as a complex counterexample.

This experiment also explains why "a new sample that all current end states answer correctly" can still change the training result: hard-sign agreement does not mean the BCE at that point is zero, and a newly added sample changes the gradient path from initialization and early training onward. A sample looking certain under the final hard posterior therefore does not mean its information gain on retraining is strictly zero.

#### 3.12.3 Adversarial Prefixes Trace Out a Program-Complexity Frontier

To avoid small n repeatedly discovering only a few huge short-rule basins, E21 shares 16 anti trunks, freezes them at `n=8,16,24,32,48,64`, and then switches to pro until re-condensation. The 96 endpoints are confirmed with 512 fresh seeds, giving 80 fresh-narrow functions. The longer the anti prefix, the more pro samples re-condensation overall requires, with Spearman 0.433 between cutoff and extra sample count.

The endpoint complexity rises systematically: the mean number of essential variables rises from 5.06 to 7.91, the median ANF terms from 24 to 100, and the median ROBDD nodes from 10 to 29; within the 11 paired starting points that jointly survive, all three quantities increase in 11/11. The fraction of general linear thresholds drops from 100% to 0%.

The 31 candidates unexplained by the preregistered language are 31/31 representable under a post-hoc second-order PTF audit; all 80 narrow endpoints belong to polynomial thresholds of degree at most 2, while 0/64 of the 64 random balanced functions belong to that family. The median number of quadratic support terms rises from 6.5 to 31. The experiment did not obtain high-consensus random look-up tables, but formed a measurable program frontier from "sparse linear thresholds to denser quadratic thresholds".

The quadratic PTF is a unified finding made after seeing the candidates and cannot count as a preregistered hit. What E21 supports is the empirical relation under a bounded protocol: the data budget required for agreement convergence and the symbolic complexity of the stable endpoint rise together; it has not proven that all high-consensus functions are human-readable.

#### 3.12.4 Practical Implications and the Boundary of Conceptualization

This instrument has direct application potential: scanning for the inputs of greatest committee disagreement, requesting labels from an external oracle, then retraining and checking global agreement can eliminate the currently competing functions more efficiently than random sample addition. It is isomorphic to model-vulnerability patching, adversarial-example discovery, continual learning, and active data-set maintenance; but every sample addition reshapes the global terrain, may fix old errors and create new ones, and cannot be packaged as a monotone local patch.

"Human-readable" in this document means that the function can be extracted as a short, strict, executable symbolic program, not that a researcher can necessarily discover or intuitively understand it easily. The deeper meaning of E18--E21 is: at least on the measured natural rule families, the protocol-relative complexity of MLPs and human symbolic compression have a significant intersection. This gives "why finite systems form concepts" a candidate macro-level explanation: reusable concepts can shorten descriptions, transfer intermediate computations, and save capacity.

Extrapolating the same picture to human cognition or LLMs remains discussion. An LLM can be understood as compressing conditional relations between tokens, whose success is compatible with the reuse efficiency of the human concept system; but the Boolean consensus experiments of this document do not directly prove that large models or brains adopt the same microscopic mechanism.

### 3.13 Per-Sample Free Energy Becomes an Exact State Function

#### 3.13.1 [E22] From Order-Dependent Surprise to Path-Independent Endpoint Cost

The earlier sample-by-sample addition experiments had repeatedly shown: the more unexpected a new label is to the current committee, the more it rearranges the function distribution; but the information content of each step depends on the sample order, and no invariant of the total difficulty had been found. E22 rewrites the problem as a partition function over the same static reference measure. For a partially labeled state `S`, define

$$
J(S)=-\log_2 Z(S),
$$

and write the generalized cost increment, in bits, of adding sample `i` as

$$
\Delta J(i\mid S)
=J(S\cup\{i\})-J(S)
=-\log_2\frac{Z(S\cup\{i\})}{Z(S)}.
$$

As long as all states use the same reference measure and the same definition of `Z` (with $\beta$ fixed in the Gibbs case), the increments of any sample order telescope to the difference in $J$ between the complete-rule endpoint and the empty state. This does not require every step to be equal; the order merely redistributes the total cost among the stages.

[E22](experiments/experiment_22_static_free_energy_information/README.md) uses 2,097,152 prior networks to save the logits of all 8 points of 3-bit at once, and exhausts the `3^8=6,561` partially labeled states, the 256 complete rules, and the `8!=40,320` addition orders of each rule.

Under the hard-conditioned and several Gibbs calibers, the maximum path-invariance error is `2.84e-14 bit`, and the Shapley efficiency error is `5.68e-14 bit`. Rule150, i.e. parity3, ranks the hardest at about 29 bit under the hard caliber, and ranks second under the `beta=1` caliber.

Per-sample generalized cost increments and complete-rule cost are therefore described by one static state function. Under pure hard conditioning or a standard likelihood at $\beta=1$, this also directly connects predictive surprise and the prequential code; general Gibbs scores require additional label normalization. The real empirical content here is the difference in endpoint costs under the network reference measure and the numerical closure; "the increments of the same potential function are path-independent" is itself a strict identity, is not passed off as an empirical discovery, and does not imply that real SGD paths are conservative.

#### 3.13.2 The Connection to Statistical Mechanics and Its Historical Position

The same structure can be written in canonical form:

$$
Z_{\Pi,D}(\beta)
=\int \exp[-\beta nL_D(\theta)]\,d\mu_\Pi(\theta),
\qquad
F_{\Pi,D}(\beta)=-\frac1\beta\ln Z_{\Pi,D}(\beta).
$$

Here $J_{\Pi,D}=-\log_2Z_{\Pi,D}$ and $F_{\Pi,D}$ have different scales. At fixed $\beta>0$:

$$
\Delta J=\frac{\beta}{\ln 2}\Delta F.
$$

Let $r_\beta(y\mid x,D)=Z_{\Pi,D\cup\{(x,y)\}}(\beta)/Z_{\Pi,D}(\beta)$. This is a Gibbs branch score, not generally a normalized label probability. If $p_\beta(y\mid x,D)=r_\beta(y\mid x,D)/\sum_{y'}r_\beta(y'\mid x,D)$, then

$$
-\log_2p_\beta(y\mid x,D)
=\Delta J+\log_2\sum_{y'}r_\beta(y'\mid x,D).
$$

For pure hard branches or a standard negative log likelihood at $\beta=1$, the label scores sum to 1, allowing the increment to be read directly as predictive surprise; telescoping itself does not supply this normalization condition.

The sublevel volume mainly used in this document is a microcanonical object; the Gibbs partition function is a Laplace-type summary of the entire loss density of states. The two languages are different summaries under the same reference measure, and do not guarantee that a volume crossing within a threshold window is reproduced pointwise in the canonical ensemble. E24 compares mass curves for separate complete label targets, not restricted branch partition functions within one finite training event. The "temperature" here is a mathematical control variable of loss tolerance, and does not mean SGD is really in thermal equilibrium.

This connection was not first proposed by this document. In 1989, Esther Levin, Naftali Tishby, and Sara A. Solla's [*A Statistical Approach to Learning and Generalization in Layered Neural Networks*](https://mlanthology.org/colt/1989/levin1989colt-statistical/) already wrote a fixed network architecture as a Gibbs ensemble, expressed the prediction cost of a new sample as a free-energy difference, and connected predictive MDL with generalization.

This document does not claim to have invented "neural-network free energy"; the difference is that, with modern deep networks and reproducible experiments, it directly measures loss-resolved complete-function cells, fixed-D candidate mass, margin cores, optimizer transport, and the measure closure among them, and connects these quantities to the grokking data transition.

#### 3.13.3 When Adding a Consistent Sample Guarantees an Increase in Target Mass

Under pure hard conditions a clean conclusion can be given. If the complete function `f` is consistent with the newly added sample `s`, denote the current compatibility event by `C_D`, the new constraint by `C_s`, and the function cell by `H_f`; then

$$
H_f\cap C_D\cap C_s=H_f\cap C_D,
\qquad
C_D\cap C_s\subseteq C_D.
$$

The unnormalized mass of `f` is therefore unchanged, the denominator can only decrease, and its hard-conditioned relative probability does not decrease. This strictly explains "adding rule-consistent evidence eliminates competing extensions".

But at a fixed raw-BCE threshold, a newly added sample also changes the average loss and the sublevel set, so the numerator of the target cell also changes; there is no analogous theorem of "necessarily monotone increase at arbitrary epsilon". Empirically, a newly added rule sample often first pushes up the target advantage in the deep-loss region, and then gradually extends the rule-aligned channel toward shallower loss; E11 and E23 in the next section support this picture, but this document does not write it as an unconditional mathematical law.

### 3.14 Neural K-profile, Deep-Tail Crossing, and MNIST

#### 3.14.1 [E23] Full-Target Volume Prospectively Predicts Random-Data Transitions

E23 first accepts a key verdict: can the static volume curve of a complete rule predict how many samples an independent random training set needs to recover that rule. After fixing the `8 -> 16 x 2 -> 1 tanh` network, stage A, before reading any random-subset training results, freezes the volume scores of eight rules over BCE `0.690 -> 0.670` with the complete 256-state SMC; the prediction file SHA256 is `0fd10563cc7059d3d1925b883f27c07a5c4d409a92a281cca2383f611f9519f9`.

Stage B independently trains 19 non-complete sample sizes, 8 rules, 64 random data sets per condition, and 24 paired initializations, for 9,728 random-subset conditions in total, plus 8 full-data qualification checks. `n50/n90` are defined as the sample sizes at which the isotonic recovery rate across random training sets first reaches 50%/90%.

The confirmatory parity results hit the preregistered ordering exactly:

| Rule | volume score | `n50` | `n90` |
|---|---:|---:|---:|
| parity1 | 125.21 | 24 | 48 |
| parity2 | 3266.21 | 64 | 80 |
| parity3 | 4618.60 | 96 | 112 |
| parity4 | 6719.58 | 160 | 160 |

The four transition grid intervals do not overlap, and the Spearman coefficients are 1.0/1.0. For the first time, full-target static geometry thereby prospectively predicted the rule-identification data size on random partial training sets, rather than renaming "difficult" after the training results appeared.

Because each $n$ already contains 64 random training sets, continuous agreement can be used for a posterior sub-grid estimate on the coarse grid. Skipping wrong-constant high consensuses, the 0.99 crossing is read from the target-aligned re-condensation branch after the agreement minimum, requiring unseen target accuracy of at least 0.90; 2,000 bootstraps give parity2 59.98[58.77,61.07], MUX3 69.47[65.02,72.40], parity3 88.94[85.72,91.05], parity4 151.93[149.65,153.81], and random is >240 with 100% right censoring. On the paired data at the same `n=64`, parity2 agreement is 0.0106 higher than MUX3, with 95% interval [0.0078,0.0134]. This analysis is a posterior resolution enhancement and does not change the preregistered main verdict.

Across all eight rules, the correlation remains high: Spearman 0.898/0.868; but three groups of **shallow-score misorderings** appear: MUX3/parity2, and random versus parity3/4. This first refutes that a shallow-loss single slope is a sufficient complexity scalar across arbitrary function families; the later Gaussian deep tail further shows that the full profile and `n50/n90` should never have been treated as the same definition.

#### 3.14.2 Local Speed, Absolute Volume, and Ranking All Depend on the Loss Scale

The multi-threshold curves saved in stage A further show that the local contraction rate of random-balanced increases rapidly as the loss deepens, successively overtaking parity2, parity3, and parity4; its absolute volume ranking is also overtaken by parity2 and parity3 in turn. Using the local speed over a deeper window to predict `n50/n90`, the Spearman rises to 0.970/0.952. The Gaussian deep tail then confirms that the direction of random relative to parity3/4 agrees with the transitions, but the full-target volume of MUX3 remains stably larger than parity2, while the uniform random-data transition order is the opposite. The latter reveals the missing fixed-$D$ denominator, sample coverage, and optimizer layer in between, which can no longer be automatically removed by "reading a deeper profile".

The subsequent coverage/shortcut interventions give direct causal evidence. Under uniform, parity2/MUX3 have `n50=64/80` and `n90=80/104`; after strict balancing of the eight cells, they become `56/72` and `64/88`, with MUX3 still later; after raising the selector-conflict samples to 75%, they become `n50=72/56` and `n90=88/72`. Relative to uniform, MUX3's two thresholds move earlier by 24/32 samples, while parity2 is instead delayed by 8 each; 2,000 paired bootstraps and the time-resolved readings over 500--40,000 steps all support this direction. This directly refutes "all targets share a uniform and equally strong noise/shortcut background", and proves that the operational transition can be reversed by the sampling method alone while the complete target and the network are unchanged.

The fixed-$D$ Gaussian SMC then removes the optimizer and still reproduces the same cross-target order in the static distribution. At `n=32,epsilon=0.02`, the parity2/MUX3 target mass under uniform is 0.266/0.000214, under cell 0.469/0.284, and under conflict 0.498/0.782; conflict-MUX3 is the modal target in 8/8 data sets, while uniform-MUX3 is 0/8. The posterior errors are further localized to the selector-conflict cells: uniform has accuracy 0.993 on ordinary cells and 0.777 on conflict cells, while conflict raises the conflict cells to 0.995. Deep-tail lineage has fallen to about 1--2 per replica, so the absolute small numbers are only rough estimates; but conflict-MUX3 relative to uniform is in the same direction in 8/8 data sets and 32/32 replicas, so the order-of-magnitude verdict does not depend on a single mode. The static competition denominator thereby becomes the first-order cause of this reversal, and the optimizer only needs to explain the remaining quantitative deviation.

This is not a refutation of the framework, but a substantive correction of the early coarse statement: neural relative complexity is not a permanent scalar independent of precision. The complete object should be

$$
K^N_\Pi(f;\epsilon)=-\log_2V^{full}_\Pi(f;\epsilon),
$$

together with its local slope and curvature. The prior mass is only information at very shallow `epsilon`; any shallow secant is also only a segment of the profile. At this point it is formally called the **Neural Kolmogorov complexity profile**, or **Neural K-profile** for short. If a single scalar is needed, the loss precision, the integration functional, or an independent operational definition must be explicitly specified; one cannot pretend the whole curve naturally has only one number.

#### 3.14.3 [E24] Parity4 and a Single-Point Exception Really Cross in the Hard-Exact Deep Tail

To test whether the ranking is stable at lower loss, [E24](experiments/experiment_24_deep_neural_k_crossing/README.md) first uses `4 -> 16 -> 1 tanh`, a standard Gaussian reference measure, and pCN SMC to scan 54 complete rules, advancing from BCE 0.700 to 0.01913, crossing the hard-exact sufficiency boundary 0.04332.

The structured rules form a stable skeleton in the deepest five windows, and parity4 is the smallest-volume entry on the panel at all 25 thresholds; but the only single-point exception, `parity4_flip0000`, has a contraction rate of 159.49 in the final window, higher than parity4's 152.75.

Independent curves extrapolate the crossing in advance to about `epsilon=0.0025`. Then, advancing synchronously from a shared parent with 8 replicas of 32,768 particles each, the median crossing is directly measured as

$$
\epsilon_{cross}\approx0.002308.
$$

This result also closes with the earlier leave-one-out real-training preference: when `0000` is left out of the parity4 truth table, the majority prediction for the missing label favors parity4, not `flip0000`, which flips only at that point, although it does not reach 100%. After the deep-tail crossing, parity4's full-target volume is larger, in the same direction as the optimizer preference. What E24 proves is therefore that the volume profile itself reorders, and it provides a test case for how a shallow misordering disappears in the deep tail.

But leave-one-out is only a local label branch under an $n=15$ fixed training set, not a 4-bit random-training-set recovery curve, and cannot define `n50/n90` by itself. The strict relations should be written as: E20 establishes the conditional measure bridge from full-rule volume to fixed-$D$ candidate/held-out branches; E23 establishes the direct prospective correspondence from the full profile to the random-training-set average transition; E24 only provides additional consistency evidence for the first half. The three are related but cannot replace each other.

At the deepest threshold 0.000952, the median volume ratio has already reversed by about `2.0e7` times; 7/8 replicas have absolutely overtaken, and 8/8 replicas continue to separate along the same direction. The strictly preregistered stopping criterion of 8/8 for five consecutive windows has not yet been met, and the main text retains this boundary.

When the crossing occurs, the two hard IDs have each long been 100% fixed, and the internal representation indicators also change smoothly. The result directly proves: the hard function is not the finest state of the complexity flow; the deep-tail cost of a non-reusable single-point exception can manifest later than the architecture-mismatch cost of parity. This document has not proven that any two profiles cross only finitely many times, but the strong proposition of "permanently ranking functions by a shallow slope" has been ruled out.

#### 3.14.4 The Random-Training-Set Transition Is an Independent Identification Complexity

The curve is the most complete static definition; if a single reproducible experimental number is needed, one can use the random-training-set transition under a fixed protocol:

$$
n_q^\Pi(f)
=\min\left\{n\in\mathcal G,\ n<N:
\Pr_{D_n\sim f}[\mathrm{Recover}_\Pi(f\mid D_n)]\ge q
\right\}.
$$

Here `Recover` must freeze the training budget, loss depth, seed count, complete-function mass, and agreement threshold, and $\mathcal G$ is the preregistered grid of non-complete sample sizes. `n50/n90` are two instances of this quantity, but a point value only represents an interval crossing between adjacent grid points; bootstrap intervals should be reported together. If there is still no crossing by $N-1$, it is recorded as right-censored, and the complete-truth-table fit at $n=N$ cannot be used as a substitute for the generalization transition. Extremely small spaces such as 4-bit are dominated by the coarse grid, the holdout position, and the whole-space cap, and are unsuitable for finely ranking the hardest rules with this scalar. It is not equal to machine-independent Kolmogorov complexity, but when the space is large enough it has an MDL-style meaning: the harder a rule is to identify in the current neural language, the more random evidence is needed on average to eliminate the competing extensions.

E23 used a volume profile frozen in advance to predict this independent quantity, thereby avoiding the pure circle of "defining complexity by training difficulty, then explaining training difficulty by complexity".

#### 3.14.5 [E25] Static Branches Predict MNIST and Reconstruct the NLL U-Shape

Finally the method is moved out of the Boolean enumeration space. [E25](experiments/experiment_25_mnist_static_prediction/README.md) uses 7x7 MNIST and a `49 -> 32 -> 1 tanh` network.

Stage 0 first trains 512 trajectories to calibrate data sufficiency: with the same four balanced samples, the median test accuracy of `0 vs 1` is 98.31%, while `3 vs 8` is only 69.66%; at the 95% threshold, the former has already crossed at `n=4`, while the latter only at `n=512`.

Stage 1 freezes two `n=4,dataset0` conditions and nine raw-BCE sections, then compares the two label branches of each unseen image from the same static parent. The test hard accuracy of 0/1 rises from 92.97% to a peak of 96.35%, and 3/8 rises from 76.69% to about 78.13%.

The static soft NLL forms a U-shape on its own: the 0/1 lowest interval is one preregistered grid cell away from the SGD minimum, and the 3/8 lowest interval directly covers the train-loss location of SGD.

As the loss deepens, the function concentration of both 0/1 and 3/8 rises; but after 3/8 accuracy stalls, the true-label mass of the persistently wrong samples still decreases, forming more confident errors. Accuracy, NLL, and agreement can therefore turn at different loss scales. The low-loss volume difference between the two tasks expands from 0.56 to 55.43 decimal orders of magnitude, in the same direction as the sample sizes `n=4 vs 512`.

This experiment is calibrated-then-frozen rather than a blind test on a brand-new task, and it only covers two binary classifications and a tiny MLP; but it proves that loss-resolved static volume can make per-sample predictions on real images, and that the classical validation-loss U-shape has at least one geometric source that already forms before the optimizer.

#### 3.14.6 [E26] Balanced Unlabeled MNIST Partitions

[E26](experiments/experiment_26_mnist_balanced_label_volume/README.md) fixes 5 digit-0 images,
5 digit-1 images, one label-0 anchor, and a 5:5 ratio, and blind-tests static volume over the
126 candidate partitions. In two independent evaluation panels, the natural partition ranks 126
at loss 0.8; by 0.6 both rise to top1 simultaneously; at 0.3 the mean normalized volume mass
reaches 0.99972. It proves that the natural-grouping preference can reverse from last place to
monopoly along the loss axis, but the class ratio itself already provides information, so this
cannot be called fully unsupervised learning.

#### 3.14.7 [E27] Unseen Agreement Before and After Grokking

[E27](experiments/experiment_27_grokking_agreement/README.md) re-analyzes the 32-seed
trajectories of E15 Mod97, excluding the training inputs before measuring agreement. The four
data levels have unseen agreement at hard fit of only 0.027--0.033, far from 1; the 90% training
condition ends with unseen accuracy 0.916 and agreement 0.839. In the middle and late stages,
agreement is almost entirely explained by different seeds jointly answering the target
correctly, and the remaining errors are mainly seed-specific. The result refutes the simple
version of "one same complete wrong function is already stably shared before grokking", while
preserving the local phenomenon of an early, brief shared output bias.

#### 3.14.8 [E28] Full-Network HMC and Adam on 50k MNIST

**The evidence type is a likelihood-weighted posterior comparison, not a hard-threshold volume validation.**
The target is a Gaussian prior multiplied by the full training-data likelihood,
with mean cross-entropy weighted by beta=N=50,000. This weighting redistributes
relative mass inside the low-loss region; it is not conditioning at a fixed loss
threshold, nor can proximity between the two be assumed.

[E28](experiments/experiment_28_mnist50k_hmc_adam/README.md) runs 16-chain full-network HMC on
all 50,000 MNIST training images on a 4,266-parameter small CNN, and compares it with 32-seed
plain/MAP Adam. The ensemble test accuracies of HMC, plain Adam, and MAP Adam are 99.03%,
98.95%, and 98.82%; HMC and plain Adam differ by only 8 test images, with McNemar p=0.2005,
and should be judged at the same performance level. The 480 HMC samples correspond to 480
different complete test functions, yet the pointwise agreement reaches 0.9877. Cross-chain
function disagreement is still higher than within-chain, so prediction success does not equal
a proof of global mixing.

This implementation uses gradients and estimates neither threshold volume nor Neural K.
Thus 99.03% belongs only to the ensemble prediction from the specified posterior sampling;
it is not direct evidence that a hard-threshold ensemble is equally accurate or that the
volume-contraction law holds on full MNIST.

#### 3.14.9 [E29] Loss-Resolved Stability of a Dead Bit

[E29](experiments/experiment_29_dead_bit_loss_resolved/README.md) keeps one input bit
constantly 0 across all 8 training states, then compares static SMC, 512-seed no-decay Adam,
L2/MAP, NNGP, and the temperature-1 posterior on five Boolean functions. The dead column keeps
variance about 1 in both static SMC and no-decay Adam; but the static z=1 strictly correct mass
rises to 0.9959--1.0000 as the loss deepens, and hits matched-loss Adam with an absolute
difference of 0.0010 on average and 0.0042 at worst. L2 obtains a different robustness by
pressing the dead weight to 0. The standard temperature-1 posterior does not contract
sufficiently at 8 samples, while NNGP passes z=1 under a small ridge. The stronger z=2 shift
re-exposes a static/optimizer difference of 0.15--0.18.

#### 3.14.10 [E30] Weight Decay Reshapes the Static Function Landscape

[E30](experiments/experiment_30_weight_decay_static_landscape/README.md) compares the static
complete-function mass under explicit L2 in a 433-parameter balanced-AND network. The raw BCE
means of three approximately matched-BCE conditions are 0.0025815, 0.0025105, and 0.0025898;
when $\lambda$ rises from 0 to $5\times10^{-5}$ and $10^{-4}$, the complete AND mass rises from
53.683% to 58.423% and 73.318%, and the number of observed complete functions falls from 4190
to 1246 and 86. The endpoint difference is 19.635 percentage points, in the same direction in
16/16 replicas. At deeper $J$ levels all observed modes are AND, but the absolute logV has not
converged, so they serve only as supplements.

Without L2, AND already owns more than half of the static mass, showing that grokking-style
rule takeover does not need weight decay. This experiment does not add a new theoretical
element; it clarifies the original "loss landscape + optimizer" layering: weight decay written
into the loss only co-shapes the same landscape, and is neither mysterious nor special. This
round's enhancement effect is already explained by the static landscape change; the conclusion
does not directly extrapolate to decoupled AdamW or the original Mod97 Transformer.

## 4. The Theoretical Picture to Which the Experimental Chain Finally Converged

### 4.1 Minimal Principle: The Network Only Optimizes What Is Written into the Objective

The most important conceptual clarification of the whole document looks trivial: what standard supervised training directly pushes is the training objective `L_D`, not the generating rule in the researcher's mind, validation accuracy, generalization, compression, or human readability. Without explicit regularization, the only scalar pressure continuously applied in the training program is the training loss; if regularization or auxiliary terms are written into the same scalar loss, they also only co-shape this landscape and are not special. Any generalization, feature reuse, symbolic semantics, or MDL-style preference must be examined as a result jointly produced by the loss landscape, the network geometry, and the optimizer.

This also explains why "the data are generated by Rule30" cannot be swapped into "the training objective is Rule30". A partial training set only defines the set of compatible extensions and the sample-frequency weights; when samples are few, unbalanced, or the input space is small, the researcher-assumed generator may well not be the dominant explanation under the current protocol.

### 4.2 Two Static Empirical Principles

Under a fixed neural reference protocol `Pi`, the experimental chain finally leaves two minimal static principles.

**Principle one: the reference function mass is highly nonuniform.** The many-to-one mapping from parameters to functions gives different functions extremely different reference masses; natural architectures typically produce a simplicity preference for certain reusable, low-order, or aligned functions. "Simple" here is protocol-relative and cannot be directly replaced by human gate-circuit length.

**Principle two: tightening the raw loss produces function-specific differential contraction.** In general there is no

$$
\Omega_D(f;\epsilon)=A_D(f)B_D(\epsilon)
$$

common loss factor. Different functions, and even different continuous realizations inside the same hard function, lose mass at different speeds as `epsilon` decreases; the speeds themselves can also change and cause ranking crossings. The training loss is therefore not only a binary filter of "whether the samples are satisfied", but also a scale parameter that continuously raises realization precision and rearranges relative mass.

These two principles are weaker than "the lower the loss, the more the probability of a prespecified simple function monotonically rises everywhere", yet they are compatible with all counterexamples and retain the core explanatory power. A concrete hard function can first increase and then decrease; the truly stable object is the complete loss-resolved distribution, not a prespecified permanent winner.

### 4.3 Four Distributional Objects That Must Not Be Conflated

1. **Reference prior.** The zeroth-order function mass induced by `mu` through the parameter-to-function mapping;
2. **Static loss section.** `Qstatic` at fixed `D,epsilon`, or the full-target `Vfull`;
3. **Optimizer ensemble.** The `Qopt(t)` obtained by pushing the initialization distribution forward through the training map;
4. **Single-trajectory time samples.** The sequence of states visited by one seed along training time.

The reference measure is the ruler for measuring the static geometry, not the whole source of the loss landscape. The architecture, parameterization, input encoding, loss, and training set determine the energy function; `mu` determines how we assign mass to its regions. The real optimizer additionally determines where to enter, along which flow tubes to move, and which regions are reachable.

What was written down early as

$$
Q_O(f\mid D,\epsilon)\propto V_\mu(f;D,\epsilon)R_O(f;D,\epsilon)
$$

is only a layered ledger. If `R` is defined after the fact as the ratio, the equation has no explanatory power. Both E17 and E25 show that static mass has strong first-order predictive power but is not equal to AdamW; E16 shows that a good deep-tail endpoint can exist and be locally stable while being globally unreachable from random initialization.

### 4.4 Neural Kolmogorov Complexity Is a Profile, Not a Permanent Scalar

This document calls

$$
K^N_\Pi(f;\epsilon)
=-\log_2V^{full}_\Pi(f;\epsilon)
$$

the Neural Kolmogorov complexity profile under protocol `Pi`. It turns "how rare a parameter mass this network language must pay to realize the target `f` to precision `epsilon`" into a measurable object. The value at some fixed loss, the local slope, the curvature, and the deep-tail asymptotic behavior are all projections of the profile and cannot replace one another; the hard prior is the separate baseline distinguished in Section 2.3.

It differs from the machine-independent Kolmogorov complexity `K_U(f)` in three respects:

1. changing the architecture, encoding, activation, scale, loss, or reference measure changes the profile;
2. inside a hard function there is still continuous margin and representation complexity;
3. there is currently no neural coding theorem that strictly equates `-log volume` with shortest program length.

But "Neural Kolmogorov" is not an arbitrary name. `-log` mass has information-cost form and admits a code-length interpretation when coding conditions hold; E22's per-sample $\Delta J$ increments strictly accumulate to the complete-rule endpoint cost; and E23 used a profile frozen in advance to predict the `n50/n90` of independent random training sets. Under the normalization and coding conditions above, this connects volume cost, predictive code, and identification sample size.

**Defining** `-log volume` as protocol-relative difficulty itself can of course not be overturned by experiment; the falsifiable scientific content is whether it can predict quantities outside the definition. This document uses three external anchors: rule families with strict inclusion relations fixed in advance, random-data transitions measured after freezing the profile before training, and real training's preferences over unseen functions/images.

If all three systematically reversed, Neural K would be reduced to a name and lose explanatory power; E10, E23, and E25 give positive results, and cross-family reversals delimit its scope as they are.

If a single training operational quantity must be given, one can report the random-training-set average `n50/n90`, freezing the sampling distribution, recovery threshold, and training protocol at the same time; but it should be called the protocol-relative identification/recovery sample complexity, not a synonymous scalar of full-target Neural K. Rules of similar difficulty are allowed to have overlapping confidence intervals; the profile, the fixed-$D$ denominator, and optimizer transport must still be reported in separate layers.

### 4.5 Free Energy, MDL, and Algorithmic Information Theory in One Language

The static density of states `rho(E)`, the microcanonical volume `V(epsilon)`, and the canonical partition function

$$
Z(\beta)=\int e^{-\beta nL_D(\theta)}d\mu(\theta)
$$

are different summaries of the same loss geometry, without guaranteeing pointwise agreement of local orderings or crossings between the two ensembles. At fixed $\beta$, $\Delta J=\beta\Delta F/\ln2$ gives the generalized bit cost of a new sample; it directly equals predictive cost only when label normalization holds. Summing all $\Delta J$ increments gives the endpoint cost difference. E20 also checked the conditional-probability bridge among full-target volume, fixed-D candidate cells, and hard-margin cores under explicit event nesting and threshold conversion.

Standard MDL is written as `L(H)+L(D|H)`. A neural network usually does not explicitly compute an additional program-length objective; it only lowers the empirical loss. This document's explanation is: the architecture and reference measure have already assigned nonuniform mass to realizations, and tightening the residual tolerance differentially eliminates these realizations, so a single loss pressure can produce MDL-style function selection. "Compression" is a result of the loss geometry, not a second optimization objective sneaked in.

In this correspondence, `-log2 V_full(f; epsilon)` is the protocol-relative event information cost for the current neural language to realize the complete target to precision `epsilon`, and the entire Neural K-profile is the cost curve as the residual precision varies. A function-code-length interpretation requires disjointness or suitable normalization and does not apply directly to generally overlapping full-target events. It shares with algorithmic probability the structure of "larger mass, lower negative-log cost", but it is not machine-independent Kolmogorov complexity. E22's endpoint-cost closure and E23's `profile -> n50/n90` prospective prediction are two independent empirical anchors of this conditional MDL/algorithmic-information interpretation beyond the naming level.

This also delimits the statement "loss decrease equals complexity decrease". When training learns any nontrivial task from constant output early on, the absolute program complexity can obviously rise; fitting noise also increases description cost. The reliable proposition is: under fixed training constraints and protocol, loss decrease rearranges the compatible realizations, and often raises the relative mass of reusable low-description realizations in the deep tail. Its direction must be measured by the profile, not presupposed by slogan.

### 4.6 Data Size Pushes the Rule Advantage Toward Shallower Loss

For pure hard conditioning, adding a sample consistent with function `f` does not delete `f`'s cell, but does delete some competing cells, so `f`'s normalized mass strictly does not decrease. There is no equally unconditional theorem for raw-BCE sublevel sets, but E11, E23, and more than one hundred synthetic tasks jointly support the following macroscopic picture:

1. **Very small n.** The training set is underconstrained, and a few huge but not necessarily generator-aligned simple extensions dominate;
2. **Medium n.** More candidates begin to compete, and agreement can first decrease; the target rule gains an advantage only at deeper loss, forming the grokking window;
3. **Sufficient n.** The rule-consistent constraints already overwhelm the alternative extensions at shallower loss, train/validation decrease in sync from the start, appearing as direct learning;
4. **Finite-precision boundary.** Any finite training set ultimately supports only finite depth. Continuing to press the loss down may re-expose training-set-specific extensions, causing validation NLL or accuracy to separate; adding samples pushes this separation point toward deeper loss, rather than creating a magical sample size that absolutely "never overfits again".

The grokking transition point is therefore not a single universal constant, but an interval jointly defined by `agreement/target mass/accuracy` thresholds, loss depth, architecture, and training budget. It still has operational value: the random-training-set average transition location can serve as a sample-complexity reading of Neural K.

### 4.7 Grokking, Shortcuts, Overfitting, and Noise Belong to the Same Distributional Flow

**Grokking.** Interpolation only means the hard constraints are satisfied, not that the function or the internal realization is frozen. Continuing to lower the raw loss raises margin precision and changes relative volumes; early memorization/shortcuts can be the most economical solutions at the loss scale of their time, until deeper loss makes them insufficient and the rule realization takes over. There is no need to assume the network has been secretly chasing the external true rule from the first step.

**Overfitting.** When D is insufficient to distinguish the externally correct extension at the required loss precision, deeper loss still lets the function distribution converge, but possibly onto a training-set-specific direction. E25's 3/8 shows that agreement continuing to rise, hard accuracy plateauing, and wrong NLL worsening can happen simultaneously.

**Label noise.** Early on, the rule direction coherent across samples is exploited first; after the training loss is pressed further down, non-reusable exceptions begin to dominate the remaining residuals, and external-rule performance declines. Early stopping is not a universal mysterious trick, but a choice of a shallower loss section under the current data noise and target definition.

**The classical U-shape.** It need not be entirely attributed to the optimizer trajectory. The static branch mass of a fixed training set can itself first improve unseen samples and then increase wrong confidence on persistently wrong points, thereby producing the validation NLL U-shape; the optimizer determines how the actual curve passes through this geometry.

### 4.8 Agreement Measures Condensation, Not Truth

One must distinguish pointwise agreement, complete-function collision, modal-function mass, and external target mass. High pointwise agreement means that under the current protocol most models give the same answer on the probe; it can correspond to the external target rule, to a constant rule that is wrong relative to the external generator but still extremely short and readable, or to a training-set-specific extension. Single-sample constant condensation is therefore not a counterexample to the readability conjecture; it only refutes "agreement automatically recovers the hidden teacher". Low agreement means the function distribution has not yet condensed, but cannot by itself distinguish insufficient data, noise, multimodality, or optimization inaccessibility.

E18--E21 support a one-directional conjecture still awaiting extension: in the measured 8-bit MLPs, strict high complete-function consensus systematically falls into short signed-threshold or quadratic PTF families; the longer the adversarial prefix, the more samples re-condensation needs and the higher the symbolic complexity, rising together. It suggests that neural relative complexity and human-extractable symbolic complexity have a significant intersection, but it has not proven that all high-agreement functions are human-readable.

If this correspondence can be maintained in more architectures and larger spaces, it would suggest a possibility stronger than "the network happened to pick a readable function": both neural complexity and human intuitive complexity are capturing robust compressible structures such as locality, symmetry, compositionality, and computation reuse. "Objective" here does not mean a unique absolute encoding, but that multiple effective representation languages give broadly similar orderings of a batch of reusable regularities. Parity and architecture reordering have already delimited that this alignment can only be partial. The more than one hundred deterministic rule tasks simultaneously exhibit extremely low unseen-sample loss and high function consistency when data are sufficient, providing breadth-side evidence for this; since they have not all undergone the same fresh-seed symbol audit, they cannot replace the direct tests of E18/E21.

As a tool, preferentially querying the samples with large committee disagreement, obtaining labels from an oracle, and retraining globally is usually more efficient than random sample addition. Every new sample reshapes the global terrain, and may fix old vulnerabilities while exposing new ones; this is the same problem as adversarial patching, continual learning, and active data maintenance, not a monotone process of local patching.

### 4.9 Feature Learning Is a Descent Mechanism, Not Generalization Itself

Shared intermediate representations, feature reuse, and semantic binding can reduce repeated computation under finite capacity, making it easier for the network to press the same training objective to lower loss. E08 shows that the shared-task benefit increases with prefix depth and weakens with width; E05 shows that even when a fixed NTK is already sufficient to fit, a finite network still actively reorganizes its representation.

Feature learning is therefore a dynamical mechanism for realizing more economical deep-loss solutions, but it is neither a synonym of generalization nor a necessary condition for all solvable tasks.

In the OOD rule experiments, the network interprets the same role bit as reusable semantics and composes it into unseen rule combinations, consistent with the picture that "stable semantic binding is more economical than per-combination look-up". This provides an experimental bridge for the formation of symbolic macrostates in connectionist systems: symbols are not written into neurons in advance, but can emerge as low-description programs reused across tasks. Level differences should be retained when extrapolating to human concepts or large models.

### 4.10 Capacity, Data, Training Depth, and Capability Emergence

Whether a capability appears is controlled by at least three kinds of conditions: the network capacity must be able to represent the corresponding realization; the data must identify the target direction to sufficient precision; and training must enter the loss region where that realization dominates and the optimizer can reach. Evaluation, moreover, often uses discrete thresholds to cut continuous probability changes into "can/cannot". Many emergence phenomena can therefore be understood without appealing to mysterious new mechanisms: capacity provides candidate programs, data move the volume-competition boundary, loss depth strengthens compression, and the readout threshold displays a continuous change as a sudden jump.

This is compatible with the picture that "large models gain capabilities by compressing tokens and concept relations", but this document has not directly measured a complete Neural K-profile on an LLM. What is offered here is a candidate explanation supported by controlled experiments, not a completed proof for all LLM emergence.

### 4.11 Architecture Is the Reference Language, and Dead Bits Are Naturally Included

The network structure, input encoding, and parameterization determine which programs are easy to express. Convolution, attention, local connectivity, or artificially added rule primitives amount to changing the reference language: they can make some class of tasks shorter, but cannot make all tasks simultaneously shorter at no cost. This document does not study how to analytically derive the profile from the architecture; it only measures the results under a given protocol.

Dead bits need no extra patch. With 0/1 encoding, an all-0 input dimension is multiplied to zero in the first affine layer, and the corresponding weights are free directions; an all-1 dimension can be merged into the bias, changing the effective parameterization and mass. If the training set never varies that bit, the data have not identified off-support invariance either. The framework naturally retains these extensions and their protocol-relative masses, without additionally stipulating that the optimizer must press the dead-bit weights to zero.

Weight decay likewise only modifies the objective and the path:

$$
J(\theta)=L_D(\theta)+\lambda\lVert\theta\rVert^2.
$$

It can change margins, entries, and certain memorization realizations, but the norm is not equal to Neural K. Grokking and post-fit drift without weight decay show that it is not a defining condition of this document's mechanism.

### 4.12 The Overall Causal Diagram

The final framework can be compressed into one measurement chain:

$$
(\mathcal A,\varphi,\ell,D)
\longrightarrow L_D(\theta),
\qquad
\mu\longrightarrow \text{region mass},
$$

$$
(L_D,\mu)
\longrightarrow
\{\Omega_D(f;\epsilon),V_f^{full}(\epsilon),K_f^N(\epsilon)\},
$$

$$
\text{initialization and optimizer}
\longrightarrow Q^{opt}_{D,t}
\longrightarrow
\text{prediction, agreement, generalization, and failure modes}.
$$

The data determine the constraint content and weights, the loss depth determines the resolution precision, the architecture determines the reference language, the reference measure determines the static mass, and the optimizer determines the reachable transport. Prior simplicity, loss-differential contraction, data transitions, grokking, overfitting, feature learning, and symbolic condensation thereby no longer need mutually independent stories; they are observations of the same function ensemble at different coordinates and scales.

## 5. Relation to Existing Theories

### 5.1 Parameter-to-Function Mapping, Simplicity Priors, and the Static Bayesian Approximation

[Dingle et al. (2018)](https://www.nature.com/articles/s41467-018-03101-6), [Valle-Pérez et al. (2019)](https://arxiv.org/abs/1805.08522), [Mingard et al. (2019)](https://arxiv.org/abs/1909.11522), [Mingard et al. (2021)](https://www.jmlr.org/papers/v22/20-676.html), and [Mingard et al. (2025)](https://www.nature.com/articles/s41467-024-54813-x) systematically demonstrated that random parameters, after being mapped through the network, produce a highly nonuniform function prior, and used hard conditioning to explain simple functions and generalization. This document accepts their most important starting point: the many-to-one mapping of parameter space itself already produces a simplicity preference. E19 also confirmed that the priors and shallow-loss masses of different complete targets are extremely nonuniform.

This document's corrections have three layers. First, training is not just one hard filtering; `Qstatic(D,epsilon)` undergoes function-specific rearrangement as the continuous raw loss varies. Second, a partial training set has no unique "true rule" designated by the researcher, and all compatible extensions must be retained. Third, the optimizer pushforward is related to, but not equal to, the static conditional mass.

E01--E07 and E17 directly rule out the universal strong versions of "first interpolation is the endpoint" and "SGD transparently samples the static posterior".

### 5.2 Grokking, Post-Interpolation Training, and Representation Reorganization

[The original Grokking work of Power et al. (2022)](https://arxiv.org/abs/2201.02177) took training error reaching zero first, with validation performance jumping later, as the core phenomenon; [Nanda et al. (2023)](https://arxiv.org/abs/2301.05217) further decomposed modular-addition grokking with Fourier circuits and continuous progress measures. This document does not replace these microscopic mechanisms, but provides a higher-level static coordinate: after interpolation the raw loss can still decrease, and different function cells and the internal realizations within the same cell continue to contract differentially.

E12--E16 show that shortcut reordering, marginal condensation, and global accessibility can be separated, so grokking has both static competition and an independent dynamical threshold.

### 5.3 MDL, Algorithmic Probability, PAC-Bayes, and Prequential Coding

[Solomonoff (1964)](https://mlanthology.org/misc/1964/solomonoff1964misc-formal/) used algorithmic probability to give short programs higher prior mass, [Grünwald (2004)](https://arxiv.org/abs/math/0406077) systematically organized MDL, [McAllester (1999)](https://doi.org/10.1145/307400.307435) used PAC-Bayes prior-to-posterior complexity to control generalization, and [Blier and Ollivier (2018)](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3b712de48137572f3849aabd5666a4e3-Abstract.html) measured the prequential code in deep networks.

Neural K shares the `-log mass` structure with these directions, but its object of study is more concrete: the measurable parameter mass of a fixed network protocol at a specified loss precision, and its predictions for complete functions, data transitions, and real-image branches.

E22 makes the per-sample bit-cost increments $\Delta J$ strictly accumulate to the endpoint cost difference, and E23 connects the full-target profile to the random-training-set identification threshold. A neural coding theorem is still missing, so Neural K cannot be written as an estimator of standard Kolmogorov complexity, nor can the ranking crossings observed in this document be automatically derived from PAC-Bayes bounds.

### 5.4 Flat Minima, SLT/RLCT, and Function Volume

[Hochreiter and Schmidhuber (1997)](https://doi.org/10.1162/neco.1997.9.1.1) connected flat minima with MDL/generalization, [Chaudhari et al. (2017)](https://arxiv.org/abs/1611.01838) used local entropy to construct Entropy-SGD, the SLT/RLCT of [Watanabe (2009)](https://doi.org/10.1017/CBO9780511800474) studies evidence and free-energy asymptotics in singular models, and [Lau et al. (2025)](https://proceedings.mlr.press/v258/lau25a.html) further proposed an estimable Local Learning Coefficient. This document shares with them the "mass rather than single point" perspective, but insists on function resolution and loss resolution: the parameter norm, Hessian flatness, or some local minimum does not automatically equal complete-function mass.

Reparameterization also changes naive Euclidean flatness, so this document always reports `mu` and SMC volumes relative to an explicit reference protocol.

The most direct future bridge is to compare the asymptotic relations among Neural K-profile local slopes, RLCT/local learning coefficients, and margin cores; this document does not claim their equivalence in advance.

### 5.5 Network Ensembles and Free Energy in Statistical Mechanics

[Levin, Tishby, and Solla (1989)](https://mlanthology.org/colt/1989/levin1989colt-statistical/) already wrote layered networks as a statistical ensemble, using training free-energy differences to describe prediction and generalization. This shows that the mathematical ancestor of "neural networks + partition function + predictive MDL" existed very early. Among modern extensions, [Pacelli et al. (2023)](https://www.nature.com/articles/s42256-023-00767-6) established a finite-width Bayesian deep-network statistical-mechanics framework beyond the infinite-width limit; [Jacot et al. (2018)](https://arxiv.org/abs/1806.07572) gave the NTK kernel limit of infinite-width training. They all rely on their own explicit assumptions and cannot directly replace this document's function-level empirical measurements.

This document's contribution is not in renaming the partition function, but in completing the empirical objects: modern nonlinear networks, complete hard functions, continuous loss, partial-training-set candidate cells, hard-margin internal cores, optimizer transport, and sample transitions are placed in one reproducible measurement chain. We have not analytically solved the density of states; "why some architecture assigns this profile to some function" is explicitly left to harder mathematical theory.

## 6. Boundaries of the Conclusions and Open Problems

This document measures the function-resolved density of states of different rules, but does not analytically derive its shape from the architecture. Why some network assigns a particular low-loss volume to parity, 3/8, or some shared program remains a deeper mathematical question. This document also has not proven that any pair of functions crosses only finitely many times in volume, has not established a cross-architecture coding theorem, and has certainly not written the real optimizer as a known sampler of the static ensemble.

Five negative boundaries must also be made explicit:

1. it is not claimed that any prespecified hard function's probability rises monotonically over all loss intervals;
2. it is not claimed that every added consistent sample raises the target mass at any fixed raw-BCE threshold;
3. it is not claimed that the parameter norm, flatness, LZ, spectral order, or a single shallow slope is Neural K;
4. it is not claimed that high agreement guarantees external correctness, or that all high-agreement mappings have been proven human-readable;
5. it is not claimed that static volume eliminates the independent roles of the optimizer, initialization, and global accessibility.

The early evidence concentrated on finite Boolean rules, cellular automata, modular arithmetic, tiny MLPs, and two MNIST binary-classification tasks. The October 2026 work further extends to multiple MNIST digit partitions, data seeds, sample sizes, and training methods, as summarized later on this page. The more than one hundred synthetic tasks provide breadth, and the SMC and closure experiments provide depth, but there is still a clear scope gap from "a universal law for arbitrary modern architectures and real tasks". All quantities should be reported together with the protocol.

Next-step falsifiable predictions include:

1. blindly predicting the static/SGD U-shape lowest interval on new MNIST digit pairs;
2. comparing the rank correlation between Neural K-profiles and n50/n90 across natural architectures;
3. under the same fixed training set, testing whether the profile can quantitatively predict complete-function odds;
4. increasing SMC particles, replicas, and independent implementations to verify the deep-tail orders of magnitude;
5. testing whether high-agreement mappings still systematically fall into short, extractable program families;
6. testing whether the profile deep-tail rankings of new function families stabilize, and whether the number of crossings is finite;
7. using preregistered profiles rather than post-hoc metrics to predict new grokking transition intervals.

## 7. Conclusion

This research started from a counterexample problem against the strong static conditional prior, and through function-ensemble drift, continuous loss sampling, shortcut reordering, parity accessibility, the optimizer gap, agreement intervention, and measure closure, finally converged to a framework weaker than the original claim and more measurable.

The explicit pressure of neural-network training is only to lower the training objective. The architecture and reference measure first assign nonuniform mass to functions; the training set defines the competing constraints; the loss depth differentially contracts functions and their internal realizations; the optimizer determines how the system actually enters and passes through these regions. The simplicity preference is not a one-shot hard filtering, nor an oracle that is monotone everywhere for every function, but a loss-resolved mass flow that can bend and cross.

This document defines this flow as the Neural K-profile, and constrains its meaning with three kinds of independent evidence: full-target volume prospectively predicts the parity data transitions of random training sets; shared-parent SMC confirms that the deep-tail complexity ordering really reverses; and MNIST static branches produce high accuracy on real images and reconstruct the validation NLL U-shape. Per-sample $\Delta J$ further connects endpoint information cost and the statistical-mechanics partition function exactly, and connects predictive surprise and code length when label-normalization and coding conditions hold.

Grokking, direct learning, overfitting, noise memorization, feature reuse, symbolic condensation, and partial capability emergence therefore need not be regarded as an unrelated collection of mechanisms. They can be understood as different regions of the same training-set-centered function ensemble along the axes of data size, loss precision, network language, and accessibility.

This framework has not yet resolved "why a concrete architecture produces a concrete profile", nor has it completed a universal theory; but it has already rewritten a problem long entangled in "why generalization happens" into a set of measurable, bridgeable, and prospectively decidable objects.

## October 2026 Website Supplement

[E38](experiments/e38.html) completes a width/depth SMC scan with five configurations, 19 targets, and nine thresholds. At BCE≤0.30, the estimated mass disadvantage of all 17 selected non-single-bit targets relative to the same single-bit reference grows across four two-hidden-layer widths. This is evidence about continuous-loss events under the specified uniform measure and scaling, not hard-cell mass, measure-independent complexity, or a causal proof of double descent. The experiment page retains the width-dependent budgets, shared pool adaptation, and mixing boundaries.

The website also provides [Double descent and width evidence](width-double-descent.html) and [Complexity measurement methods](complexity-methods.html). The latter proposes measurements of rules and functions implemented by weights; no new method-validation experiment was performed in this round.

## Four areas of evidence and updated limits on 2026-10-27 {#evidence-update-20261027}

The additions are integrated into E36, E31/E33, E37, and E32 while retaining the original reports; research topics do not receive new experiment numbers merely because they have expanded. This section summarizes completed results without deleting unreached conditions, sampling uncertainty, or unresolved explanations.

### Static prediction and different training methods

[The E36 extension](experiments/e36.html#update-20261027) completes 34 original and 24 added conditions, totaling 58. The small training sets use five-versus-five partitions of the ten MNIST digits, multiple sample seeds, and label interventions. They are relatively arbitrary small-sample constraints, not certified pure random labels, and the training set need not uniquely identify the full rule.

| Condition group | Conditions | Hard-fit accuracy MAE for predicting SGD / percentage points | Low-loss SMC MAE / percentage points |
|---|---:|---:|---:|
| CNN W4, clean n32 | 9 | 8.18 | 0.74 |
| MLP, clean n32 | 9 | 6.05 | 1.16 |
| CNN W8, clean n32 | 9 | 8.43 | 5.02 |
| CNN W4, added n64 | 6 | 8.13 | 0.58 |
| MLP, added n64 | 6 | 6.63 | 0.26 |

The low-loss node is BCE≤0.003. The comparison concerns mean single-model accuracy, not ensemble accuracy or equality of full function distributions. Direct G&C reaches its sample quota in 13 conditions; across 12 n8 conditions, its accuracy MAE against hard-fit SMC is about 0.120 percentage points, which cannot be extrapolated into certification of deep-tail mixing.

STP completes 16 conditions, three step sizes, and 128 initializations each, totaling 48 groups. It is a mechanistic control testing whether the distinctive gradient updates of SGD/Adam are indispensable; correspondence with low-loss static predictions is more informative than merely demonstrating fit. At the smallest step size, STP–SGD accuracy MAE across six clean conditions is 0.793/0.192 percentage points for the smaller CNN/MLP, but smaller steps do not monotonically improve CNN agreement.

For the two original wider-CNN conditions, enhanced SMC reaches 1e-5 with accuracies 75.59%/75.16%, versus 79.68%/77.25% for SGD at lr=.001 and about 69.65%–70.85% for STP. The optimizers themselves disagree, so there is no unique answer shared by all training methods from which SMC alone departs; these concrete discrepancies nevertheless constrain exact correspondence and must not be set to zero or hidden.

### Independent complexity criteria and amplitude intervention

[The added f/g experiment in E31](experiments/e31.html#update-20261027) starts from an independent logical comparison: f is two-bit XOR, while g ORs its result with a third bit. In a fixed gate library containing XOR and OR, with at most two inputs per gate, their minimum gate counts are strictly 1 and 2; this comparison does not depend on Neural K. After observing that g was cheaper under raw BCE, the user prospectively proposed an easy-branch precision-allocation explanation and predicted that amplitude alignment could reverse the ordering.

| Score threshold 0.001 | K_g−K_f / bit | Estimation basis |
|---|---:|---|
| Original mean BCE | −37.79 | First pass |
| Pointwise BCE bound | +29.47 | Increased budget |
| BCE plus margin-variance constraint | +50.28 | Increased budget |
| Bridge from the original BCE parent event to amplitude alignment | +49.77 | Inherits parent-pool error |

The two branch-mean BCE values for g change from 0.0018534/0.0001242 to 0.0009212/0.0009130 under alignment. The result supports the prospective intervention direction while showing that mean BCE may be unsuitable as a universal logical-complexity ruler: a valid original volume measurement does not imply that precision-allocation effects have been removed, nor does it prove alignment the uniquely correct definition.

[The added external calibration in E33](experiments/e33.html#update-20261027) includes proxy recalculation and 60 logical-cost problems, with 59 exact results and 1 bounded result. Of 45 pairs among ten balanced orbits, 43 share their direction across four architectures, yet robust local reversals remain against specified external logical costs. Shared ordering is independent evidence, not a theorem of equivalence to classical Kolmogorov complexity.

### Measures and fixed radius

[The E37 extension](experiments/e37.html#update-20261027) presents earlier multi-measure evidence alongside the new fixed-sphere controls, explicitly dating the older data. At R=40, the ΔK values from BCE 0.03 to 0.001 are 14.66, 27.81, and 69.04 bit for projection, majority3, and parity2, respectively; rule-dependent angular contraction persists at fixed radius.

Eight pools for the more complex rules at R=20 do not reach every node; projection has ΔK=70.61 bit at that radius, so the change is not only an intercept shift. The original operator failure across 96 pools is an implementation issue; of the 24 pools after repair, 16 reach all nodes and 8 stall. The conclusion is that the main phenomenon is not exclusive to Gaussian radial tails, not that all natural measures share the same formula or local ordering.

### Memory-coefficient calibration and the decision to defer

[The E32 extension](experiments/e32.html#update-20261027) asks whether proportionality between independent memory burden and the coefficient can provide strong external calibration, rather than being guaranteed by definition. It completes 384 main pools across one to three hidden layers, four tables, eight sample sizes, and four pools per table; additions include 24 enhanced three-layer pools, 20 five-rule pools, and margin-range bridges.

| n | c for one layer, 32 | c for two layers, 32/32 | c for three layers, 32/32/32 |
|---:|---:|---:|---:|
| 16 | 153.93 | 57.45 | 35.98 |
| 32 | 274.87 | 82.62 | 46.78 |
| 64 | 508.54 | 121.45 | 62.58 |
| 96 | 668.14 | 148.72 | 74.07 |
| 128 | 815.48 | 169.82 | 83.07 |

c uses the within-pool difference K(.001)−K(.002), taking the median across four pools before averaging four tables. Growth persists, but the two- and three-layer curves have systematic curvature that enhanced sampling does not remove. Strict amplitude tubes have insufficient reproducibility at larger n and cannot be said to restore linearity.

Additional diagnostics comprise 3840 agreement-training trajectories, 720 per-example-loss trajectories, structural checks of 132 random tables, and min–max enumeration using existing full-function catalogues. The two candidate memory criteria are maximum cost at fixed n and the weakest unseen-input prediction tendency; domination by the cheapest compatible candidate is a mechanistic hypothesis connecting them, not a third established definition. Full-rule catalogue min–max is not a direct measurement of actual fixed-D mass.

The three current difficulties are certifying individual memory datasets, the cost of reliable SMC, and the lack of a necessity theorem for linear neural implementation costs. Additional exploration is deferred while nonlinearity, unreached conditions, and sampling disagreements remain reported; strict linearity is not treated as a necessary foundation for the entire framework. See the [current boundaries](theory-core.html#open-questions-20261027) for the six theoretical open questions.

## Planned Appendices

- A. Networks, data, optimizers, and all hyperparameters;
- B. SMC, pCN, lineage, and volume-error diagnostics;
- C. Set identities for full-rule/fixed-D/hard-margin;
- D. Additional function-distribution, path, and optimizer comparisons;
- E. The early matrix of more than one hundred deterministic-task experiments;
- F. Failed experiments, noise, and accessibility boundaries.
