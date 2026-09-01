# The Core Neural K Framework

> This page contains only the final theoretical framework. Experiment IDs are evidence anchors, not a reconstruction of the research timeline.

## One-sentence version

Once architecture, encoding, parameter measure, and loss are fixed, a neural network already represents some simple functions with more parameter mass than others. Tightening training loss then removes the mass of different functions and within-function implementations at different rates, typically favoring lower neural-relative complexity and greater reuse. On structured data, lower training loss therefore often acts like pressure for a more economical program describing the training set. Data determines which programs remain compatible, architecture determines what counts as economical, and optimization determines whether those low-loss programs can actually be reached.

## 1. Two forms of simplicity at two scales

### 1.1 Prior simplicity

Random parameter sampling induces a highly nonuniform function distribution. A function that can be implemented by more parameter configurations receives more mass. Prior work and the project’s own sampling experiments show that structurally simple and architecture-aligned functions often receive greater prior mass.

This does not mean every function simple to humans is easy for every network. Parity is the standard counterexample: it is short in an XOR language but can be expensive in the language of an ordinary tanh MLP.

### 1.2 Loss-resolved simplicity

The cleanest definition and direct evidence for loss-resolved simplicity use **all input-output examples of each candidate function** and measure a separate full-target loss-volume profile for that function. In a finite Boolean space this means the complete truth table. In a non-enumerable space, one must preregister a sufficiently comprehensive target set applied consistently to every candidate. A small partial training set does not automatically grant privileged status to the generator known by the researcher.

Under a complete target, parameters can have every correct hard output and still occupy very different loss depths.

> **Core claim: as raw loss is tightened, available parameter volumes of different functions do not shrink proportionally. Implementations requiring many nonreusable exceptions, or poorly aligned with the architecture, often contract faster in the deep tail; more economical implementations gain relative mass.**

With only a partial training set $D$, the measured object is instead the mass of each complete extension inside the low-loss parameters compatible with $D$. This fixed-$D$ candidate mass is rigorously related to, but is not identical with, full-target volume. [E20](experiments/e20.html) connects them through conditional probability and a margin bridge. Under partial data, the preferred extension also depends on coverage, imbalance, and the competing compatible functions; full-target rankings cannot be transferred without that bridge.

This empirical rule cannot be written as “one prespecified simple function rises monotonically at every loss.” [E12](experiments/e12.html), [E14](experiments/e14.html), and [E24](experiments/e24.html) show that winners and local contraction rates can change with depth. The correct object is a full complexity profile, not a permanent scalar.

### 1.3 Relation to “compression is intelligence”

Ordinary training explicitly optimizes training loss, not a second machine-independent program-complexity term. Compression emerges from architecture, parameter measure, and loss geometry. If feature, rule, or intermediate-computation reuse allows one mechanism to satisfy many examples with fewer independent degrees of freedom, it is often the cheaper route to deeper loss.

On clean, structured, data-sufficient tasks, reducing loss can therefore be interpreted as searching for a more economical description of the training set. This directly connects to “compression is intelligence”: intelligent behavior comes from reusable regularities rather than independent answers for every input. [E08](experiments/e08.html) provides a causal shared-computation test, while the original one-hundred-plus rule experiments provide breadth.

The relation is not unconditional. Absolute program complexity may rise during early learning, and deeper loss on random or corrupted labels can increase exception memory.

> **Core boundary: loss is the explicit objective; compression is a common economical path under structured data and a finite neural language.**

## 2. Complexity is an architecture-relative profile

Bundle architecture and parameterization, input-output encoding, parameter reference measure, and per-example loss into one neural protocol:

$$
\Pi=(\mathcal A,\varphi,\mu,\ell).
$$

For a complete target function $f$, define its parameter volume below loss $\epsilon$:

$$
V^{\mathrm{full}}_\Pi(f;\epsilon)
=
\mu_\Pi\{\theta:L_{D_{\mathrm{full}}(f)}(\theta)\le\epsilon\}.
$$

The Neural K-profile is

$$
K^N_\Pi(f;\epsilon)
=
-\log_2V^{\mathrm{full}}_\Pi(f;\epsilon).
$$

It is not machine-independent Kolmogorov complexity. It measures how rare it is for the current neural language to implement $f$ to precision $\epsilon$. Changing architecture, activation, width, encoding, initialization scale, or loss can change the curve.

Complexity is primarily a profile. Prior probability is only a shallow starting point, volume at one loss is one slice, and local contraction rate is one slope. Curves can cross. Hard function identity is still too coarse: margins, logits, and internal representations can continue contracting and changing order after all hard signs are fixed. [E24](experiments/e24.html) is the direct demonstration.

### Independent operational quantity: identification and recovery transitions

The Neural K-profile is the static geometric object for a complete target. Real training also permits a directly measurable sample complexity, but it must **not be identified with the Neural K-profile**. Fix the neural reference protocol $\Pi$, a training-set sampling distribution $q$, and a complete audit protocol $\mathcal T$: capacity, initialization, optimizer, training budget, seed count, complete probe range, and recovery thresholds must all be preregistered.

For a complete target $f$ and sample count $n$, draw many size-$n$ training sets $D_j$ uniformly from $D_{\mathrm{full}}(f)$ and train an independent seed committee on each. A dataset-level recovery predicate $\mathcal R_{\Pi,\mathcal T}(D_j,f)$ should not use validation accuracy alone. It should jointly require:

1. most seeds fit the training set;
2. the complete target function has sufficient mass among fitted seeds;
3. the target is the modal hard function;
4. complete-function collision / agreement exceeds a preregistered threshold.

The recovery rate at sample count $n$ is

$$
\widehat r_{\Pi,\mathcal T}(f,n)
=
\frac{1}{M}
\sum_{j=1}^{M}
\mathbf 1\!\left[
\mathcal R_{\Pi,\mathcal T}(D_j,f)=1
\right].
$$

After monotone estimation of this curve, define

$$
n_q^{\mathrm{id}}(f;\Pi,q,\mathcal T)
=
\min\left\{
n\in\mathcal G,\ n<N:
\widehat r_{\Pi,\mathcal T}^{\,\uparrow}(f,n)\ge q
\right\},
\qquad q\in\{0.5,0.9\}.
$$

Here $N=|D_{\mathrm{full}}(f)|$ and $\mathcal G$ is a preregistered grid of non-full sample counts. `n50` and `n90` are the first grid crossings at which 50% and 90% of random datasets stably recover the complete rule. A point estimate `n_q=n_k` only localizes the continuous transition to $(n_{k-1},n_k]$ and should be reported with bootstrap intervals. If no crossing occurs by $N-1$, report `n_q>N-1` or right-censored; fitting the full truth table at $n=N$ is a reachability check, not a generalization transition.

These quantities therefore bracket a data-axis grokking / rule-recovery transition rather than locating an infinitely precise point. A tie, one-grid reversal, or overlapping confidence intervals for nearby rules is not evidence against a statistical relationship. This limitation is severe in a 4-bit space with only sixteen states: each example occupies 6.25% of the domain, holdout position and symmetry dominate, and hard rules can hit the full-space ceiling before a transition is identifiable. In that regime `n50/n90` may be unresolvable and only full profiles plus local leave-one-out evidence remain available.

The more accurate name for this quantity is **protocol-relative identification/recovery sample complexity**. Its unit is examples, and it asks how many random constraints the current training system needs before it stably selects the complete rule under $q$. It contains at least four ingredients: the complete target's Neural K-profile, the denominator of competing extensions under a fixed partial dataset, identifiability induced by sample coverage and shortcuts, and optimizer reachability. It can therefore serve as an external empirical anchor for Neural K, but not as an equivalent definition without additional assumptions.

Put differently, complete-target volume measures a numerator. A random-dataset transition measures when that numerator dominates a target-dependent denominator under optimizer transport. Only when these extra factors are controlled across tasks, or proved to preserve order, can `n50/n90` rank be used as a proxy for full-target Neural K rank.

### How these experiments corrected the original claim

Our initial hypothesis was natural: **rules that the network finds more complex should require more training examples before they are stably identified.** A preregistered parity1-to-parity4 experiment followed this order exactly. Complete-rule Neural K therefore has genuine prospective predictive power; it is not merely a label assigned after seeing the result.

Parity2 and MUX3 then produced a counterexample that could not be ignored. Complete-rule volume made MUX3 look easier, yet ordinary random training required more data to recover it. Target difficulty alone was therefore insufficient.

The missing quantity was **which alternative explanations remain compatible with the current training set**. For MUX3, many examples agree both with the true selector rule and with a shortcut that copies one bit and memorizes a few exceptions. Such examples add data without actually distinguishing the competing functions.

The first follow-up changed only the sampling distribution. Once training emphasized inputs that separate true MUX3 from the copy shortcuts, MUX3 recovered earlier than parity2. Neither the target rule nor the network changed; only whether the examples struck the places where the competing functions disagree.

The second follow-up removed the optimizer entirely and directly sampled the low-loss functions allowed by the same datasets. The result remained: ordinary samples left a large shortcut population, while conflict examples eliminated it and made true MUX3 dominant. The effect therefore begins in the static function competition created by the dataset; the optimizer mainly changes how quickly and by which path that distribution is reached.

> **Corrected claim: the amount of data needed for a rule depends not only on the rule's own difficulty, but also on which functions compete with it and whether the sampled examples efficiently eliminate those competitors. Complete-rule Neural K remains an important first-order predictor, but it is not identical to the grokking transition.**

Detailed numbers, all three follow-ups, and numerical boundaries are on the [E23 page](experiments/e23.html). The core page keeps only the corrected causal relation.

[E24](experiments/e24.html) adds a different lesson: even after two complete mappings are fixed, their relative volume can change as loss deepens. Rule difficulty therefore belongs to a full loss profile rather than one shallow slice.

### Relation to MDL and algorithmic information theory

Standard MDL writes learning as two code lengths:

$$
L_{\mathrm{MDL}}(H,D)=L(H)+L(D\mid H).
$$

$L(H)$ is the cost of describing a hypothesis or program, and $L(D\mid H)$ is the cost of encoding the residual data not explained by it. Algorithmic probability and Kolmogorov complexity express the same simplicity principle from another direction: shorter programs receive greater prior mass.

Neural K turns this language into a measurable but protocol-dependent object. The reference measure $\mu_\Pi$ already assigns nonuniform mass to implementations. The full-target volume $V^{\mathrm{full}}_\Pi(f;\epsilon)$ of a complete function at precision $\epsilon$ induces the codelength $-\log_2V^{\mathrm{full}}_\Pi(f;\epsilon)$. The Neural K-profile can therefore be read as **the precision-dependent codelength paid by the current neural language to describe a target at different residual tolerances.**

This does not mean that training explicitly minimizes “loss plus program length.” Ordinary code usually minimizes empirical loss alone; MDL-like selection emerges from the interaction between the architectural reference measure and loss-sublevel geometry. Nor is Neural K machine-independent Kolmogorov complexity: changing the network, encoding, parameterization, or reference measure changes the codelength. No neural coding theorem currently makes the two equivalent.

The falsifiable content goes beyond renaming volume. [E22](experiments/e22.html) makes per-example generalized surprise telescope exactly into an order-invariant endpoint codelength, while [E23](experiments/e23.html) uses preregistered full-target profiles to predict n50/n90 on independent random datasets. Together they connect parameter mass, predictive coding, and identification sample complexity empirically.

## 3. Static geometry and optimization dynamics must remain separate

For a fixed dataset $D$ and loss threshold $\epsilon$, the static mass of a complete function is

$$
\Omega_{\Pi,D}(f;\epsilon)
=
\mu_\Pi\{\theta:L_D(\theta)\le\epsilon,\ h_\theta=f\}.
$$

The normalized static function distribution is

$$
Q^{\mathrm{static}}_{\Pi,D,\epsilon}(f)
=
\frac{\Omega_{\Pi,D}(f;\epsilon)}
{\sum_g\Omega_{\Pi,D}(g;\epsilon)}.
$$

An optimizer pushes the initialization distribution through a training map and induces another distribution:

$$
Q^{\mathrm{opt}}_{\Pi,D,t}
=
(T_{\Pi,D,t})_\#\mu_\Pi.
$$

> **Key separation: static volume determines which candidate regions are large and which remain robust under deeper loss. It is an important first-order influence. Optimization additionally determines entrances, flow channels, speed, connectivity, numerical noise, and history. Static mass must not simply be called the SGD posterior.**

The parity scaffold shows that a low-loss endpoint can exist and be locally stable while remaining unreachable from random initialization. [E17](experiments/e17.html) shows that SMC and AdamW can select very different functions under the same architecture, dataset, and comparable loss.

[E28](experiments/e28.html) supplies the complementary real-task anchor. On the same small CNN and all 50k MNIST training images, whole-network HMC and plain Adam reached 99.03% and 98.95%, respectively, while retaining different diversity and time-averaging behavior. [E29](experiments/e29.html) gives a stricter matched-loss counterfactual: across five dead-bit functions, no-decay Adam and static SMC differed by only 0.0010 on average, whereas L2/MAP created a distinct mechanism by shrinking the unidentified weights. Together these results show that static geometry can be a strong predictive base map without making static and optimizer distributions identical.

[E30](experiments/e30.html) adds only a supplementary clarification: grokking does not require
weight decay, because the unregularized raw-loss landscape can already favor rule recovery. Weight
decay strengthens that preference by reshaping the same static landscape; it is neither mysterious nor
a special theoretical mechanism.

Every task therefore requires three separate checks: can the network represent the target, does the data identify it, and can the optimizer reach it?

### Decoupling, scaffolding, and optimization accessibility are not complexity

Let the original endpoint objective be $L_{\mathrm{end}}(\theta)$. Decoupled training, intermediate supervision, or scaffolding typically adds an auxiliary objective:

$$
L_{\lambda}(\theta)
=
L_{\mathrm{end}}(\theta)
+
\lambda L_{\mathrm{aux}}(\theta),
$$

with $\lambda$ annealed to zero later or with auxiliary inputs and intermediate labels removed entirely. The first effect is to change the gradient vector field and the path into the low-loss region:

$$
-\nabla L_{\lambda}
=
-\nabla L_{\mathrm{end}}
-
\lambda\nabla L_{\mathrm{aux}}.
$$

Near random initialization, the original endpoint loss may provide almost no useful direction, or gradients associated with different computational stages may cancel. Auxiliary objectives factor a distant, compositional, or symmetric computation into locally learnable steps. The network can first form intermediate representations and later internalize them into endpoint computation. This improves search without implying that the original target function has lower Neural K.

If the scaffold is removed and the network still remains correct under $L_{\mathrm{end}}$ alone, or recovers after perturbation, then the original endpoint task has a stable low-loss region supported by its own loss. The initial failure was primarily an entrance or gradient-accessibility problem. The parity scaffold–removal–perturbation experiment in [E16](experiments/e16.html) is such a test. Earlier Mod 3, multiplication, recursive/search, and Towers-of-Hanoi experiments show the same qualitative separation.

Conversely, success with auxiliary supervision does not by itself prove that the endpoint function is simple, and endpoint-only failure does not prove that it is complex. Four objects must be separated:

1. **Representational capacity:** do parameters implementing the target exist?
2. **Data identifiability:** do the examples eliminate competing extensions?
3. **Protocol-relative complexity:** what is the target’s Neural K-profile under the original neural protocol?
4. **Optimization accessibility:** can the specified initialization and optimizer enter that region?

If auxiliary variables, modules, or inputs remain permanently in the final system, the neural protocol $\Pi$ has changed and the target may genuinely be simpler in the new language. If auxiliary information is used only during training and fully removed, it primarily supplies a continuation or curriculum path; final complexity should still be measured under the original endpoint protocol. Any comparison between decoupled and endpoint-only training must state which case applies.

## 4. Statistical-physics language

### 4.1 Dictionary

| Statistical physics | Neural-network object |
|---|---|
| Microstate | One concrete parameter setting $\theta$ |
| Macrostate | A complete hard function, or a finer function-margin/representation cell |
| Energy | Total training loss |
| Density of states | Number or measure of parameter microstates near one energy |
| Entropy | Log density or log cumulative volume |
| Temperature | Tolerance for training error; not physical temperature |
| Free energy | Combined cost of energy requirements and parameter entropy |
| Phase change | A change in the dominant function macrostate as data, temperature, or loss depth varies |

Use the additive total energy

$$
E_D(\theta)=\sum_{i=1}^{n}\ell(h_\theta(x_i),y_i)=nL_D(\theta).
$$

For function $f$, define the energy density of states

$$
\rho_{D,f}(E)
=
\int \delta(E-E_D(\theta))\,\mathbf 1[h_\theta=f],d\mu_\Pi(\theta).
$$

The microcanonical view takes the cumulative volume below a loss threshold:

$$
\Omega_{D,f}(\epsilon)
=
\int_{-\infty}^{n\epsilon}\rho_{D,f}(E),dE.
$$

Its logarithm acts as a cumulative parameter entropy. A function with more implementations at the same loss has greater parameter entropy and greater static mass.

### 4.2 Canonical ensemble, partition function, and free energy

For inverse temperature $\beta$,

$$
Z_D(\beta)=\int e^{-\beta E_D(\theta)}d\mu_\Pi(\theta),
$$

with function-restricted partition function

$$
Z_{D,f}(\beta)
=
\int \mathbf 1[h_\theta=f]e^{-\beta E_D(\theta)}d\mu_\Pi(\theta).
$$

The Gibbs mass of a function macrostate is

$$
Q_\beta(f\mid D)=\frac{Z_{D,f}(\beta)}{Z_D(\beta)}.
$$

Define total and restricted free energies by

$$
F_D(\beta)=-\frac1\beta\log Z_D(\beta),
\qquad
F_{D,f}(\beta)=-\frac1\beta\log Z_{D,f}(\beta).
$$

Function odds are controlled by restricted free-energy differences:

$$
\log\frac{Q_\beta(f\mid D)}{Q_\beta(g\mid D)}
=
-\beta\bigl(F_{D,f}-F_{D,g}\bigr).
$$

Larger $\beta$ emphasizes lower energy, hence lower training loss. If a complex function has fewer low-energy microstates, its restricted free energy deteriorates faster and its mass falls. This is the statistical-physics expression of faster deep-loss contraction for complex functions.

Microcanonical loss slices and canonical Gibbs weighting summarize the same density of states; the canonical quantity is a Laplace-type aggregation of the former. Both have been measured in this project.

### 4.3 What this is not

Training is not automatically an equilibrium thermal system, and $\beta$ is not GPU temperature. Without additional Langevin or equilibrium assumptions, SGD and AdamW are not Gibbs samplers. Statistical physics is used first as a mathematical language for state density, energy–entropy competition, and changes in dominant macrostates.

Grokking can appear as rule-macrostate takeover at deeper loss or larger dataset size, but a finite-network crossover need not be a strict thermodynamic singularity. The project has not derived $\rho_{D,f}(E)$ analytically from architecture; that is the deeper “why simplicity?” question and is currently out of scope.

## 5. Predicting an unseen label from volume

Given training set $D$, unseen input $x$, and candidate label $y$, define

$$
D_y=D\cup\{(x,y)\}.
$$

The microcanonical label-branch mass is

$$
P_{\mathrm{hard}}(y\mid x,D,\epsilon)
=
\frac{\mu_\Pi\{\theta\in A_D(\epsilon):h_\theta(x)=y\}}
{\mu_\Pi(A_D(\epsilon))}.
$$

It asks: among all parameters already fitting the training set to the current loss, how many predict label $y$ on the new input? The larger branch is the current static prediction.

One can instead compare partition functions of label-augmented datasets:

$$
\widetilde P_\beta(y\mid x,D)
=
\frac{Z_{D_y}(\beta)}{Z_D(\beta)},
$$

and normalize across labels:

$$
P_\beta(y\mid x,D)
=
\frac{Z_{D_y}(\beta)}{\sum_{y'}Z_{D_{y'}}(\beta)}.
$$

With standard negative log likelihood and $\beta=1$, this is the ordinary posterior-predictive evidence. More generally it is a protocol-relative Gibbs branch score.

> **Prediction principle: if the volume associated with one label contracts more slowly as loss deepens, that label retains more compatible implementations under stricter training precision and gains predictive support.**

Label profiles can still cross, so prediction must state the chosen loss depth or $\beta$. MNIST [E25](experiments/e25.html) directly tests this branch prediction.

[E26](experiments/e26.html) extends the same principle from one unseen label to a complete labeling. Given one anchor and a 5:5 balance constraint, the natural MNIST 0/1 split moved from last place at shallow loss to top one in both blinded panels and held about 99.97% normalized mass at BCE 0.3. This supports volume-based prediction of unseen label combinations, not unconstrained clustering after removing class-balance information.

## 6. Surprise, information gain, and order invariance

### 6.1 Surprise of an observed label

For an observed label $y$,

$$
s(y\mid x,D)=-\log_2P(y\mid x,D).
$$

Define dataset-state cost

$$
C(D)=-\log_2Z(D).
$$

The generalized surprise of adding one example is

$$
\Delta C_t=C(D_t)-C(D_{t-1})
=
-\log_2\frac{Z(D_t)}{Z(D_{t-1})}.
$$

Summing over any sample order gives

$$
\sum_{t=1}^{m}\Delta C_t=C(D_m)-C(D_0).
$$

Intermediate terms telescope, so total cost depends only on the starting and final datasets. **Each increment remains order-dependent; only the total is invariant.** [E22](experiments/e22.html) numerically closes this identity for all 256 rules and all 40,320 orders per rule.

The invariant belongs to one static partition function. It is not a conserved quantity along an SGD trajectory.

### 6.2 Under hard conditioning, realized surprise exactly equals function-distribution information gain

The data point itself need not be sampled from a random generator. Let the current complete function be the random variable

$$
F\sim Q_D(f),
$$

and let the label at candidate input $x$ be the projection

$$
Y_x=F(x),
\qquad
P_D(y\mid x)
=
\sum_fQ_D(f)\mathbf 1[f(x)=y].
$$

After observing $y$ and applying pure hard conditioning,

$$
Q_{D'}(f)
=
\frac{Q_D(f)\mathbf 1[f(x)=y]}
{P_D(y\mid x)},
\qquad
D'=D\cup\{(x,y)\}.
$$

With base-two KL, this gives the exact identity

$$
D_{\mathrm{KL},2}\!\left(Q_{D'}\Vert Q_D\right)
=
-\log_2P_D(y\mid x)
=
s(y\mid x,D).
$$

> **Hard-conditioning identity: the less probable the true label is under the current function distribution, the more candidate-function mass it deletes. Realized predictive surprise is exactly the information gain in the function posterior.**

This is [Itti and Baldi's Bayesian surprise](https://pmc.ncbi.nlm.nih.gov/articles/PMC2782645/) instantiated over complete functions. What changes is not a presumed sample-generating distribution but the belief distribution over which complete function remains possible. The identity requires coherent hard updating of one static $Q_D$; retraining a multi-seed optimizer ensemble does not automatically obey this Bayes formula.

### 6.3 Realized surprise is not expected information gain before the label is known

Surprise is the realized coding cost after a label is observed. Before querying an unlabeled input, the relevant object is expected information gain: the average shrinkage of the function or parameter distribution after observing the label.

$$
\mathrm{IG}(x\mid D)
=
\mathbb E_{y\sim P(y\mid x,D)}
\left[
\mathrm{KL}\bigl(q(\theta\mid D,x,y)\,\|\,q(\theta\mid D)\bigr)
\right]
=
I(\Theta;Y_x\mid D).
$$

The equivalent [BALD](https://arxiv.org/abs/1112.5745) form is

$$
I(\Theta;Y_x\mid D)
=
H[Y_x\mid D]
-
\mathbb E_{\theta\sim q(\theta\mid D)}H[Y_x\mid\theta].
$$

If each hard function deterministically labels $x$, the second term vanishes and expected information gain equals predictive label entropy. Inputs closest to 50:50, and therefore with lowest agreement, are maximally informative on average. This motivates the active-disagreement experiment [E21](experiments/e21.html).

For binary labels, if $p=P_D(Y_x=1)$, then

$$
A(x\mid D)=p^2+(1-p)^2.
$$

Before the label is known, low agreement therefore corresponds monotonically to high predictive entropy and high **expected** information gain. Once the label is observed, however, the relevant quantity is $-\log_2P_D(y\mid x)$. A roughly 50:50 query is informative on average; a high-agreement label that contradicts the consensus can produce even larger **realized** surprise. These intuitions answer different questions: expected learning before a query versus the amount of function mass eliminated by an observed outcome.

A practical caveat remains: zero hard-posterior information does not guarantee that retraining with an agreed-upon sample has no effect. Soft margins can differ, and the sample changes the optimizer path from initialization.

For a soft Gibbs update,

$$
q'_D(\theta)
\propto
q_D(\theta)e^{-\beta\ell(\theta;x,y)},
$$

the generalized surprise is

$$
\Delta C
=
-\log\mathbb E_{q_D}
\left[e^{-\beta\ell(\theta;x,y)}\right],
$$

whereas posterior KL is

$$
D_{\mathrm{KL}}(q'_D\Vert q_D)
=
\Delta C
-
\beta\,\mathbb E_{q'_D}\ell.
$$

Thus sample codelength, posterior KL, and one-network BCE are related but not identical under soft loss. The exact identity of the previous subsection is recovered in the hard-indicator limit.

### 6.4 After adding a sample: local concentration does not imply monotone global agreement

After hard conditioning, all surviving functions assign $y$ at the queried point $x$, so local agreement at that point immediately becomes one. Agreement at another probe $x'$ may rise or fall: the removed function cluster might have maintained a global consensus or might have caused the disagreement. Global mean agreement, complete-function collision, and target-function mass therefore have no per-example monotonic theorem.

This explains concentration, branching, and reconcentration. A highly surprising label can first destroy the dominant function cluster, leave several smaller clusters more evenly weighted, and reduce global agreement. Later informative examples remove more competitors and reconcentrate the distribution. Local KL information gain remains nonnegative; global agreement need not increase.

The MUX3 cell/conflict experiment makes this distinction concrete. On ordinary cells $x_1=x_2$, true MUX3, `copy x1`, and `copy x2` give the same label, so such examples barely distinguish them. Selector-conflict cells $x_1\ne x_2$ separate the shortcuts from the true rule. Under uniform fixed-$D$ SMC, posterior target accuracy was 0.993 on ordinary cells but only 0.777 on conflict cells. Raising conflict examples to 75% increased conflict-cell accuracy to 0.995 and complete MUX3 target mass from 0.000214 to 0.782. The intervention increases not an abstract sample count but the amount of current competitor mass removed per example.

## 7. The agreement conjecture

For one input $x$, pairwise agreement between two independent function draws is

$$
A(x\mid D)=\sum_yP(y\mid x,D)^2.
$$

For the complete function distribution,

$$
C(D)=\sum_fQ(f\mid D)^2.
$$

High mean pointwise agreement does not imply high complete-function collision; many functions can differ only on a few locations. Applications should report representative pointwise agreement together with complete-function modal mass, collision, or Hamming-ball mass.

> **Agreement is valuable because it does not require knowledge of an external generator. It measures whether the empirical function distribution induced by the current dataset and protocol has concentrated. Agreement below one indicates competing extensions; agreement near one indicates high confidence in one dataset-induced complete rule. In the early rule experiments that motivated this conjecture, agreement was clearly below one only in overfitting regimes where the sample count lay below the grokking transition, whereas agreement near one occurred precisely when the training data were already sufficient for rule generalization. This paired observation motivated the conjecture that, in a sufficiently large problem space and for a nontrivial dataset, the complete function distribution approaches concentration only after the data stably identify a rule and cross the corresponding transition. A near-one-agreement dataset should therefore be likely to contain a human-readable rule that can in principle be discovered and extracted.**

[E27](experiments/e27.html) directly checks the temporal ambiguity in this intuition. Across four Mod97 data fractions, agreement on unseen inputs was only 0.027–0.033 at hard fit; memorizing the training set did not make it approach one. During grokking, agreement rose mainly because seeds recovered the same correct targets, while residual errors remained seed-specific. The current evidence therefore does not support one shared complete wrong function before grokking, nor a second agreement jump independent of accuracy.

High agreement does not guarantee recovery of the external generator prespecified by the researcher. One example can make networks converge unanimously to a constant extension, but a constant function is itself an extremely short and human-readable rule. It is therefore not a counterexample to the conjecture that high agreement selects readable rules. It only shows why the conjecture must be teacher-free: agreement identifies the rule jointly selected by the dataset and neural protocol, not necessarily the generator hidden behind the data. The conjecture is more specific:

> In a sufficiently large problem space, if a nontrivial training set makes the **complete function distribution** concentrate near one under strict fresh-seed auditing, the dataset usually corresponds to a shorter, human-readable logical rule. Discovering that rule can still be difficult.

Requiring more samples before concentration usually indicates a more complex identifiable rule or harder-to-eliminate competitors. The sample count needed for high agreement can therefore serve as another protocol-relative complexity measure.

The conjecture has survived two initial tests. Strict high-consensus endpoints in [E18](experiments/e18.html) all belonged to a signed-threshold family. In [E21](experiments/e21.html), longer anti-consensus prefixes increased sample requirements and moved endpoints from linear thresholds to more complex but still readable quadratic polynomial thresholds. These results support the conjecture but do not prove that every high-agreement function is human-readable.

The conjecture has a deeper implication. If high agreement continues to select human-readable short rules across more tasks, architectures, and larger spaces, then neural complexity and intuitive human complexity are not unrelated orderings. Both may exploit the same reusable structures: locality, symmetry, compositionality, low-order interactions, and shared computation. The original one-hundred-plus deterministic rule experiments provide broad evidence in the same direction: when an exact reusable generator exists and data and optimization access are sufficient, networks often drive both unseen-example loss and cross-seed function disagreement very low; underdetermination and noise do not automatically yield the same complete concentration.

> **If this survives broader stress tests, the implication is not one unique absolute “universal code,” but a class of robustly compressible structures that remain economical in many effective representation languages. Neural systems and humans may partly converge on similar complexity judgments because they face the same structure under finite computational resources.**

This remains a strong conjecture, not a conclusion. Architecture dependence and examples such as parity already show that the two complexity orderings do not coincide perfectly. The original rule matrix also did not apply the strict fresh-seed symbolic audit of E18/E21 throughout, so it is supporting breadth rather than a substitute for preregistered cross-architecture tests.

A possible application is rule discovery and data-sufficiency diagnosis. Once agreement approaches one, the dataset has likely constrained a stable rule; symbolic regression, program search, interpretable models, or human analysis can then attempt to extract it. When agreement remains low, arguing over one unique rule is premature.

## 8. Why symbolic concepts may emerge

OOD rule-composition experiments show that a network can interpret a rule code or role bit as reusable semantics and execute unseen combinations. The shared-computation experiment [E08](experiments/e08.html) shows that when several tasks require one expensive intermediate state, representation reuse reaches lower loss with less capacity and data.

Symbols and concepts may therefore be economical macroscopic representations under multi-task compression pressure, rather than structures imposed from outside connectionism. A stable symbol can bind roles, compose rules, and avoid storing a complete separate computation for each task.

The current evidence is behavioral and economic. It does not show one unique discrete hidden variable corresponding to each human symbol, nor that every semantic representation forms by the same mechanism.

## 9. Relation to existing AGI and deep-learning theories

### 9.1 A shared mathematical core: the Gibbs variational principle

Let $\mu_\Pi$ be the architecture-induced reference measure and $E_D(\theta)$ the training-data energy. Then

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

The first term fits the data and lowers loss; the second measures the complexity cost of moving $q$ away from the architectural reference measure. This “data fit plus descriptive deviation” is the genuine mathematical core shared with several AGI, cognitive, and statistical-learning frameworks.

Further decomposing parameter space by complete function $f$ yields function-restricted partition functions and free energies. That is the resolution emphasized here: not only total evidence, but competition among complete-function macrostates as loss, data, and architecture change.

### 9.2 Solomonoff induction and AIXI: a resource-bounded analogue

[Solomonoff induction](https://mlanthology.org/misc/1964/solomonoff1964misc-formal/) gives shorter programs greater prior weight. [Hutter’s AIXI](https://arxiv.org/abs/cs/0701125) combines universal sequence induction with action and reward maximization to define an incomputable ideal agent.

The corresponding neural object is

$$
P_\Pi(f)\propto e^{-C_\Pi(f)},
\qquad
C_\Pi(f)=-\log P_\Pi(f),
$$

where $\Pi$ is a concrete finite neural protocol rather than a universal Turing machine. It can be sampled and tested, but it loses the universal optimality claims of Solomonoff induction and AIXI.

The accurate connection is that a neural network can be viewed as a **computable, resource-bounded, architecture-relative induction system**. The present framework selects functions from data; it has no state–action–future-history–reward loop and is therefore not an AIXI-like agent theory.

### 9.3 Schmidhuber’s compression progress: from static compression to curiosity

[Schmidhuber’s compression-progress theory](https://people.idsia.ch/~juergen/driven2008.pdf) argues that curiosity should reward improvement in a compressor, not merely data that are already compressible. Interesting experience is both surprising and learnably regular.

This framework measures a sample’s free-energy increment:

$$
\Delta\Phi(z\mid D)=\Phi(D\cup\{z\})-\Phi(D).
$$

An already predicted sample has small increment. A sample eliminating many candidate functions has a large increment. If learning it also makes many future samples unsurprising, it produces genuine compression progress.

E21’s disagreement querying is a passive-learning version of this idea: search for samples most likely to change the current function distribution. A curiosity-driven agent would additionally choose environment actions and reward expected future compression progress.

### 9.4 Free Energy Principle, predictive coding, and active inference

[Friston and Kiebel (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/) formulate predictive coding and the Free Energy Principle as approximate Bayesian inversion of hidden causes using variational free energy. The mathematical structure is closely related, but the variables differ:

| This framework | Free Energy Principle / Active Inference |
|---|---|
| $q(\theta)$ is a distribution over parameters or program implementations | $q(z)$ is an approximate posterior over hidden environmental causes |
| Data are training examples | Data are sensory observations |
| Function free energy measures rule-implementation mass | Variational free energy measures how well an internal generative model explains observations |
| Disagreement locates informative training samples | Prediction error drives perceptual updating |
| No action loop yet | Active inference changes future observations through action |

Here $-\log Z$ is evidence under a specified measure; FEP typically minimizes a tractable variational upper bound on surprise. Introducing an approximate $q$ connects the two through the Gibbs variational identity above.

This is a mathematical and functional relation, not evidence for the biological version of FEP or proof that brains organize states by the complete-function macrostates measured here.

### 9.5 PAC-Bayes and Gibbs posteriors

[Catoni (2007)](https://arxiv.org/abs/0712.0248) explicitly uses temperature, Gibbs posteriors, and relative entropy to connect empirical risk with generalization bounds:

$$
q_\beta(\theta\mid D)
\propto
\mu_\Pi(\theta)e^{-\beta E_D(\theta)}.
$$

PAC-Bayes primarily asks how empirical risk and KL complexity control test risk for a chosen posterior. This project additionally decomposes $q$ by complete function, measures $Z_f(\beta)$ and function-rank changes, and compares the static distribution with real optimizer ensembles.

The mathematics is shared, but the resolution and question differ. A PAC-Bayes bound does not determine a Neural K-profile or prove that SGD samples the Gibbs posterior without bias.

### 9.6 Singular Learning Theory

[Watanabe’s Singular Learning Theory](https://doi.org/10.1017/CBO9780511800474) replaces ordinary parameter counting with RLCT to describe Bayesian evidence and free-energy asymptotics in singular models. [WBIC](https://jmlr.csail.mit.edu/papers/volume14/watanabe13a/watanabe13a.pdf) and the later [Local Learning Coefficient](https://proceedings.mlr.press/v258/lau25a.html) move this complexity toward estimable quantities.

A direct candidate relation is

$$
\text{RLCT / LLC}
\quad\longleftrightarrow\quad
\text{local or asymptotic contraction exponent of low-loss volume}.
$$

SLT mainly studies large-sample Bayesian asymptotics and local singular structure. This project measures finite data, finite networks, named complete-function macrostates, profile crossings, and optimizer accessibility. SLT may eventually derive volume slopes but does not currently replace the experiments.

### 9.7 Classical statistical mechanics of learning, flat minima, and local entropy

[Seung, Sompolinsky, and Tishby (1992)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.6056) already used Gibbs ensembles to study sample count, temperature, generalization curves, and learning transitions. Classical work focused primarily on teacher–student perceptrons, thermodynamic averages, and a few analytic order parameters.

[Flat Minima](https://doi.org/10.1162/neco.1997.9.1.1) and [Entropy-SGD](https://arxiv.org/abs/1611.01838) study width or local entropy near one parameter solution. The present framework groups all parameter regions implementing one complete function and asks how the global mass of that macrostate contracts with loss.

Function volume is therefore coarser than one minimum and closer to program selection, but it remains reference-measure and parameterization dependent rather than an absolute coordinate-free complexity.

### 9.8 Information Bottleneck and MDL probing

The [Information Bottleneck](https://arxiv.org/abs/physics/0004057) compresses input-irrelevant information in a representation while retaining target-relevant information. It is adjacent to “compression is intelligence,” but its object is mutual information between random variables, not parameter volume of a complete function. [Saxe et al. (2018)](https://openreview.net/forum?id=ry_WPG-A-) and [Kolchinsky, Tracey, and Van Kuyk (2018)](https://arxiv.org/abs/1808.07593) identify important limits of the claimed deep-network compression phase and degeneracies in deterministic settings.

[Voita and Titov’s MDL probing](https://aclanthology.org/2020.emnlp-main.14/) asks not only whether a layer linearly predicts a label, but how much online codelength is required to learn the label from that representation. This is closely related to treating a hidden representation as input and measuring the remaining free energy of the label map.

Both can serve as representation-level auxiliary measurements, but probe codelength, mutual information, parameter norm, Neural K, and machine-independent Kolmogorov complexity must not be equated.

### 9.9 Why this is still not a complete AGI theory

> **AGI positioning: the current framework explains induction over functions under a given dataset and neural protocol. It is presently a theory of learning and induction, not a complete AGI theory.**

A complete agent theory additionally needs

$$
\text{state}
+
\text{action}
+
\text{future prediction}
+
\text{reward}.
$$

One possible extension would choose actions by expected free-energy reduction, information gain, and external reward:

$$
a^*
=
\arg\max_a
\mathbb E
\left[
\Phi(D)-\Phi(D\cup z_a)
+
\lambda R(a)
\right].
$$

This would approach AIXI-style sequential decision making, active inference’s expected free energy, compression-progress curiosity, and active-learning information gain. No current experiment establishes this action loop, so it remains an extension path.

### 9.10 The potentially new connection

The project cannot claim to have invented free energy, Gibbs posteriors, MDL, AIXI, or the statistical mechanics of learning. Its more specific combination is:

1. decompose parameter space by complete-function macrostates;
2. measure each function’s full low-loss density of states / Neural K-profile;
3. connect per-example surprise to the same endpoint free energy;
4. prospectively predict random-dataset transitions from full-rule profiles;
5. treat optimizer deviation from static volume as nonequilibrium transport;
6. use agreement to measure teacher-free function concentration and test symbolic readability.

It is best understood as an experimentally measurable common coordinate system:

<div class="theory-link-chain" aria-label="Shared theoretical coordinate system">
  <span>Solomonoff / MDL</span>
  <span>Bayesian evidence / PAC-Bayes</span>
  <span>statistical-physics free energy</span>
  <span>predictive coding</span>
  <span>neural function volume</span>
</div>

## 10. Why the framework is not circular

Defining

$$
K^N_\Pi(f;\epsilon)=-\log_2V^{\mathrm{full}}_\Pi(f;\epsilon)
$$

as protocol-relative volume complexity is not itself falsifiable; it names a measurement. The scientific claim is whether that measurement predicts quantities outside its own definition.

Four independent anchors currently provide that test:

1. [E10](experiments/e10.html) preregisters strictly ordered simple/complex pairs by linear separability before measuring low-loss odds.
2. [E23](experiments/e23.html) measures and hashes full-rule volume scores before predicting random-dataset n50/n90 transitions; the parity ordering is exact.
3. [E25](experiments/e25.html) uses static branches to predict unseen MNIST labels and NLL turning points.
4. [E18](experiments/e18.html) and [E21](experiments/e21.html) use independent symbolic audits to test whether high-agreement endpoints are human-readable.

Counterexamples have changed the theory rather than being renamed away. The AND shortcut rejected a prespecified winner, weighted rule bits rejected all-range monotonicity, shallow random/parity mismatches rejected a one-slice scalar, deep crossing rejected hard-function identity as sufficient, and MUX3/parity2 rejected the claim that complete-rule difficulty alone determines the sample transition. The framework is therefore not “whatever the network selects is simple.”

## 11. Dataset size, loss depth, and regimes

For one rule, four common regimes appear:

1. **Underconstrained.** Too few examples leave many large extensions. Agreement can be low or can be misleadingly high on a constant or shortcut.
2. **Critical/grokking.** Data permits rule identification, but the rule dominates only at deep loss. Training fits first and the rule takes over later.
3. **Data-sufficient/direct learning.** The rule overwhelms alternatives at shallow loss, and training and validation fall together from early training. Most of the original successful rule experiments lie here.
4. **Finite precision and noise.** Any finite dataset supports only finite extrapolation precision. Deeper loss can emphasize dataset-specific residuals or incorrect labels, separating NLL, accuracy, or external rule performance.

Under pure hard conditioning, adding an example consistent with the target preserves the target cell while deleting competitors, so normalized hard target mass cannot decrease. No equally unconditional theorem holds at a fixed raw-BCE slice because the new example also changes the average loss and margin constraint.

Dataset size, model capacity, and training time play different roles: capacity controls representation, data controls identifiability, and dynamics controls accessibility. Collapsing all three into one “task difficulty” obscures the real differences among parity, Mod 3, grokking, and overfitting.

## 12. Current boundaries

The framework does not yet explain or derive:

- why broad neural architectures exhibit prior simplicity bias;
- why a specific function has its measured low-loss density of states;
- how to derive a complete Neural K-profile analytically from architecture;
- whether every pair of profiles eventually stabilizes or crosses finitely many times;
- whether high complete-function agreement generally implies a human-readable rule;
- or a closed quantitative relation between static volume and SGD/AdamW transport.

The current result is therefore an experimentally supported static–dynamic framework, not a completed unified law of deep learning.

> **Strongest claim: continuous reduction of training loss has function-selection meaning, and that selection can be measured through protocol-relative function volume, free energy, predictive branches, and sample transitions.**
