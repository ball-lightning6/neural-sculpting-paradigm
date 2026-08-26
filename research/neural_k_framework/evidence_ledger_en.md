# Neural K Evidence Ledger

> **Status: internal evidence ledger, not a finished paper.** This English edition preserves the measurement definitions, decisive numbers, negative results, and scope boundaries. The [research narrative](index.html) is the recommended first read.

## Abstract

Overparameterized neural networks can fit one finite training set with many incompatible complete functions, yet repeated training often concentrates on a small subset. Function-prior research has shown that random neural parameterizations are strongly nonuniform and often biased toward simple functions. This ledger tests what that static picture misses when training loss is treated as a continuous coordinate.

The evidence separates four objects: the initialization function prior; static function-resolved mass at a fixed training set and loss threshold; the optimizer-induced pushforward over parameters and functions; and time samples from one trajectory. Complete-function enumeration, constrained SMC, paired interventions, multi-seed training, and real-image branch prediction show that these objects are related but not interchangeable.

The central static result is differential contraction: compatible functions, and continuous implementations inside one hard-function cell, lose parameter mass at different and scale-dependent rates as raw loss is tightened. Rankings can reverse in the deep tail. The resulting Neural K-profile prospectively predicts parity sample transitions and produces strong MNIST label predictions, while optimizer-accessibility experiments show that static preference alone does not guarantee reachability.

## 1. Framing and scope

The optimizer receives the training objective and any explicit regularizer. It does not receive the researcher’s generator, validation set, semantic invariances, or a separate instruction to generalize. A finite dataset defines a family of compatible complete extensions, not a privileged “true function” inside the training objective.

The broad repository matrix predates the present theory. It includes cellular automata, arithmetic, logic, algorithmic tasks, visual transformations, rule-conditioned pattern recognition, OOD rule composition, active-learning interventions, and explicit failures such as high-dimensional parity and endpoint-only Mod 3. Its value is breadth: exact rule learning is real across many controlled protocols, but representational capacity, data identifiability, and optimization accessibility remain distinct.

The small Boolean experiments are measurement instruments. They do not claim that all real tasks are 3-bit systems. They make complete functions, compatible sets, and loss-resolved volume exactly observable so that explanations suggested by the broad experiments can be falsified.

## 2. Measurement objects

### 2.1 Neural reference protocol and training loss

$$
\Pi=(\mathcal A,\varphi,\mu,\ell),
$$

where architecture and parameterization, encoding, parameter reference measure, and per-example loss are fixed together. For

$$
D=\{(x_i,y_i)\}_{i=1}^{n},
\qquad
L_D(\theta)=\frac1n\sum_i\ell(h_\theta(x_i),y_i),
$$

the loss sublevel set is

$$
A_D(\epsilon)=\{\theta:L_D(\theta)\le\epsilon\}.
$$

Every volume claim in this archive is relative to an explicit measure. It is not a coordinate-free Euclidean volume.

### 2.2 Function-resolved mass under a fixed dataset

$$
\Omega_{\Pi,D}(f;\epsilon)
=
\mu_\Pi\{\theta:L_D(\theta)\le\epsilon,\ h_\theta=f\},
$$

and

$$
Q^{\mathrm{static}}_{\Pi,D,\epsilon}(f)
=
\frac{\Omega_{\Pi,D}(f;\epsilon)}{\sum_g\Omega_{\Pi,D}(g;\epsilon)}.
$$

The rejected separable hypothesis is that every function receives one fixed factor multiplied by a common loss-dependent factor. E06, E10, E12, E14, E19, and E24 show genuine function-specific reweighting and ordering changes.

### 2.3 Full-target volume and Neural K

For a complete finite target,

$$
V^{\mathrm{full}}_\Pi(f;\epsilon)
=
\mu_\Pi\{\theta:L_{D_{\mathrm{full}}(f)}(\theta)\le\epsilon\},
$$

and

$$
K^N_\Pi(f;\epsilon)=-\log_2V^{\mathrm{full}}_\Pi(f;\epsilon).
$$

The full curve is the Neural K-profile. It is protocol-relative and not claimed to equal machine-independent Kolmogorov complexity.

### 2.4 Optimizer-induced distribution

$$
Q^{\mathrm{opt}}_{\Pi,D,t}=(T_{\Pi,D,t})_\#\mu_\Pi.
$$

Static mass can be a strong first-order determinant, while entrances, gradients, connectivity, optimizer state, numerical environment, and history generate additional transport. E01, E07, E16, and E17 directly test this separation.

### 2.5 Predictive branches and free energy

For one unseen input,

$$
P_{\mathrm{hard}}(y\mid x,D,\epsilon)
=
\frac{\mu_\Pi\{\theta\in A_D(\epsilon):h_\theta(x)=y\}}
{\mu_\Pi(A_D(\epsilon))}.
$$

The canonical partition function and free energy are

$$
Z_{\Pi,D}(\beta)=\int e^{-\beta nL_D(\theta)}d\mu_\Pi(\theta),
\qquad
F_{\Pi,D}(\beta)=-\beta^{-1}\log Z_{\Pi,D}(\beta).
$$

E22 verifies that per-example generalized-surprise increments telescope to the complete-rule endpoint cost under one coherent static measure. It does not claim that real SGD samples the Gibbs ensemble.

## 3. Experimental evidence chain

The following sections are generated from the same English experiment records used by the E01–E25 detail pages. Each preserves the motivation, actual measurement, decisive result, and limitation.

## 4. Final integrated picture

### 4.1 Two static empirical principles

1. The parameter-to-function map is highly nonuniform: a fixed neural protocol gives some functions much more reference mass than others.
2. Tightening raw training loss contracts functions and within-function implementations at different, scale-dependent rates.

These are weaker than saying that every prespecified simple function rises monotonically. They are compatible with mid-loss shortcuts, nonmonotone candidate mass, and deep ranking reversals.

### 4.2 Data amount and loss depth

Random-dataset `n50/n90` is a protocol-relative identification/recovery sample complexity, not an equivalent operational definition of the full-target Neural K-profile. The latter measures a complete-target numerator; the former additionally depends on the fixed-$D$ denominator of compatible extensions, the sampling distribution and effective pattern coverage, recovery thresholds, and optimizer reachability. E23 prospectively links the quantities within the parity family, while the MUX3/parity2 exception shows that cross-family order can differ when these intermediate factors change.

A causal coverage/shortcut intervention confirms the distinction. Under uniform sampling, parity2/MUX3 had `n50=64/80` and `n90=80/104`. Strict eight-cell balancing changed these to `56/72` and `64/88` without changing the order. Raising selector-conflict examples to 75% changed them to `n50=72/56` and `n90=88/72`. Relative to uniform sampling, MUX3 moved 24/32 examples earlier while parity2 moved eight later at both thresholds. Two thousand paired-bootstrap replicates and every evaluated step from 500 through 40,000 retained the direction. A sampling change alone can therefore reverse an operational transition while the complete target and network remain fixed.

Fixed-$D$ Gaussian SMC then removed the optimizer and reproduced the same cross-target ordering statically. At `n=32, epsilon=0.02`, uniform parity2/MUX3 target masses were 0.266/0.000214, cell masses were 0.469/0.284, and conflict masses were 0.498/0.782. Conflict-MUX3 made the target modal in 8/8 datasets, versus 0/8 under uniform sampling. Posterior errors localized to selector-conflict cells: uniform target accuracy was 0.993 on ordinary cells but 0.777 on conflict cells, while conflict enrichment raised the latter to 0.995. Deep-tail lineages fell to roughly one or two per replica, so absolute decimals are coarse; the conflict-versus-uniform direction nevertheless held in all eight datasets and all 32 conflict-MUX3 replicas had nonzero target mass. The static competing denominator is therefore a first-order source of the reversal, while optimizer transport accounts for remaining quantitative differences.

Under pure hard conditioning, adding an example consistent with a complete target retains that target cell and deletes some competitors, so normalized target mass cannot decrease. Fixed raw-BCE sublevels have no equally unconditional monotonic theorem because the new example also changes the average loss and margin constraints.

Empirically, more balanced rule-consistent data extends the rule-aligned descent channel to deeper loss. Small datasets support large but often generator-misaligned extensions. Near a transition, the rule wins only in a deep tail and appears as grokking. With abundant data, the rule dominates from shallow loss and training and validation fall together. Any finite dataset still supports only finite extrapolation precision.

### 4.3 Grokking, overfitting, and noise

Interpolation fixes training signs, not the complete function or internal representation. Continued raw-loss reduction changes margins and relative masses. An early shortcut can be the most economical solution at its loss scale; the reusable rule can take over later.

Overfitting occurs when deeper training resolves distinctions specific to the finite dataset rather than the external generator. Agreement can keep rising while calibration or external accuracy stops improving. Under label noise, shared structure is learned first and example-specific corrupt residuals later.

### 4.4 Feature reuse and symbolic structure

Shared computation gives a direct route from one explicit loss objective to economical representation. E08 shows that reusable intermediate computation can reach deeper loss with less capacity and data. E05 shows that representation change can occur even when a fixed kernel already generalizes, so feature learning is a mechanism, not a synonym for generalization.

Rule-code OOD behavior and the consensus-symbolicity experiments support a candidate bridge from connectionist computation to symbolic macrostates: reusable bindings can be cheaper than separate lookup programs. The evidence is behavioral and protocol-specific, not a proof about brains or all LLMs.

### 4.5 Agreement

Agreement measures concentration, not recovery of a hidden teacher. High agreement can identify a narrow empirical function ensemble without knowing a generator, including a constant rule that is wrong relative to that generator but remains short and human-readable. Single-example constant concentration is therefore not a counterexample to the readability conjecture; it only refutes the claim that agreement automatically recovers the external target. Low agreement indicates unresolved extension competition but does not diagnose whether the cause is insufficient data, noise, multimodality, or inaccessible optimization.

The open conjecture is one-way: in the measured settings, strict high complete-function agreement repeatedly led to short extractable threshold programs. This suggests partial alignment between neural and human symbolic complexity, but does not establish a universal theorem.

If the correspondence persists across more architectures and larger spaces, it would suggest more than an accidental preference for readable functions. Neural and human complexity may both track robustly compressible structure such as locality, symmetry, compositionality, and shared computation. “Objective” here would not mean one absolute code, but broad agreement among multiple effective representation languages over a class of reusable regularities. Parity and architecture-dependent reversals already show that such alignment can only be partial. The one-hundred-plus deterministic rule tasks add breadth because sufficient data often produced both very low unseen-example loss and high function consistency, but they did not all use the same fresh-seed symbolic audit and therefore do not replace E18/E21.

## 5. Relation to external theory

The static foundation comes from the function-prior and simplicity-bias work of [Dingle et al.](https://www.nature.com/articles/s41467-018-03101-6), [Valle-Pérez et al.](https://arxiv.org/abs/1805.08522), and [Mingard et al.](https://www.jmlr.org/papers/v22/20-676.html). The present experiments add continuous loss, complete-function transport, deep-tail volume flow, and prospective sample-transition prediction.

The algorithmic-information and coding background includes [Solomonoff](https://mlanthology.org/misc/1964/solomonoff1964misc-formal/), [Kolmogorov](https://www.mathnet.ru/eng/ppi68), [MDL](https://arxiv.org/abs/math/0406077), and [prequential coding in deep models](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3b712de48137572f3849aabd5666a4e3-Abstract.html). Neural K is not identified with any one of these quantities.

The relation is nevertheless precise enough to test. Standard MDL separates hypothesis codelength from residual-data codelength. In this framework, `-log2 V_full(f; epsilon)` is the protocol-relative implementation codelength for realizing a complete target to precision `epsilon`, and the full Neural K-profile is that codelength as residual tolerance changes. Training usually minimizes loss alone; MDL-like selection emerges from nonuniform architectural reference mass and loss geometry. E22's endpoint-codelength closure and E23's prospective `profile -> n50/n90` prediction are independent empirical anchors beyond terminology.

The statistical-mechanics lineage begins at least with [Levin, Tishby, and Solla (1989)](https://mlanthology.org/colt/1989/levin1989colt-statistical/) and includes modern finite-width analyses such as [Pacelli et al. (2023)](https://www.nature.com/articles/s42256-023-00767-6). Flat minima, local entropy, SLT/RLCT, and LLC are adjacent mass-based approaches with different objects and assumptions.

## 6. Evidence boundaries and next falsifiable tests

The project has not derived profile shape from architecture, proved finite crossing counts, or made optimizer transport analytically predictable. It has not proven that all high-agreement functions are human-readable or that finite data can support arbitrarily deep correct extrapolation.

The MNIST turning-point result is calibration-assisted, not yet blind. The deepest SMC magnitudes require independent implementation and larger particle/replica audits. Cross-architecture profile-to-transition predictions and a new digit-pair prediction are the clearest next tests.

The evidence supports a unified experimental framework over the tested Boolean, cellular-automaton, modular-arithmetic, small-MLP, and binary-MNIST protocols. It does not establish one universal quantitative law for all modern architectures or real-world datasets.
