# A strict probability language: joint outputs, function mass, and information updates

This is the website's entry point to the theory. The [Core Theory page](theory-core.html) provides the main framework, while this page states the strict objects and audit boundaries. The original formal specification, tutorial, and verification scripts remain in the release tree. The formulas below are definitions or identities under stated conditions. “Complex functions contract faster” does not follow automatically from them.

## 1. Fix the output object and the measure first

Fix a protocol `Π`: parameter space, architecture, encoding, parameter reference probability measure `μ`, per-example loss, and hard readout rule. Fix a finite input table `U`. The continuous joint output `Z_U(θ)=(F_θ(x))_{x∈U}` induces the pushforward measure `ν_U=(Z_U)_#μ`. The hard function `H_U(θ)` is the complete discrete output table on the same inputs.

The cells `C_U(f)={θ:H_U(θ)=f}` are disjoint and cover parameter space; their reference masses `p_U(f)=μ(C_U(f))` sum to one. “Complete” here is relative to the specified `U`. A single hard cell can contain different logits, margins, and losses. Changing the loss while leaving the network, readout, and reference measure fixed changes the event imposed on `ν_U`, not the underlying joint-output distribution.

## 2. Cumulative loss mass is not an exact-loss shell

Mean training loss on `D` defines `A_D(ε)={θ:L_D(θ)≤ε}`. The cumulative mass of a function branch and its normalized conditional distribution are:

$$
Ω_{U,D}(f;ε)=μ(C_U(f)∩A_D(ε)),\qquad
Q_{U,D,ε}(f)=\frac{Ω_{U,D}(f;ε)}{\sum_gΩ_{U,D}(g;ε)}.
$$

`Q` is the cumulative ensemble below a loss threshold. For a thin loss window use `Ω_f(b)-Ω_f(a)`. To discuss an exact-loss density one must first establish that the relevant loss measure has a density `ρ_f` on that interval. Where differentiability and positive denominators hold, the local contraction index is `λ_f(ε)=ερ_f(ε)/Ω_f(ε)`; the function distribution in a thin shell is the cumulative `Q` reweighted by `λ_f`. These are not interchangeable distributions.

## 3. Full-target, fixed-D, and training endpoints are different distributions

`K^{N,full}(f;ε)=-log₂ μ{L_{D_full(f)}≤ε}` measures the loss volume of a complete target. `K^{N,FD}_{U,D}(f;ε)=-log₂ Ω_{U,D}(f;ε)` measures the intersection of a fixed partial dataset event with a function cell. The cell factor can be omitted only when the full-target threshold already forces every hard output to be correct. In general these profiles must not share one suffix-free `K` symbol.

The function distribution produced by SGD or Adam is the pushforward of the initialization measure under the training map; it is not the static conditional distribution above. Static mass can make first-order predictions, while entrances, gradients, connectivity, numerical noise, and history can still alter the endpoints. A change of parameter measure must be stated explicitly; Gaussian probability is not the Euclidean volume of an unbounded parameter space.

## 4. Adding an example denotes three different operations

Fix the old event `E=A_D(ε)` and use one input table that already includes the query point. At least three distinct operations can be meant by adding a sample:

1. **Hard conditioning:** retain parameters in the old event whose predicted label is `y`, obtaining `E∩B_y`;
2. **Continuous filtering inside the old parent:** retain `E` and require the new mean loss to meet another threshold, obtaining `E∩A^+_y`;
3. **Redefining the event over all parameter space:** require only `A^+_y={L_{D∥(x*,y)}≤ε'}`, without separately requiring old `E`.

The third event can absorb parameters from outside the old event, so it is generally not the conditional update in the first or second operation. The second operation requires the joint distribution of the old loss and continuous output at the query point. A hard label alone is insufficient because one hard output can have many margins.

## 5. Surprise, KL, and within-function information

In the first operation the old ensemble predicts the new hard label with `p_E(y|x*)=μ(B_y|E)`. If that probability is positive, the information cost of the observed label is exactly `-log₂ p_E(y|x*)` bits. The direction of the KL identity is essential:

$$
D_{KL}(μ(\cdot|E∩B_y)\,\|\,μ(\cdot|E))=-\log p_E(y|x_*).
$$

The reverse KL generally diverges. This is the surprise of an observed label, not expected information gain before observation and not the information acquired by retraining SGD. For a function partition, parameter KL further decomposes into KL between function probabilities plus conditional parameter KL within each hard cell. Large within-cell reallocation remains possible even if the distribution of complete hard functions is unchanged.

## 6. Complexity distributions and conditional narrowing

If candidates are grouped by an external cost `c(f)` frozen before scanning, its distribution is the pushforward of `Q` through `c`. It is neither a histogram counting functions nor a histogram of the current values of `-log Ω_f`. When loss is tightened at a fixed dataset, put `r_f=Ω_f(ε₂)/Ω_f(ε₁)` (`ε₂<ε₁` and only where the old mass is positive). Then

$$
Q_2(f)=\frac{Q_1(f)r_f}{\sum_gQ_1(g)r_g},\qquad
\mathbb E_{Q_2}c-\mathbb E_{Q_1}c=
\frac{\operatorname{Cov}_{Q_1}(c,r)}{\mathbb E_{Q_1}r}.
$$

A lower retention rate for complex functions is sufficient to lower mean fixed cost. The covariance sign is empirical; lower variance or eventual monopoly by one function needs additional conditions. Shortcut elimination and contraction of complex functions among surviving candidates must be measured separately. Neither follows from nested loss events alone.

## 7. Audit and limits of use

The itemized audit checks the old/new mean-loss event substitution, the mistaken assumption that every parameter in a hard cell survives, the replacement of fixed-D by full-target mass, the direction of KL, the difference between densities and cumulative mass, and extra assumptions in amplitude-translation arguments. The finite verification script checks formulas and counterexamples; it does not prove that neural networks generally contract complex functions faster. The original audit and script remain in the release tree.

This language specifies the objects measured and the probability updates precisely. It does not replace empirical tests such as E20, E22, E23, and E31, or prove that SGD samples from the static measure. The finite verification scripts and JSON results remain available as reproducible release materials.
