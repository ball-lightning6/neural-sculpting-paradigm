# MDL minimum viable complexity, rearrangement of remaining candidates, and grokking

This is the uploadable version of the theoretical discussion from 2026-09-19 to 2026-09-20. For the full derivation see [MDL identifiable samples and the derivation of grokking conditions](mdl_identifiability_grokking_derivation_zh.md); for the strict probabilistic objects see [minimal statistical language](minimal_strict_statistical_language_zh.md).

## Minimum viable complexity

Fix the training set `D`, the network protocol, and the reference complexity `K_ref(f)`. For a full function `f`, let `Omega_D(f; epsilon)` be the parameter mass within the function cell whose training loss does not exceed `epsilon`. Then:

\[
\mathcal F_D(\epsilon)=\{f:\Omega_D(f;\epsilon)>0\},
\qquad
K_D^*(\epsilon)=\inf_{f\in\mathcal F_D(\epsilon)}K_{\rm ref}(f).
\]

It is the rigorous definition of the problem "the simplest feasible function when the maximum error is allowed". For the same training set and nested sublevel sets, when `epsilon` decreases, `F_D(epsilon)` can only shrink, and therefore `K_D^*(epsilon)` does not decrease.

This rigorous property does not imply that the network necessarily selects that function, nor that that function has the largest parameter volume.

## Simplicity bias among the remaining candidates

What the network actually corresponds to is:

\[
Q_{D,\epsilon}(f)=
\frac{\Omega_D(f;\epsilon)}{\sum_g\Omega_D(g;\epsilon)}.
\]

Lowering the loss has two effects:

1. eliminating the shortcuts that no longer satisfy the training constraint;
2. redistributing mass among the functions that remain feasible.

If `t` is the loss-depth coordinate and `s_t(f)=-partial_t log Omega_t(f)` is the local contraction rate of a function, then:

\[
\partial_t E_{Q_t}[\kappa]
=-\operatorname{Cov}_{Q_t}(\kappa,s_t).
\]

When complex functions contract faster, the mean complexity decreases; if no new modes reappear, the complexity distribution narrows. This is a conditional mechanism jointly determined by the joint output distribution, the parameter measure, and the loss, not a universal theorem for all losses.

## MDL interpretation

\[
L_{\rm MDL}(f;D)=K_{\rm ref}(f)+L_{\rm residual}(D\mid f).
\]

`K_D^*(epsilon)` corresponds to the lowest model code-length frontier, and `Q_D,epsilon` corresponds to the mass distribution over the remaining feasible candidates. The network does not explicitly optimize a second "program length" objective; the MDL-style preference arises from the joint action of the network reference mass and the loss geometry.

## Relation to grokking

The data-size axis, the loss axis, and the training-time axis must be kept separate: data samples eliminate functions according to the positions where competing functions disagree, loss tightening rearranges the remaining implementation mass, and training time additionally includes optimizer transport. Therefore complexity can have a conditional approximate relationship with `n50/n90`, but it is not equal to the data transition point or the grokking time point.
