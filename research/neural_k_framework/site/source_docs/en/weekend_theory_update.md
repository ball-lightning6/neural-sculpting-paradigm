# Theory Update from 2026-09-19 to 2026-09-20

This file records the theory developed from Saturday to Sunday, separately from the earlier set of increments.

## 1. Statistical language changed from analogy to strict objects

This round separated the parameter measure, joint output, hard-function cells, fixed-D loss events, full-target events, shells, information gain, parameter/function KL, and optimizer distributions. The central chain is:

\[
\theta\longrightarrow Z_U(\theta)\longrightarrow \text{joint-output distribution}
\longrightarrow \text{loss/amplitude cross-section}
\longrightarrow \text{function-mass distribution}.
\]

Loss is a measurable event on the joint-output distribution, not a prior analogy to physical energy. The full-target Neural K profile and the fixed-D function-cell profile no longer share an unrestricted symbol.

## 2. The simplest feasible function under a maximum error

This was the most important MDL increment of the week. Fix an input domain, training set `D`, network protocol, and reference complexity `K_ref(f)`. When the mean loss is allowed to be at most `epsilon`, define:

\[
\mathcal F_D(\epsilon)
=
\{f:\Omega_{D}(f;\epsilon)>0\}.
\]

The minimum feasible complexity under the error budget is:

\[
K^*_D(\epsilon)
=
\inf_{f\in\mathcal F_D(\epsilon)}K_{\rm ref}(f).
\]

This is the strict definition of:

> What is the simplest function that can still complete the task when the maximum training error is specified?

For the same `D` and nested loss sublevel sets, lowering `epsilon` can only shrink the feasible set, so `K^*_D(epsilon)` cannot decrease. This is a strict property of the set definition. It does not say that the network chooses that function, or that the function has the largest parameter volume.

## 3. Simplicity bias among the remaining candidates

The quantity that describes the network's actual selection is the full mass distribution:

\[
Q_{D,\epsilon}(f)
=
\frac{\Omega_D(f;\epsilon)}
{\sum_g\Omega_D(g;\epsilon)}.
\]

Two processes must be distinguished:

1. **Feasibility elimination:** shortcuts or wrong extensions no longer reach the lower loss and leave `F_D(epsilon)`;
2. **Reallocation among feasible candidates:** functions that remain feasible lose implementation mass at different rates, so `Q` changes.

With a monotone loss-depth coordinate `t` and

\[
s_t(f)=-\partial_t\log\Omega_t(f),
\]

the fixed-complexity-scale identity is:

\[
\partial_tE_{Q_t}[\kappa]
=-\operatorname{Cov}_{Q_t}(\kappa,s_t).
\]

If complex functions contract faster, the covariance is positive and mean complexity decreases. If no new distant mode appears, the complexity distribution narrows. This is the precise conditional version of “simplicity bias strengthens as BCE/loss decreases.”

It is still a conditional mechanism, not a theorem for every loss. `s_t` is jointly determined by the output distribution, parameter measure, architecture, and loss; BCE is only one measured direction.

## 4. Relation to MDL

Two-part MDL is:

\[
L_{\rm MDL}(f;D)
=K_{\rm ref}(f)+L_{\rm residual}(D\mid f).
\]

`K^*_D(epsilon)` describes the minimum model-code-length frontier under a residual budget; `Q_{D,epsilon}` describes the mass distribution over remaining candidates at that budget. The former is the “simplest feasible solution” frontier; the latter is what the network actually favors among surviving implementations.

It is therefore wrong to write “MDL selects the simplest function” as if training code contained a second explicit program-length objective. The code explicitly lowers training loss; MDL-like preference comes from the joint effect of the architecture's reference mass and the geometry of the loss cross-section.

## 5. Relation to the grokking data transition

Adding data and lowering loss are different axes:

- data samples eliminate candidates according to where they disagree with the target;
- lowering loss continues to rearrange implementation mass within the fixed training-set joint-output cross-section;
- the grokking time axis also contains optimizer transport and reachability.

In a competing-function disagreement-spectrum model, the data transition is determined by target mass, total competitor mass, and the probability that each competitor is hit by a sample. Thus `K^*_D(epsilon)` and Neural K are important first-order objects, but neither is by itself `n50/n90` or the grokking time.

## 6. Other theory increments from the weekend

- Local contraction rates of joint outputs, endpoint interaction terms, and conditional additivity in finite windows;
- Robustness boundaries of complexity ordering under Gaussian, fixed-norm-ball, and natural-measure choices;
- The implementation-mass connection between AIT/Omega and Neural K, without machine-independent equivalence;
- The Peleg/G&C recheck as an external check on the relation between static low-loss mass and the SGD distribution.

## 7. Corrections concerning joint outputs, amplitude alignment, and contraction laws

The weekend discussion corrected a tempting derivation: one cannot assign an independent “function complexity” to a single sample, and one cannot simply multiply the volumes of multiple samples because shared network parameters create joint correlations.

The correct object for asking whether amplitude regularization favors simple functions is the within-cell retention rate under the same joint training event. A difference in BCE contraction rates does not automatically imply a corresponding difference in function odds after amplitude alignment. Translation symmetry can reduce certain coordinates, but cannot by itself prove exponential volume contraction.

A finite-window contraction approximation needs additional conditions:

- the common measure change induced by output translation is small in the window;
- the interaction cost of adding positive and negative output groups is small;
- deep-tail local errors are controlled; two endpoints cannot determine an entire line.

What can be released is therefore a conditional derivation plus falsifiable experimental protocol, not a cross-architecture or infinite-tail contraction law.

## 8. Full boundary of the Peleg/G&C weekend recheck

The literature audit found that Peleg's G&C work does consider loss, but its normalized loss bins are not the same object as this project's raw-BCE deep-tail event. The active parameter region, initialization scale, and learning rate also need to be kept separate between G&C and SGD.

The local recheck obtained: hard-start G&C about 82.13%, low-loss static samples about 96.37%, and the key SGD group about 96.36%. This supports predictive value in static low-loss mass, but does not show that the static distribution equals the SGD distribution, and does not allow the difference to be attributed simply to “insufficiently low loss.”

## 9. Minimal content retained from AIT/Omega

Omega, incompleteness, and algorithmic probability provide a structure for summing implementation mass, coding length, and uncomputability boundaries. They do not imply:

- Neural K equals machine-independent Kolmogorov complexity;
- lower complexity variance equals a lower function-mixture correction;
- finite-network agreement inherits the undecidability of the halting problem.

This is not uploaded as a separate literature audit; only the boundaries that entered the strict statistical language are retained in the theory appendix.

## 10. Algorithmic mutual information is a future interface, not a completed experiment

Rule-bit or concatenated outputs may provide an approximate conditional information cost for redundancy and synergy among output blocks. This is not an independently completed experiment, and it is not automatically identifiable information about the true rule. Small tasks and network/input-output head overhead can create substantial numerical differences between protocols that are theoretically close, so the draft experiment is not presented as a completed result.

## 11. Correspondence with the Complete Documents

This page is the public body-text entry for the weekend theory update. The release package also keeps the complete source documents:

| Document | Role | Web entry |
|---|---|---|
| `minimal_strict_statistical_language_zh.md` | Minimal strict definitions, objects, and conditional identities | [Strict probability language](../../site/en/post-upload-language.html) |
| `minimal_strict_statistical_language_tutorial_zh.md` | Step-by-step teaching version | [Full theory increment](../../site/en/post-upload-theory-update.html) |
| `minimal_statistical_language_audit_zh.md` | Previous-draft gaps, finite counterexamples, and corrections | [Full theory increment](../../site/en/post-upload-theory-update.html) |
| `MDL_FRONTIER_AND_GROKKING.md` | Minimum feasible complexity and rearrangement among candidates | [MDL-grokking](../../site/en/post-upload-mdl.html) |
| `mdl_identifiability_grokking_derivation_zh.md` | Conditional derivation using the complexity-disagreement spectrum | [MDL-grokking](../../site/en/post-upload-mdl.html) |
| `MEASURE_ROBUSTNESS_AND_GAUSSIAN_ALIGNMENT_ZH.md` | Gaussian, Euclidean-ball, and training-correspondence boundaries | [E37 Measure robustness](../../site/en/experiments/e37.html) |

The web body uses the public versions whose links and bilingual structure have been validated. The source documents retain the full derivations, audit records, and rerun entry points; they are not redundant substitutes for one another.

Complete source files can be opened directly: <a href="../source_docs/zh/minimal_strict_statistical_language_tutorial_zh.md">Chinese teaching version</a> · <a href="../source_docs/zh/minimal_strict_statistical_language_zh.md">Chinese strict-definition version</a> · <a href="../source_docs/en/minimal_statistical_language_audit.md">English audit and corrections</a> · <a href="../source_docs/en/weekend_theory_update.md">English weekend update</a>.
