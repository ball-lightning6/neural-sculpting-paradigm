# Neural K measurement methods for training snapshots and finite samples

Date: 2026-10-21. Status: website methods draft, for the Chinese and English websites only, not the formal papers or PDFs. This document retains method definitions and estimation proposals awaiting calibration and reports no new measurements; the new MNIST snapshot measurements and Rule30 proxy-accuracy validation are unfinished, and this update launches no experiments.

The two methods connect: first, treat the hard function implemented by any training snapshot as a target and measure its Neural K; second, use data transitions and query saturation to select finite target samples and reduce SMC evaluation costs on large tasks. The former expands what can be measured; the latter attempts to reduce measurement cost.

## 1. Fixing the measurement object

For source-network weights theta_t at training time t, define the hard output function

$$
f_t(x)=h(F_{\theta_t}(x)).
$$

The measurement object is the function implemented by the source weights, not the byte count, compressed length, or numerical storage complexity of the source weight file. Source weights are used only to generate target behavior; identical hard functions on the same measurement domain specify the same target event even when the weight representations differ.

Measurement uses a separate, fixed reference protocol Pi: a reference network and parameterization, parameter probability measure mu, input domain U or input distribution rho, hard readout rule, and output-error definition. The reference network may have the same architecture as the source network, but SMC estimates target-event mass under the reference measure, not the probability that SGD takes the source network to that state.

Keep the same reference protocol Pi across all snapshots being compared. Changes in the reference network, measure, input scope, or precision nodes must be marked separately as changes in the measurement protocol, not interpreted directly as changes in source-function complexity.

For a finite input domain U, let D_U(f_t) be the complete target table and define

$$
V_{\Pi,U}^{\mathrm{full}}(f_t;\epsilon)
=\mu_\Pi\{\phi:L_{D_U(f_t)}(\phi)\le\epsilon\},
\qquad
K_{\Pi,U}^{N,\mathrm{full}}(f_t;\epsilon)
=-\log_2 V_{\Pi,U}^{\mathrm{full}}(f_t;\epsilon).
$$

For an input distribution rho, define the corresponding event using expected loss and specify how finite queries estimate that expectation. Record Neural K as a profile; compare snapshots at fixed measurement precision or the same set of precision nodes.

For binary outputs, BCE can define the target event above; multiclass and other outputs require the corresponding loss and readout rule. An average-loss threshold need not force every hard output to be correct. Full-target mass and hard-cell mass remain distinct as in the [strict probability language](release_20260920/theory/minimal_strict_statistical_language_zh.md).

## 2. Method one Generating target functions from training snapshots

Save the source network at prespecified training nodes and generate hard labels on a fixed input domain or set of query inputs. Each snapshot yields a full or partial target table; run SMC under the same reference protocol to obtain comparable Neural K curves.

This permits measuring transient functions without readable rule names, structured overfitting functions, and final functions. Source networks with different neuron permutations or weight values but identical hard behavior specify the same target under the same measurement protocol.

The measurement domain must include inputs outside the training set. After the source network fully fits the original training set, using hard labels only on those training inputs cannot observe continuing changes in extensions on unseen inputs.

With hard targets, unchanged classification behavior on the same measurement domain means unchanged target-function cost. Changes in continuous logits or margins may be defined separately as output-reproduction cost, but are not mixed into this method.

### 2.1 Proposed application to MNIST training trajectories

Fix a reference classification network, measure, and query protocol. Generate labels and measure complexity for initialization, early learning, snapshots around training-set fit, and late training. Record the source network's actual training loss, test performance against true labels, changes in hard predictions, and Neural K separately.

The relationships to test are how cost changes while structure is learned and exceptions are fitted, and whether selection of economical extensions can lower cost after sufficient fitting in some settings. The eventual curve for a noisy task may keep rising, stabilize, or decline locally.

The source network's predictions on query images supply measurement labels, which are distinct from the images' true classes. If the input domain is a fixed image library, conclusions are relative to that library; if a random input distribution is used, specify its construction and an independent audit set.

## 3. Method two Approximating a full rule with samples beyond its transition

First use actual training to roughly locate a target function's data-recovery transition. Choose a sample count beyond that transition in a stable generalization region, and then run SMC on those target samples. The working hypothesis is that these queries distinguish the main incorrect extensions, so further queries have progressively less effect on target-event mass.

The Rule30 example discussed is a one-step 30-bit to 30-bit mapping. Its complete input domain has 2^30 states; if the data transition is around 900, approximately 2000 samples are a candidate measurement design. These numbers describe a research proposal, not a completed validation of high-precision volume estimation with 2000 samples.

What is directly measured is the mass for a finite target table D,

$$
V_{\Pi,D}(\epsilon)
=\mu_\Pi\{\phi:L_D(\phi)\le\epsilon\}.
$$

Use this to estimate full-rule Neural K only when it reliably approaches full-target mass in the specified measurement window. The SGD recovery transition locates candidate sample sizes; the extension distribution in static SMC, independent queries, and stability under added samples provide evidence about proxy accuracy. Crossing the SGD recovery transition is not sufficient for proxy calibration; before calibration, report only the mass of the finite-query target event, not a validated estimate of full-function complexity.

Target functions from different snapshots may require different sample counts. The original task's transition cannot automatically cover the exceptions and error structure of intermediate functions; the same undersized query set may also underestimate complexity late in training.

## 4. Judging whether the proxy is sufficient

Use nested query sets while holding the source target, reference protocol, and precision window fixed. Record changes in negative log mass as queries are added, and inspect candidates' extensions on independent inputs excluded from the SMC constraints. Evidence of saturation should come from multiple query sets and independent sampling replicas, rather than a single SGD curve.

Distinguish agreement of hard behavior from agreement of continuous loss events. If the target's complete hard cell has conditional mass q inside a finite-target event A_D, the exact relation is

$$
-\log_2\mu_\Pi(A_D\cap C_U(f))
=-\log_2\mu_\Pi(A_D)-\log_2 q.
$$

For example, q=0.99 gives a hard-cell correction of approximately 0.0145 bits. This error connects only the finite-target event to its target-hard-cell branch; it does not automatically connect to the full-domain average-loss event at the same threshold.

Implementations of the same hard function may have different logits and losses. Moreover, A_D and A_U defined by the same average-loss threshold are generally not nested: a network with low average loss over the full domain may still have relatively high loss on D, and conversely. High hard generalization alone therefore cannot guarantee a high-precision estimate of full-target volume.

If pointwise-tolerance events are used instead, the full-domain constraints include the finite-query constraints, allowing exact accounting with a conditional retention rate. Such an event must be named separately and cannot silently replace the existing average-BCE protocol. The actual average-loss method requires auditing the loss distribution on unqueried outputs and checking stability under added queries within the chosen precision window.

## 5. Using the two methods together

1. Fix the reference network, measure, input scope, readout rule, and measurement precision.
2. Generate target hard labels from source-network snapshots.
3. First compare full-target and finite-query SMC on enumerable tasks to calibrate error.
4. For large tasks, locate sample counts using transitions or query saturation and retain an independent input audit.
5. Use the same cost scale for snapshots, recording necessary query expansion and estimation uncertainty.

When comparing training snapshots across widths, a fixed reference network avoids mixing changes in source functions with changes in the measurement language. If the object of study is each architecture's own complexity, retain architecture subscripts and report it separately from fixed-reference measurement.

## 6. Relationship to existing evidence

[E23](translations_en/experiments/e23.md) supports a relationship between static rule costs and data recovery, while showing that sampling identifiability and the competitor denominator cannot be omitted. It does not prove that any finite sample set beyond an SGD transition approximates full-target volume.

Looking up fixed K values for functions along training trajectories on small truth tables was a precursor to method one. The new proposal aims to measure targets generated by arbitrary snapshots directly, using method two to reduce costs on large input spaces. MNIST trajectories and Rule30 proxy accuracy still require experimental validation; the [website explanation of width and double descent](width_double_descent_en.md) describes the mechanism relationships these measurements would test. These steps are a method proposal, not experiments completed for this release or experimental tasks that must be executed now.
