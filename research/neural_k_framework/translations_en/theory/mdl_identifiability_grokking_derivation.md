# MDL-identifiability samples, Neural K, and the grokking transition

Date: 2026-09-20.

This memo addresses why complexity often correlates with the grokking transition. It separates three quantities and develops a testable connection:

1. Full-target implementation mass under the current neural protocol;
2. The efficiency with which random training samples eliminate competing functions;
3. Late function reallocation caused by loss reduction and optimizer transport after fixing the training set.

It does not define \(n_{50}/n_{90}\) as Neural K or identify grokking time with an MDL crossing. It gives a bridge model under explicit assumptions.

## 1. Hard constraints and competitor survival

Fix a finite input domain \(\mathcal X\), target \(f_*\), input sampling distribution \(q(x)\), and normalized reference function mass \(\pi(f)\), with \(\pi(f_*)>0\). This mass \(\pi(f)\) may be the initialization function mass or function mass under a fixed static protocol; it must be frozen before the model is applied and cannot be replaced directly by potentially overlapping full-target event masses.

Draw \(n\) inputs independently from \(q\) and observe noiseless labels:
\[
y_i=f_*(x_i).
\]

For a competitor \(g\), define its disagreement probability with the target:
\[
\delta_q(g,f_*)=\Pr_{x\sim q}[g(x)\ne f_*(x)].
\]
The probability that competitor \(g\) matches a single random sample is \(1-\delta_q(g,f_*)\). Under independent sampling, its probability of satisfying all \(n\) hard constraints is
\[
S_n(g)=(1-\delta_q(g,f_*))^n.
\]
This is purely a sampling probability and does not involve SGD.

For a realized dataset \(D_n\), the total surviving mass of incorrect competitors is exactly
\[
M_{D_n}^{\rm comp}=\sum_{g\ne f_*}\pi(g)\mathbf 1[g\models D_n].
\]
The target agrees with every true label and is not eliminated. Its hard-conditioned probability is
\[
Q_{D_n}(f_*)=\frac{\pi(f_*)}{\pi(f_*)+M_{D_n}^{\rm comp}}.
\]
This gives an exact conclusion:

> Along a nested data sequence, each competitor survives or is eliminated according to whether a sample hits its disagreement set; the smooth attenuation spectrum describes expected mass over random datasets.

For random datasets, the survival indicator of an individual competitor \(g\) satisfies
\[
\mathbb E_{D_n}\mathbf 1[g\models D_n]
=(1-\delta_q(g,f_*))^n.
\]
The expected competitor mass is therefore exactly
\[
\bar M_n^{\rm comp}
=\mathbb E_{D_n}M_{D_n}^{\rm comp}
=\sum_{g\ne f_*}\pi(g)(1-\delta_q(g,f_*))^n,
\]
while the proxy obtained by normalizing mean unnormalized masses is defined as
\[
\bar Q_n(f_*)=\frac{\pi(f_*)}{\pi(f_*)+\bar M_n^{\rm comp}}.
\]

These are an expectation identity and a proxy definition, not one shared approximation. Only when using \(\bar Q_n\) to approximate \(\mathbb E_{D_n}Q_{D_n}(f_*)\) do we additionally need concentration or self-averaging of surviving mass. The expectation of a ratio is not generally the ratio of expected masses, and neither is automatically a dataset-level recovery rate.

An incorrect function that disagrees with the target on very few inputs can survive many random samples even if it is complex. This is the structural explanation for the MUX3/Parity2 misordering.

## 2. Conditional-mass, proxy, and recovery-rate thresholds

For the fixed dataset above, reaching target probability \(q_0\in(0,1)\) is equivalent to
\[
Q_{D_n}(f_*)\ge q_0
\quad\Longleftrightarrow\quad
M_{D_n}^{\rm comp}\le\frac{1-q_0}{q_0}\pi(f_*).
\]
Define effective competitor complexity:
\[
K_*=-\log_2\pi(f_*),\qquad
K_{\rm comp}^{\rm eff}(D_n)=-\log_2M_{D_n}^{\rm comp}.
\]
Then
\[
Q_{D_n}(f_*)=\frac{1}{1+2^{K_*-K_{\rm comp}^{\rm eff}(D_n)}}.
\]
Reaching \(q_0\) is equivalent to
\[
K_{\rm comp}^{\rm eff}(D_n)-K_*
\ge \log_2\frac{q_0}{1-q_0}.
\]

Along a fixed nested data sequence \(D_\bullet\), denote the first sample count reaching this conditional-mass threshold by
\[
\tau_{q_0}(D_\bullet)=\inf\{n:Q_{D_n}(f_*)\ge q_0\}.
\]
Separately denote the mean-mass proxy threshold by
\[
\bar n_{q_0}=\inf\{n:\bar Q_n(f_*)\ge q_0\}.
\]
E23's `n50/n90` instead refer to dataset-level recovery-rate thresholds. Fixing training protocol \(\Pi\), the recovery criterion, and a grid \(\mathcal G\) of non-complete sample sizes, write
\[
n^{\Pi}_{\mathrm{rec},r}
=\min\{n\in\mathcal G:n<|\mathcal X|,\;
\Pr_{D_n}[\mathrm{Recover}_\Pi(f_*\mid D_n)]\ge r\}.
\]

Here \(r=0.5\) or \(r=0.9\) corresponds to experimental `n50` or `n90`, respectively; \(q_0\) is a target-mass threshold within one conditional distribution or proxy, a different probability level. A missing crossing remains right-censored and cannot be replaced by fitting the complete dataset. The first two thresholds are not target complexity itself, but ask:

> When does the effective total competitor complexity exceed target complexity by enough as samples accumulate?

If competitors have approximately common disagreement rate \(0<\delta<1\) and initial total mass \(M_0\), expected mass satisfies
\[
\bar M_n^{\rm comp}\approx M_0(1-\delta)^n.
\]
Write
\[
K_{\rm comp}^{0}=-\log_2M_0,\qquad
a_\delta=-\log_2(1-\delta),
\]
so that
\[
\bar K_{\rm comp}^{\rm eff}(n)
=-\log_2\bar M_n^{\rm comp}
\approx K_{\rm comp}^{0}+n a_\delta.
\]
Here \(\bar K_{\rm comp}^{\rm eff}\) is the negative log of expected mass, not the expectation of random cost. Solving the \(\bar Q_n\) threshold under the common-disagreement approximation yields the proxy formula below, in the positive-crossing regime, with integer sample counts additionally requiring rounding:
\[
\bar n_{q_0}\approx
\frac{K_*-K_{\rm comp}^{0}+\log_2\frac{q_0}{1-q_0}}
{a_\delta}.
\]

This approximation explains why complexity and proxy sample demand can be approximately proportional within a family with similar disagreement structure: larger \(K_*\) means smaller initial target mass, requiring more samples to attenuate expected total competitor mass. Connecting it to a realized data sequence additionally needs mass concentration or self-averaging; connecting it to E23 additionally needs correspondence between the recovery criterion and static target mass, as well as optimization accessibility. It is not an exact equation for experimental `n50/n90`.

It also shows why this is not a universal law:

- \(K_*\) can differ;
- Initial total competitor mass \(M_0\) can differ;
- Competitor disagreement rates \(\delta_g\) can differ;
- Sampling distribution \(q\) can differ.

The MUX3 counterexample primarily changes the last two, not the target's own implementation mass.

## 2.1 The joint complexity-disagreement spectrum

The intuition that different hard functions resemble a continuous distribution or manifold can be expressed through a joint competitor spectrum. For each competitor \(g\), record
\[
K_g=-\log_2\pi(g),\qquad
\lambda_g=-\ln(1-\delta_q(g,f_*)),
\]
so that expected competitor mass is exactly
\[
\bar M_n^{\rm comp}=\sum_{g\ne f_*}2^{-K_g}e^{-n\lambda_g}.
\]
This does not assume that hard functions form a continuous manifold; it embeds a discrete function set in an implementation-cost/disagreement-rate plane. Many nearby extensions typically have similar \(\delta_q\) and attenuate slowly together as samples accumulate. A function differing only on rare inputs has small \(\lambda_g\) and can survive for a long time even with large \(K_g\).

The proxy effective competitor complexity is therefore
\[
\bar K_{\rm comp}^{\rm eff}(n)
=-\log_2\sum_{g\ne f_*}2^{-K_g}e^{-n\lambda_g}.
\]
This is not the complexity of one competitor, but the negative log of the spectrum's expected total mass at sample count \(n\). Each added sample in the average spectrum multiplies each competitor by a survival factor determined by its disagreement set; realized \(M_{D_n}^{\rm comp}\) is still determined by hard survival indicators.

This gives the connection between complexity and sample transitions:

- If total competitor mass and the \(\lambda_g\) distributions are similar across rule families, \(K_*\) can dominate the ordering of proxy thresholds \(\bar n_{q_0}\); connecting this to experimental `n50/n90` still needs the bridge conditions above;
- If \(\lambda_g\) distributions differ substantially, sample coverage can outweigh target implementation complexity;
- MUX3 is difficult because of many low-disagreement competing extensions, so full-target Neural K alone cannot predict its transition;
- Changing sampling distribution \(q\) changes \(\delta_q\), directly changing the transition even when the network, loss, and target stay fixed.

Under a continuous approximation with joint spectral measure \(\rho(K,\lambda)\),
\[
\bar M_n^{\rm comp}
\approx\iint 2^{-K}e^{-n\lambda}\,\rho(dK,d\lambda).
\]
This is more general than a linear complexity/sample-count relation. Approximate linearity is a special case when the competitor spectrum concentrates around similar \(\lambda\), or when effective residual cost in random memory grows approximately independently per sample.

## 3. The MDL crossing picture

Two-part MDL writes
\[
L_{\rm MDL}(g;D)=L_{\rm code}(g)+L_{\rm code}(D\mid g).
\]
Given the target rule \(f_*\) in the noiseless setting, training labels require almost no additional residual code:
\[
L_{\rm code}(D\mid f_*)\approx0.
\]
An incorrect extension occasionally disagrees with training samples, and its residual code length grows with sample count. Under a hard-consistency approximation, its probability of passing all samples corresponds to
\[
-\log_2 S_n(g)=n[-\log_2(1-\delta_q(g,f_*))].
\]
This motivates an effective cost proxy based on random-sample survival probability:
\[
L_{\rm comp}^{\rm proxy}(g;n)
\approx K_{\rm ref}(g)
+n[-\log_2(1-\delta_q(g,f_*))].
\]
It grows with \(n\), while the target's program cost is approximately fixed. This survival-probability cost is not the actual residual code length of a realized dataset; it provides an MDL-style competition analogy at the mean-mass level. One can therefore discuss a proxy cost crossing:
\[
L_{\rm comp}^{\rm proxy}(g;n)\approx K_{\rm ref}(f_*).
\]
This is the MDL prototype of why a complex rule can require more samples to be identified against memorization or incorrect extensions.

But the coding convention behind \(K_{\rm ref}\) must be explicit:

- Classical program length;
- Neural K;
- Negative log parameter probability;
- Weight code length;
- Function-table code length.

These are not interchangeable. MDL provides structural similarity and a language for candidate competition, not a proof that SGD explicitly minimizes machine-independent Kolmogorov complexity.

## 4. Connecting the data axis to grokking

The model above concerns the data-size axis: how adding samples eliminates competitors.

Grokking usually extends training time after fixing the training set, which is a different axis. Continuous loss acts on forward output \(F_\theta\); the hard output \(h_\theta=R(F_\theta)\) only identifies the complete function. Write the readout map on a fixed input domain as \(H_U\). For a fixed dataset \(D_n\), define
\[
\Omega_{n}(f;\epsilon)
=\mu_\Pi\{\theta:L_{D_n}(\theta)\le\epsilon,\ h_\theta=f\},
\]
so the target's share of the low-loss static distribution is
\[
Q^{\rm static}_{n,\epsilon}(f_*)
=\frac{\Omega_n(f_*;\epsilon)}
{\sum_g\Omega_n(g;\epsilon)}.
\]

If target mass grows relative to competitors as \(\epsilon\) decreases, define a static transition region:
\[
\epsilon_{q}(n)=\text{target static probability reaches }q\text{: corresponding loss region}.
\]
The transition need not be monotone because profiles can cross. Record the first threshold-reaching interval instead of assuming one unique exact point.

The actual trained-function distribution is
\[
\nu_{n,t}=(T_{n,t})_\#\mu_{\rm init},
\qquad
Q^{\rm opt}_{n,t}=(H_U)_\#\nu_{n,t}.
\]
$T_{n,t}$ outputs trained parameters, and training randomness is included in the initial joint distribution and protocol. The loss depth of the static mass transition and the dynamics of reaching and traversing that region jointly influence grokking time. This is a mechanism sketch, not an equation adding loss depth to time. Any time decomposition must first convert loss depth to an arrival time through an actual training trajectory, then separately define and measure any subsequent delay.

This explains three possible patterns:

- With abundant data, the target dominates at shallower loss and direct learning occurs;
- Near the identification boundary, the target dominates only at deeper loss and grokking occurs;
- With insufficient data, continued loss reduction may never provide sufficient target advantage, leaving training on an incorrect extension.

The data and time transitions can therefore be connected, but not equated.

## 5. Why complexity correlates with the transition, but is not sufficient

Within a family with comparable competitors and disagreement rates,
\[
K_*\uparrow
\quad\Longrightarrow\quad \pi(f_*)\downarrow
\quad\Longrightarrow\quad \text{more samples are needed to suppress competing mass}.
\]
Complexity and proxy-threshold rankings can then approximately agree. Agreement with experimental \(n_{50}/n_{90}\) is further expected only under bridge conditions including mass concentration, correspondence between the recovery criterion and static mass, and optimization accessibility.

Across families, rankings can reverse:

- One target has large parameter mass but many hard-to-distinguish incorrect extensions;
- Another has smaller parameter mass, but each random sample efficiently eliminates competitors.

MUX3/Parity2 is such a case. Increasing the frequency of inputs that expose MUX-selector conflicts lowers MUX3's recovery sample count and can reverse its ordering relative to Parity2. The target, network, and loss stay fixed; \(\delta_q(g,f_*)\) changes.

Four factors must therefore be controlled separately: target implementation cost, competitor mass, disagreement coverage, and optimization accessibility. They jointly affect the dataset-level recovery threshold \(n^\Pi_{\mathrm{rec},r}\); they are not same-unit terms that can simply be added.

## 6. Testable predictions

The model gives predictions more specific than "complex rules need more data":

1. For rules with similar competitor mass and disagreement structure, satisfying the static-to-recovery bridge conditions above, \(n_{50}\) and \(n_{90}\) should roughly increase with full-target Neural K.
2. Increasing the proportion of inputs that distinguish MUX-selector candidates should lower \(n_{50}/n_{90}\) and can reverse the complexity ordering.
3. Near the identification boundary at fixed data size, static profiles around grokking should show target mass overtaking total competitor mass at deeper loss.
4. In direct-learning tasks, static target advantage should appear at shallower loss, with training and validation curves decreasing together.
5. If static target mass already dominates but training does not generalize, the discrepancy primarily concerns optimizer accessibility rather than MDL competition itself.
6. Approximately linear growth of memory residual cost with sample count in random-label experiments should serve as a calibration control for the MDL crossing model.

## 7. Strict boundaries of the current conclusion

The current defensible statement is:

> A Neural K-profile is a first-order predictor of rule-identification sample demand; when competitor disagreement structure is controlled, it can explain and predict an approximate relationship between complexity and the sample transition.

The framework does not currently claim:

- \(n_{50}\) equals Neural K;
- Grokking time equals an MDL crossing;
- Complexity alone determines sample demand;
- Complexity rankings persist under every random sampling distribution;
- MDL has proved that SGD selects the true rule.

The most useful next independent step is to back-test existing E23 results, the MUX3 sampling interventions, random-memory measurements, and grokking trajectories within one competitor-mass model, not to introduce another elaborate formula.
