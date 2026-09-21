# MDL-identifiability samples, Neural K, and the grokking transition

This is the English companion to the conditional derivation in the release tree. It separates target implementation mass, competitor elimination by random samples, and late loss/optimizer dynamics. It does not identify grokking time with a complexity value.

## 1. Hard constraints and competitor survival

Fix a target \(f_*\), an input distribution \(q\), and a frozen function mass \(\pi(f)\). For a competitor \(g\), define
$$
\delta_q(g,f_*)=\Pr_{x\sim q}[g(x)\ne f_*(x)].
$$
Under independent noiseless samples, its probability of satisfying all \(n\) hard constraints is
$$
S_n(g)=(1-\delta_q(g,f_*))^n.
$$
This is a sampling identity and does not assume SGD.

For a realized dataset \(D_n\),
$$
M_n^{\rm comp}=\sum_{g\ne f_*}\pi(g)\mathbf 1[g\models D_n],
$$
and
$$
Q_n(f_*)=\frac{\pi(f_*)}{\pi(f_*)+M_n^{\rm comp}}.
$$
Adding data therefore attenuates different competitors at different rates, according to where they disagree with the target.

For random datasets, the mean-field proxy is
$$
\bar M_n^{\rm comp}=\sum_{g\ne f_*}\pi(g)(1-\delta_q(g,f_*))^n.
$$
It must not be silently identified with \(\mathbb E Q_{D_n}(f_*)\).

## 2. Conditional sample-transition formula

Define
$$
K_*=-\log_2\pi(f_*),\qquad
K_{\rm comp}^{\rm eff}(n)=-\log_2M_n^{\rm comp}.
$$
Then
$$
Q_n(f_*)=\frac{1}{1+2^{K_*-K_{\rm comp}^{\rm eff}(n)}}.
$$
The target reaches probability \(q_0\) when
$$
K_{\rm comp}^{\rm eff}(n)-K_*
\ge \log_2\frac{q_0}{1-q_0}.
$$

If competitors have approximately common disagreement rate \(\delta\), and their initial total mass is \(M_0\), then only under this extra approximation:
$$
M_n^{\rm comp}\approx M_0(1-\delta)^n,
$$
$$
K_{\rm comp}^{\rm eff}(n)
\approx K_{\rm comp}^{0}+n[-\log_2(1-\delta)].
$$
Consequently,
$$
n_{q_0}\approx
\frac{K_*-K_{\rm comp}^{0}+\log_2(q_0/(1-q_0))}
{-\log_2(1-\delta)}.
$$

This is the conditional reason complexity and \(n_{50}/n_{90}\) can be approximately ordered together within a rule family whose competitor-disagreement structure is controlled. It is a first-order or mean-field approximation, not a universal linear law.

## 2.1 The joint complexity-disagreement spectrum

In the general case,
$$
K_g=-\log_2\pi(g),\qquad
\lambda_g=-\ln(1-\delta_q(g,f_*)),
$$
and
$$
M_n^{\rm comp}=\sum_{g\ne f_*}2^{-K_g}e^{-n\lambda_g}.
$$
The effective competitor complexity is therefore
$$
K_{\rm comp}^{\rm eff}(n)
=-\log_2\sum_{g\ne f_*}2^{-K_g}e^{-n\lambda_g}.
$$

This does not assume that hard functions form a continuous manifold. A complex extension that disagrees only on rare inputs can survive many random samples. MUX3 has many low-disagreement copy/selector extensions, so changing the sampling distribution can reverse its ordering relative to Parity2 without changing the network, target, or loss.

## 3. The MDL crossing picture

Two-part MDL writes
$$
L_{\rm MDL}(g;D)=L_{\rm code}(g)+L_{\rm code}(D\mid g).
$$
For a noiseless target,
$$
L_{\rm code}(D\mid f_*)\approx0.
$$
For an incorrect extension,
$$
-\log_2 S_n(g)=n[-\log_2(1-\delta_q(g,f_*))].
$$
Thus the competitor's effective description cost grows with sample count while the target's program cost is approximately fixed. Classical program length, Neural K, parameter-prior cost, weight coding, and truth-table length are not interchangeable; MDL supplies the competition structure but does not prove that SGD explicitly minimizes machine-independent Kolmogorov complexity.

## 4. Connecting the data axis to grokking

For a fixed dataset \(D_n\), define
$$
\Omega_n(f;\epsilon)
=\mu_\Pi\{\theta:L_{D_n}(\theta)\le\epsilon,\ h_\theta=f\},
$$
and
$$
Q^{\rm static}_{n,\epsilon}(f_*)
=\frac{\Omega_n(f_*;\epsilon)}
{\sum_g\Omega_n(g;\epsilon)}.
$$

If the target becomes dominant only at deeper loss, there is a static transition region. The observed grokking time additionally contains optimizer access:
$$
\text{grokking time}
=\text{loss depth of the static quality transition}
+\text{optimizer access delay}.
$$
With abundant data the target can dominate at shallow loss (direct learning); near the identification boundary it can dominate only at deep loss (grokking); with insufficient data, continued loss reduction can favor a training-specific extension.

## 5. Why complexity correlates with the transition, but is not sufficient

Within a family with comparable competitors,
$$
K_*\uparrow
\Longrightarrow \pi(f_*)\downarrow
\Longrightarrow \text{more samples are needed to suppress competing mass}.
$$
Across families, a target can have large implementation mass but many hard-to-distinguish extensions, while another target can have smaller mass and highly informative samples. A useful factorization is
$$
n_q=\text{target implementation cost}
+\text{competing mass}
+\text{disagreement coverage}
+\text{optimization access},
$$
which is a decomposition of factors, not a literal additive identity.

## 6. Testable predictions and limits

1. Within families with similar disagreement spectra, complete-target Neural K should roughly order \(n_{50}\) and \(n_{90}\).
2. Enriching samples that distinguish MUX selector candidates should lower MUX3's transition sample count and can reverse its ordering.
3. Near the data-identification boundary, static profiles should show target mass overtaking competitor mass only at deeper loss before grokking.
4. If static target mass is already dominant but training does not generalize, optimization accessibility is the leading remaining explanation.
5. E32's approximately linear random-memory response is an external calibration for this competition model.

The framework does not claim \(n_{50}=K\), that grokking time equals an MDL crossing, that complexity alone determines sample demand, or that MDL proves SGD selects the true rule.

## 7. Current boundary and complete equation ledger

The defensible statement is that a complete-target Neural K-profile is a first-order predictor of identification demand when the competitor spectrum and disagreement structure are controlled. E23, the MUX3 interventions, E32, and grokking trajectories should be analyzed together rather than compressed into a universal formula.

For completeness, the source derivation also contains the following exact or conditional equations:
$$
y_i=f_*(x_i).
$$
$$
Q_n(f_*)\ge q_0
\Longleftrightarrow
M_n^{\rm comp}\le\frac{1-q_0}{q_0}\pi(f_*).
$$
$$
K_{\rm comp}^{0}=-\log_2M_0,\qquad
a_\delta=-\log_2(1-\delta).
$$
$$
n_{q_0}\approx
\frac{K_*-K_{\rm comp}^{0}+\log_2(q_0/(1-q_0))}
{a_\delta}.
$$
$$
M_n^{\rm comp}
\approx\iint 2^{-K}e^{-n\lambda}\,\rho(dK,d\lambda).
$$
$$
L_{\rm code}(D\mid f_*)\approx0.
$$

$$
Q_n(f_*)\ge q_0
\Longleftrightarrow
M_n^{\rm comp}\le\frac{1-q_0}{q_0}\pi(f_*).
$$

$$
K_{\rm comp}^{\rm eff}(n)-K_*
\ge\log_2\frac{q_0}{1-q_0}.
$$

$$
Q^{\rm static}_{n,\epsilon}(f_*)
=\frac{\Omega_n(f_*;\epsilon)}{\sum_g\Omega_n(g;\epsilon)}.
$$

These equations do not strengthen the claims above: they make the conditional assumptions and the distinction between exact sampling identities and mean-field approximations explicit.
