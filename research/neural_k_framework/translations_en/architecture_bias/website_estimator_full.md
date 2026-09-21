# Neural K estimator: finite-width CGF, saddlepoint correction, and the 520-node backtest

This page combines the method note, the current-method implementation note, the backtest reports, the protocol, and the provenance correction. It is the complete website-facing explanation of the current pre-training estimator.

## 1. Scope and protocol

The estimator applies to the fixed protocol used here: a single-hidden-layer width-16 tanh network, 0/1 inputs, Gaussian whitened neural coordinates, the original fan-in scaling, and a full-truth-table mean raw-BCE sublevel event. It is not an exact algorithm for arbitrary depth, CNN, RNN, Transformer, encoding, or prior.

The goal is to estimate the low-loss volume and the corresponding Neural K before running SMC. The reference SMC values are used for a frozen backtest, not to fit per-rule correction coefficients.

## 2. Exact finite-width joint-output CGF

For the 3-bit and 4-bit experiments, let the input table have \(r=2^b\) rows and let the hidden width be \(m=16\). With independent Gaussian neural coordinates,
$$
Z=\frac1{\sqrt m}\sum_{j=1}^{m}a_jh_j
+\frac{b_o}{\sqrt m}\mathbf 1,
\qquad
h_j(x)=\tanh((w_j\cdot x+b_j)/\sqrt b).
$$

After integrating the independent Gaussian readout variables, the finite-width cumulant generating function is
$$
\Lambda(t)=\log\mathbb E e^{t^T Z}
=\frac{(\mathbf1^Tt)^2}{2m}
+m\log\mathbb E_{w,b}
\exp\left[\frac{(t^Th)^2}{2m}\right].
$$

This is an exact identity for the specified finite-width prior. It does not use an infinite-width approximation and does not calibrate parameters from SMC. The single-neuron integral has only \(b+1\) independent Gaussian coordinates and is evaluated by multidimensional Gauss–Hermite quadrature.

Unlike a covariance-only Gaussian approximation, the CGF retains finite-width representation fluctuations of all orders. The derivation is inspired by finite-width Gaussian-mixture and feature-distribution work, but the BCE estimator and its reductions are specific to this project protocol.

## 3. Chernoff action for the BCE event

Let \(s_i=2y_i-1\) and let \(E\) be the event that the mean BCE over the full truth table is at most \(\\epsilon\). Define
$$
I(E)=\sup_t\{\inf_{z\in E}t^Tz-\Lambda(t)\}.
$$

Chernoff's inequality gives
$$
\Pr(E)\le e^{-I(E)},
\qquad
K(E)\ge I(E)/\ln 2.
$$

The Chernoff action is therefore a directional lower bound on the negative log probability; it is not by itself the complete Neural K.

For BCE, parameterize (t=s\odot\omega), (omega_i\ge0), introduce (alpha>\max_iomega_i), and solve the finite-dimensional concave maximization
$$
I(E)=\sup_{\omega,\alpha}
\left\{
\alpha\sum_i H(\omega_i/\alpha)
-r\alpha\epsilon
-\Lambda(s\odot\omega)
\right\}.
$$

The implementation checks the average BCE, KKT residuals, and the Legendre identity. The mathematical Chernoff inequality is exact, but the floating-point quadrature is not interval-certified, so numerical outputs are not certified rigorous bounds.

## 4. Representation cost is included

Under the optimal exponential tilt, the action decomposes into the KL cost of tilting the hidden representation distribution plus the readout tilt cost. Thus the method does not replace rare hidden representations by an arbitrary minimum-norm output solution. The finite-width representation fluctuations are explicitly charged in the calculation.

## 5. From action to a practical probability estimate

The action is converted into a complete event-probability approximation by a local Gaussian approximation under the tilted distribution and a second-order expansion of the nonlinear BCE boundary. The fixed approximation includes the Gaussian half-space factor and a tangential boundary-curvature determinant correction.

This correction is an approximation, not an identity. It has no universal error bound over all widths, rules, or loss depths. The implementation also retains the uncorrected Chernoff action so that the dominant action is not confused with the full probability approximation. The benchmark is a theoretical prediction checked against historical SMC; it is not a second independent sampler fitted to the same SMC values.

## 6. Backtest design

The large benchmark contains:

- all 40 strictly symmetric 3-bit orbits at five BCE thresholds;
- 64 preregistered 4-bit orbits at five thresholds;
- 520 rule-by-loss nodes in total;
- no per-rule SMC fit of estimator parameters;
- fixed numerical screening before comparison.

The comparison retains original SMC replicas, warnings, rechecks, and both median-volume and volume-mean conventions. A node passing the numerical screen is not thereby certified to have error below 0.05 bits; a node failing the screen is not thereby proven wrong.

## 7. Quantitative results

The method clearly improves on the old logarithmic and Hill empirical baselines in the shallow and intermediate regions.

At BCE 0.001, the K-ranking Spearman correlations against the original SMC medians are:

- 3-bit: 0.999625;
- 4-bit: 0.997482.

For the finite-window contraction coefficient
$$
c=\frac{K(0.001)-K(0.01)}{\log_2(10)},
$$
the method preserves the observed deep-tail ordering very well:

- 3-bit coefficient (R^2=0.999594);
- 4-bit coefficient (R^2=0.992877);
- coefficient Spearman correlations 0.997561 and 0.988324.

The coefficient is a finite-window slope. It is not claimed to be the final asymptotic coefficient as (epsilon\to0).

The full benchmark has 248/520 nodes passing the fixed numerical screen. The remaining nodes are retained with approximation or unconfirmed labels. The main limitations are importance-weight concentration, replica disagreement, and occasional quadrature warnings. Deep 4-bit tails remain systematically harder.

## 8. Why this is not merely an empirical fit

The predictor does not fit a separate proportionality, offset, or rule-specific parameter to SMC K. It computes the joint-output CGF from the architecture and prior, solves the BCE convex dual, and applies a fixed saddlepoint/boundary approximation. The benchmark therefore tests whether the architecture-induced joint-output distribution contains computable complexity information.

This is still a protocol-relative result. It does not establish a universal Neural K formula, a machine-independent Kolmogorov complexity theorem, or an exact prediction of SGD endpoint frequencies.

## 9. Provenance and reproducibility

A prior finite-width CGF derivation and implementation already existed in the project on 2026-08-31. The current work reconnects that route to a faster saddlepoint approximation, a complete 3/4-bit backtest, and the documented 4-bit transfer. It is not the first discovery of finite-width CGF in this project.

The reproducible method, protocol, validation scripts, slope CSV, summary JSON, and benchmark reports are kept under `evidence/architecture_bias/current_method/`. SMC checkpoints and intermediate caches are not required for the public method description.

## 10. Supported and unsupported claims

Supported:

- the finite-width joint-output distribution contains computable information about loss-resolved implementation mass;
- the fixed estimator predicts the scale, ordering, and finite-window contraction coefficients of many rules in this protocol;
- the method transfers from the 3-bit development family to preregistered 4-bit nodes without fitting new rule parameters.

Not supported:

- one formula works for all architectures and priors;
- deep-tail probabilities are already exact;
- the estimator replaces SMC in every regime;
- estimated Neural K directly gives the true SGD function frequency;
- Neural K is identical to machine-independent Kolmogorov complexity.
