# The most mature Neural K estimation method to date

This directory keeps only the currently recommended estimator and its final backtest; it does not upload the earlier iterations on precision correction, two-layer failure, and task prediction.

## Method

The current method is:

1. Directly compute the cumulant generating function (CGF) of the finite-width joint outputs from a fixed single-hidden-layer `tanh` network with a Gaussian parameter prior;
2. Compute the Chernoff action for the full-truth-table average BCE event;
3. Approximate the full event probability using fixed saddlepoint / local Gaussian boundary curvature corrections;
4. Use conditional integration for nodes whose conditional integrals pass numerical diagnostics, and otherwise fall back to the saddlepoint estimate.

For the core objects, applicability conditions, and approximation boundaries, see the [estimator method description](ARCHITECTURE_ESTIMATOR_METHOD.md).

## Final backtest

- 3-bit: all 40 symmetric orbits, 5 BCE levels;
- 4-bit: 64 pre-fixed orbits, 5 BCE levels;
- 520 rule×loss nodes in total;
- predictor parameters are not fitted with SMC K;
- clearly better than the old empirical formula on both 3-bit and 4-bit, but deep-tail errors still grow.

Final results are in the [520-node backtest report](NEURAL_K_LARGE_BENCHMARK_REPORT.md). It is a backtest on historical data, not a fully blind test, and not a general exact algorithm for arbitrary architectures.

## Basis in the original papers

The method of this project is inspired by the following mathematical work, but is not a direct theorem of theirs:

- Fischer et al., *Critical feature learning in deep neural networks*, ICML 2024: finite-width random feature/kernel distributions and representation fluctuations; [PMLR](https://proceedings.mlr.press/v235/fischer24a.html).
- Bauer et al., *A unified theory of feature learning in RNNs and DNNs*, 2026: finite-width representation distributions and the action perspective; [arXiv:2602.15593](https://arxiv.org/abs/2602.15593).

These papers provide mathematical background; the CGF, BCE convex duality, and local boundary corrections in this directory are concretizations specific to this project's single-hidden-layer protocol and cannot be directly generalized to multi-layer, CNN, RNN, or Transformer architectures.

## Follow-up research

The final estimator drew on the following follow-up research during its development, but in each case only borrowed interfaces or diagnostic designs, and does not claim to use their theorems:

- Oketch, Fewster & Goodman (2026), *A general framework for computation and estimation using the saddlepoint approximation*: the implementation interface for third- and fourth-order derivative saddlepoint corrections ([arXiv:2607.17464](https://arxiv.org/html/2607.17464v1), the authors' R package saddlepoint); the accompanying foundation is *What is the price of approximation?* by the same authors ([arXiv:2504.07324](https://arxiv.org/html/2504.07324v1)).
- Katsevich (2025), *Saddle Point Approximation and Central Limit Theorem for Densities in high dimensions*: research on multiplicative errors of high-dimensional saddlepoint approximations, used for diagnostic design and proof-gap localization; it currently cannot attach rigorous error bars to the K values in this directory ([arXiv:2510.21545](https://arxiv.org/html/2510.21545v1)).
- Voldoire, Chopin, Rateau & Ryder (2024), *Saddlepoint Monte Carlo and its Application to Exact Ecological Inference*: a numerical-correction candidate combining exponential tilting with characteristic-function integration; its name is close to this project's parameter-space sequential Monte Carlo, but the algorithm is different ([arXiv:2410.18243](https://arxiv.org/html/2410.18243v1)).
- Papagiannouli, Trevisan & Zito (2026), *Beyond NNGP: Large Deviations and Feature Learning in Bayesian Neural Networks*: a layer-by-layer kernel-cost recursion, and a candidate route for extending this estimator to multiple hidden layers ([arXiv:2602.22925](https://arxiv.org/html/2602.22925v1), the authors' JAX code ldnn).
- Macci et al. (2026), *Functional Large Deviations for Wide Deep Neural Networks with Gaussian Initialization and Lipschitz Activations*: a large-deviation picture from finite inputs to entire output functions ([arXiv:2601.18276](https://arxiv.org/html/2601.18276v1)).

Provenance note: the finite-width CGF was already derived and implemented in this project on 2026-08-31; this round reconnected to and extended that line of work — see the [provenance correction](PROVENANCE_CORRECTION.md) for details.

## Kept source files

`current_method/` keeps the minimal reproduction scripts and protocol: the finite-width CGF, the final 520-node backtest, validation, and plotting. SMC checkpoints, historical iterations, and intermediate caches are not in the release tree.
