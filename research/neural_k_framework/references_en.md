# References and Their Role in This Archive

> This page lists the external works explicitly mentioned or directly used by the current website. Links prioritize journal, conference, PMLR, JMLR, OpenReview, ML Anthology, or arXiv primary pages.

## 1. Function priors, parameter–function maps, and simplicity bias

### R01 · Dingle, Camargo, and Louis (2018)

Kamaludin Dingle, Chico Q. Camargo, Ard A. Louis. **Input–Output Maps Are Strongly Biased Towards Simple Outputs.** *Nature Communications* 9, 761 (2018). DOI: 10.1038/s41467-018-03101-6.

- [Nature Communications](https://www.nature.com/articles/s41467-018-03101-6)
- **Role here:** General background for simplicity bias in input–output maps; also an important reminder that the result is primarily an upper-bound relation and does not make every simple output probable.

### R02 · Valle-Pérez, Camargo, and Louis (2019)

Guillermo Valle-Pérez, Chico Q. Camargo, Ard A. Louis. **Deep Learning Generalizes Because the Parameter-Function Map Is Biased Towards Simple Functions.** *ICLR 2019*.

- [arXiv:1805.08522](https://arxiv.org/abs/1805.08522)
- **Role here:** A direct precursor for nonuniform neural function priors and a key source of function-space PAC-Bayes reasoning.

### R03 · Mingard et al. (2019)

Chris Mingard, Joar Skalse, Guillermo Valle-Pérez, David Martínez-Rubio, Vladimir Mikulik, Ard A. Louis. **Neural Networks Are a Priori Biased Towards Boolean Functions with Low Entropy.** arXiv (2019).

- [arXiv:1909.11522](https://arxiv.org/abs/1909.11522)
- **Role here:** Separates low-output-entropy bias from additional structural-complexity bias within Boolean functions.

### R04 · Mingard et al. (2021)

Chris Mingard, Guillermo Valle-Pérez, Joar Skalse, Ard A. Louis. **Is SGD a Bayesian Sampler? Well, Almost.** *Journal of Machine Learning Research* 22(79):1–64 (2021).

- [JMLR](https://www.jmlr.org/papers/v22/20-676.html)
- **Role here:** The closest prior work to the multi-seed empirical function distribution. It supports first-order predictive power of a Bayesian function posterior while retaining optimizer- and hyperparameter-dependent second-order deviations.

### R05 · Mingard et al. (2025)

Chris Mingard, Henry Rees, Guillermo Valle-Pérez, Ard A. Louis. **Deep Neural Networks Have an Inbuilt Occam’s Razor.** *Nature Communications* 16, 220 (2025).

- [Nature Communications](https://www.nature.com/articles/s41467-024-54813-x)
- **Role here:** E04 reuses its core 7-bit Boolean, deep-tanh, and advSGD protocol while extending observation beyond first zero training error.

### R06 · Mingard et al. (2025)

Chris Mingard, Lukas Seier, Niclas Göring, Andrei-Vlad Badelita, Charles London, Ard A. Louis. **Characterising the Inductive Biases of Neural Networks on Boolean Data.** arXiv (2025).

- [arXiv:2505.24060](https://arxiv.org/abs/2505.24060)
- **Role here:** A close modern example of architecture as a program language; its discrete-network/DNF correspondence is a useful comparison for the architecture relativity of Neural K.

## 2. Grokking, kernel limits, and feature learning

### R07 · Power et al. (2022)

Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra. **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.** arXiv (2022).

- [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- **Role here:** Establishes delayed generalization after memorization on small algorithmic datasets and the dataset-size axis of grokking.

### R08 · Nanda et al. (2023)

Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt. **Progress Measures for Grokking via Mechanistic Interpretability.** arXiv (2023).

- [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- **Role here:** Decomposes modular-addition grokking into memorization, Fourier-circuit formation, and cleanup, providing a direct comparison for continuous coordinates beneath an apparently abrupt transition.

### R09 · Jacot, Gabriel, and Hongler (2018)

Arthur Jacot, Franck Gabriel, Clément Hongler. **Neural Tangent Kernel: Convergence and Generalization in Neural Networks.** *NeurIPS 2018*.

- [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)
- **Role here:** Supplies E05’s infinite-width fixed-kernel baseline for asking whether a finite network still reorganizes features when kernel learning is already sufficient.

### R10 · Soudry et al. (2018)

Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, Nathan Srebro. **The Implicit Bias of Gradient Descent on Separable Data.** *Journal of Machine Learning Research* 19(70):1–57 (2018).

- [JMLR](https://www.jmlr.org/papers/v19/18-188.html)
- **Role here:** Shows that logistic/cross-entropy optimization continues to drive the max-margin direction after zero classification error; this is a narrow mechanism that must be separated from complete-function drift.

### R11 · Göring et al. (2025)

Niclas Alexander Göring, Charles London, Abdurrahman Hadi Erturk, Chris Mingard, Yoonsoo Nam, Ard A. Louis. **Feature Learning Is Decoupled from Generalization in High Capacity Neural Networks.** arXiv (2025).

- [arXiv:2507.19680](https://arxiv.org/abs/2507.19680)
- **Role here:** Directly motivated E05’s separation between feature-learning strength and generalization benefit.

### R12 · Kornblith et al. (2019)

Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton. **Similarity of Neural Network Representations Revisited.** *ICML 2019*, PMLR 97:3519–3529.

- [PMLR](https://proceedings.mlr.press/v97/kornblith19a.html)
- **Role here:** Source of centered kernel alignment (CKA), used in E05 as a representation-similarity instrument rather than as a direct measure of compression or usefulness.

## 3. Algorithmic information, MDL, PAC-Bayes, and free energy

### R13 · Solomonoff (1964)

Ray J. Solomonoff. **A Formal Theory of Inductive Inference, Part I and Part II.** *Information and Control* 7 (1964).

- [ML Anthology](https://mlanthology.org/misc/1964/solomonoff1964misc-formal/)
- **Role here:** An idealized algorithmic-probability reference for preferring short programs under finite evidence. This archive does not claim that finite neural networks or SGD implement the Solomonoff mixture.

### R14 · Kolmogorov (1965)

A. N. Kolmogorov. **Three Approaches to the Quantitative Definition of Information.** *Problems of Information Transmission* 1(1):1–7 (1965; Russian original 3–11).

- [MathNet](https://www.mathnet.ru/eng/ppi68)
- **Role here:** Background for reference-machine-dependent description complexity and the invariance theorem. Neural K is explicitly protocol-relative.

### R15 · Grünwald (2004)

Peter Grünwald. **A Tutorial Introduction to the Minimum Description Length Principle.** arXiv (2004).

- [arXiv:math/0406077](https://arxiv.org/abs/math/0406077)
- **Role here:** Distinguishes two-part coding, NML, prequential coding, and Bayesian methods, preventing “SGD minimizes program length” from being used as an unsupported slogan.

### R16 · Blier and Ollivier (2018)

Léonard Blier, Yann Ollivier. **The Description Length of Deep Learning Models.** *NeurIPS 2018*.

- [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3b712de48137572f3849aabd5666a4e3-Abstract.html)
- **Role here:** Empirically measures variational and prequential codelengths in deep networks; an important methodological comparison for E22.

### R17 · McAllester (1999)

David A. McAllester. **PAC-Bayesian Model Averaging.** *COLT 1999*.

- [ACM DOI](https://doi.org/10.1145/307400.307435)
- **Role here:** Background for connecting prior-to-posterior complexity with generalization bounds. This archive does not derive loss-profile rankings from PAC-Bayes.

### R18 · Levin, Tishby, and Solla (1989)

Esther Levin, Naftali Tishby, Sara A. Solla. **A Statistical Approach to Learning and Generalization in Layered Neural Networks.** *COLT 1989*, pp. 245–260. DOI: 10.1016/B978-0-08-094829-4.50020-9.

- [ML Anthology](https://mlanthology.org/colt/1989/levin1989colt-statistical/)
- **Role here:** An early fixed-architecture Gibbs-ensemble treatment connecting prediction, free energy, and predictive MDL. “Neural-network free energy” is not claimed as a novelty here.

## 4. Flatness, local entropy, and singular learning theory

### R19 · Hochreiter and Schmidhuber (1997)

Sepp Hochreiter, Jürgen Schmidhuber. **Flat Minima.** *Neural Computation* 9(1):1–42 (1997). DOI: 10.1162/neco.1997.9.1.1.

- [DOI / MIT Press](https://doi.org/10.1162/neco.1997.9.1.1)
- **Role here:** Connects wide low-error parameter regions with MDL and generalization. Parameter-space flatness is coordinate-dependent and is not treated as complete-function volume.

### R20 · Chaudhari et al. (2017)

Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, Riccardo Zecchina. **Entropy-SGD: Biasing Gradient Descent Into Wide Valleys.** *ICLR 2017*.

- [arXiv:1611.01838](https://arxiv.org/abs/1611.01838)
- **Role here:** Uses local entropy to favor wide valleys. It is adjacent to static parameter mass but uses a different object and optimization protocol.

### R21 · Watanabe (2009)

Sumio Watanabe. **Algebraic Geometry and Statistical Learning Theory.** Cambridge University Press (2009). DOI: 10.1017/CBO9780511800474.

- [Cambridge DOI](https://doi.org/10.1017/CBO9780511800474)
- **Role here:** Foundational source for Singular Learning Theory and RLCT; used as a comparison for singular evidence and free-energy asymptotics, not as function Kolmogorov complexity.

### R22 · Lau et al. (2025)

Edmund Lau, Zach Furman, George Wang, Daniel Murfet, Susan Wei. **The Local Learning Coefficient: A Singularity-Aware Complexity Measure.** *AISTATS 2025*, PMLR 258:244–252.

- [PMLR](https://proceedings.mlr.press/v258/lau25a.html)
- **Role here:** Provides a scalable estimator of local learning coefficients, a direct future comparison for Neural K-profile slopes, RLCT/LLC, and margin cores.

## 5. Active learning

### R23 · Seung, Opper, and Sompolinsky (1992)

H. Sebastian Seung, Manfred Opper, Haim Sompolinsky. **Query by Committee.** *COLT 1992*. DOI: 10.1145/130385.130417.

- [ACM DOI](https://doi.org/10.1145/130385.130417)
- **Role here:** E21’s use of multi-seed disagreement to select informative unseen inputs is a neural-function-ensemble version of version-space committee querying.

## 6. Modern finite-width statistical mechanics

### R24 · Pacelli et al. (2023)

R. Pacelli, S. Ariosto, M. Pastore, F. Ginelli, M. Gherardi, P. Rotondo. **A Statistical Mechanics Framework for Bayesian Deep Neural Networks Beyond the Infinite-Width Limit.** *Nature Machine Intelligence* 5:1497–1507 (2023).

- [Nature Machine Intelligence](https://www.nature.com/articles/s42256-023-00767-6)
- **Role here:** Derives finite-width Bayesian corrections beyond the NNGP/infinite-width limit. It shows what modern quantitative theory can solve under explicit assumptions, but does not replace empirical measurement of ordinary optimizers and complete-function distributions.

## 7. AGI, free energy, and cognitive theories

### R25 · Hutter (2007)

Marcus Hutter. **Universal Algorithmic Intelligence: A Mathematical Top-Down Approach.** In *Artificial General Intelligence*, Springer, pp. 227–290 (2007).

- [arXiv:cs/0701125](https://arxiv.org/abs/cs/0701125)
- **Role here:** AIXI combines Solomonoff induction with sequential action and reward maximization; it clarifies why the present framework is an induction theory rather than a complete agent loop.

### R26 · Schmidhuber (2008)

Jürgen Schmidhuber. **Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes.** arXiv preprint (2008).

- [Author-hosted original PDF](https://people.idsia.ch/~juergen/driven2008.pdf)
- **Role here:** Defines curiosity reward through improvement in compression rather than static compressibility; adjacent to per-example free-energy increments and active disagreement querying.

### R27 · Friston and Kiebel (2009)

Karl Friston, Stefan Kiebel. **Predictive Coding under the Free-Energy Principle.** *Philosophical Transactions of the Royal Society B* 364:1211–1221 (2009). DOI: 10.1098/rstb.2008.0300.

- [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/)
- **Role here:** A representative entry for predictive coding and the Free Energy Principle. It shares variational-free-energy mathematics but uses different hidden-state, perception, and action objects.

### R28 · Catoni (2007)

Olivier Catoni. **PAC-Bayesian Supervised Classification: The Thermodynamics of Statistical Learning.** IMS Lecture Notes–Monograph Series 56 (2007).

- [arXiv:0712.0248](https://arxiv.org/abs/0712.0248)
- **Role here:** Connects empirical risk, Gibbs posteriors, temperature, and KL complexity. This project adds complete-function resolution but does not derive profiles automatically from PAC-Bayes.

### R29 · Watanabe (2013)

Sumio Watanabe. **A Widely Applicable Bayesian Information Criterion.** *Journal of Machine Learning Research* 14:867–897 (2013).

- [JMLR PDF](https://jmlr.csail.mit.edu/papers/volume14/watanabe13a/watanabe13a.pdf)
- **Role here:** WBIC/RLCT provides a computable entry to free-energy asymptotics in singular models and may help derive low-loss contraction exponents.

### R30 · Seung, Sompolinsky, and Tishby (1992)

H. Sebastian Seung, Haim Sompolinsky, Naftali Tishby. **Statistical Mechanics of Learning from Examples.** *Physical Review A* 45:6056–6091 (1992). DOI: 10.1103/PhysRevA.45.6056.

- [Physical Review A](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.6056)
- **Role here:** A classical teacher–student statistical mechanics of learning, establishing that temperature-, sample-, and generalization-transition analyses long predate this project.

### R31 · Voita and Titov (2020)

Elena Voita, Ivan Titov. **Information-Theoretic Probing with Minimum Description Length.** *EMNLP 2020*.

- [ACL Anthology](https://aclanthology.org/2020.emnlp-main.14/)
- **Role here:** Replaces single-point probe accuracy with online codelength; adjacent to future hidden-layer conditional-free-energy measurements.

### R32 · Tishby, Pereira, and Bialek (2000)

Naftali Tishby, Fernando C. Pereira, William Bialek. **The Information Bottleneck Method.** Allerton Conference / arXiv (2000).

- [arXiv:physics/0004057](https://arxiv.org/abs/physics/0004057)
- **Role here:** Defines a mutual-information tradeoff between compressing input information and retaining target-relevant information. Its object differs from complete-function parameter volume.

### R33 · Saxe et al. (2018)

Andrew M. Saxe et al. **On the Information Bottleneck Theory of Deep Learning.** *ICLR 2018*.

- [OpenReview](https://openreview.net/forum?id=ry_WPG-A-)
- **Role here:** Shows that a representational “compression phase” depends on activation and measurement conditions rather than being universal across deep-network training.

### R34 · Kolchinsky, Tracey, and Van Kuyk (2018)

Artemy Kolchinsky, Brendan D. Tracey, Steven Van Kuyk. **Caveats for Information Bottleneck in Deterministic Scenarios.** arXiv (2018).

- [arXiv:1808.07593](https://arxiv.org/abs/1808.07593)
- **Role here:** Identifies degeneracies of Information Bottleneck objectives under deterministic continuous mappings and labels, a direct boundary for exact-rule tasks.
