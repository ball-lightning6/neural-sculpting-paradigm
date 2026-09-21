# References and Their Citation Relationships to This Site

> This page only includes methodological and theoretical work that the current website explicitly mentions or directly relies on. For paper links, journal, conference, PMLR, JMLR, OpenReview, ML Anthology, or arXiv primary pages are preferred.

## 1. Function priors, parameter-to-function maps, and simplicity bias

### R01 · Dingle, Camargo, and Louis (2018)

Kamaludin Dingle, Chico Q. Camargo, Ard A. Louis. **Input–Output Maps Are Strongly Biased Towards Simple Outputs.** *Nature Communications* 9, 761 (2018). DOI: 10.1038/s41467-018-03101-6.

- [Nature Communications original article](https://www.nature.com/articles/s41467-018-03101-6)
- **Role on this site:** The earlier history of the bias toward simple outputs in general input-output maps; it also serves as a reminder that this relationship mainly gives an upper bound on probability, and a simple output is not guaranteed to have high probability.

### R02 · Valle-Pérez, Camargo, and Louis (2019)

Guillermo Valle-Pérez, Chico Q. Camargo, Ard A. Louis. **Deep Learning Generalizes Because the Parameter-Function Map Is Biased Towards Simple Functions.** *ICLR 2019*.

- [arXiv:1805.08522](https://arxiv.org/abs/1805.08522)
- **Role on this site:** The direct predecessor establishing that the neural-network parameter-to-function map produces a nonuniform function prior; also an important source for the function-space PAC-Bayes account.

### R03 · Mingard et al. (2019)

Chris Mingard, Joar Skalse, Guillermo Valle-Pérez, David Martínez-Rubio, Vladimir Mikulik, Ard A. Louis. **Neural Networks Are a Priori Biased Towards Boolean Functions with Low Entropy.** arXiv (2019).

- [arXiv:1909.11522](https://arxiv.org/abs/1909.11522)
- **Role on this site:** Shows the low-entropy bias of the Boolean-function prior, and continues to observe the structural-complexity preference after fixing the output entropy.

### R04 · Mingard et al. (2021)

Chris Mingard, Guillermo Valle-Pérez, Joar Skalse, Ard A. Louis. **Is SGD a Bayesian Sampler? Well, Almost.** *Journal of Machine Learning Research* 22(79):1–64 (2021).

- [JMLR official page](https://www.jmlr.org/papers/v22/20-676.html)
- **Role on this site:** The most direct predecessor of this site's multi-seed function distribution; it supports the first-order predictive power of a static Bayesian function posterior, while explicitly retaining the second-order deviations caused by the optimizer and hyperparameters.

### R05 · Mingard et al. (2025)

Chris Mingard, Henry Rees, Guillermo Valle-Pérez, Ard A. Louis. **Deep Neural Networks Have an Inbuilt Occam’s Razor.** *Nature Communications* 16, 220 (2025).

- [Nature Communications original article](https://www.nature.com/articles/s41467-024-54813-x)
- **Role on this site:** E04 reuses its core 7-bit Boolean, deep tanh, and advSGD protocol, and extends the observation time after the first zero training error.

### R06 · Mingard et al. (2025)

Chris Mingard, Lukas Seier, Niclas Göring, Andrei-Vlad Badelita, Charles London, Ard A. Louis. **Characterising the Inductive Biases of Neural Networks on Boolean Data.** arXiv (2025).

- [arXiv:2505.24060](https://arxiv.org/abs/2505.24060)
- **Role on this site:** The recent Boolean-function work closest to the idea that "an architecture is a programming language"; its correspondence between discrete networks and DNF provides a direct comparison for the architecture relativity of Neural K.

## 2. Grokking, kernel limits, and feature learning

### R07 · Power et al. (2022)

Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra. **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.** arXiv (2022).

- [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- **Role on this site:** Establishes the original phenomenon of "memorization first, delayed generalization later" on small algorithmic datasets, together with the dataset-size axis.

### R08 · Nanda et al. (2023)

Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, Jacob Steinhardt. **Progress Measures for Grokking via Mechanistic Interpretability.** arXiv (2023).

- [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- **Role on this site:** Decomposes modular-addition grokking into a continuous process of memorization, Fourier-circuit formation, and cleanup; it is an important comparison for this site's point that "an apparently abrupt surface transition can have continuous internal coordinates".

### R09 · Jacot, Gabriel, and Hongler (2018)

Arthur Jacot, Franck Gabriel, Clément Hongler. **Neural Tangent Kernel: Convergence and Generalization in Neural Networks.** *NeurIPS 2018*.

- [arXiv:1806.07572](https://arxiv.org/abs/1806.07572)
- **Role on this site:** The analytic infinite-width fixed-kernel baseline for E05; used to judge whether a real finite network still reorganizes its features when a fixed kernel is already sufficient.

### R10 · Soudry et al. (2018)

Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, Nathan Srebro. **The Implicit Bias of Gradient Descent on Separable Data.** *Journal of Machine Learning Research* 19(70):1–57 (2018).

- [JMLR official page](https://www.jmlr.org/papers/v19/18-188.html)
- **Role on this site:** Shows that after zero classification error, logistic/cross-entropy still drives the margin direction; it is the narrow mechanism that must be distinguished from "post-fit loss continuing to decrease".

### R11 · Göring et al. (2025)

Niclas Alexander Göring, Charles London, Abdurrahman Hadi Erturk, Chris Mingard, Yoonsoo Nam, Ard A. Louis. **Feature Learning Is Decoupled from Generalization in High Capacity Neural Networks.** arXiv (2025).

- [arXiv:2507.19680](https://arxiv.org/abs/2507.19680)
- **Role on this site:** Directly motivated E05's separation of feature-learning strength from generalization benefit; it is a reminder that a large change in representation does not mean the change is useful for the target.

### R12 · Kornblith et al. (2019)

Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton. **Similarity of Neural Network Representations Revisited.** *ICML 2019*, PMLR 97:3519–3529.

- [PMLR official page](https://proceedings.mlr.press/v97/kornblith19a.html)
- **Role on this site:** E05 uses centered kernel alignment (CKA) to compare hidden representations with the empirical NTK; CKA is only a similarity instrument and does not automatically amount to compression or task relevance.

## 3. Algorithmic information, MDL, PAC-Bayes, and free energy

### R13 · Solomonoff (1964)

Ray J. Solomonoff. **A Formal Theory of Inductive Inference, Part I and Part II.** *Information and Control* 7 (1964).

- [ML Anthology original entry](https://mlanthology.org/misc/1964/solomonoff1964misc-formal/)
- **Role on this site:** Provides an idealized algorithmic-probability reference for "preferring short programs under finite evidence"; this site does not claim that finite neural networks or SGD implement a Solomonoff mixture.

### R14 · Kolmogorov (1965)

A. N. Kolmogorov. **Three Approaches to the Quantitative Definition of Information.** *Problems of Information Transmission* 1(1):1–7 (1965; Russian original 3–11).

- [MathNet original page](https://www.mathnet.ru/eng/ppi68)
- **Role on this site:** Provides the background of reference-machine-relative description complexity and the invariance theorem; Neural K is only called a protocol-relative candidate quantity.

### R15 · Grünwald (2004)

Peter Grünwald. **A Tutorial Introduction to the Minimum Description Length Principle.** arXiv (2004).

- [arXiv:math/0406077](https://arxiv.org/abs/math/0406077)
- **Role on this site:** Used to distinguish two-part codes, NML, prequential coding, and Bayesian methods, avoiding turning MDL into the slogan that "SGD automatically minimizes program length".

### R16 · Blier and Ollivier (2018)

Léonard Blier, Yann Ollivier. **The Description Length of Deep Learning Models.** *NeurIPS 2018*.

- [NeurIPS official page](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3b712de48137572f3849aabd5666a4e3-Abstract.html)
- **Role on this site:** Empirically measures the description length of deep networks with variational and prequential codes; an important methodological reference when E22 connects per-example surprise with endpoint cost.

### R17 · McAllester (1999)

David A. McAllester. **PAC-Bayesian Model Averaging.** *COLT 1999*.

- [ACM DOI page](https://doi.org/10.1145/307400.307435)
- **Role on this site:** The function-prior line of work uses PAC-Bayes to connect prior/posterior mass with generalization bounds; this site does not derive loss-profile rankings from those bounds.

### R18 · Levin, Tishby, and Solla (1989)

Esther Levin, Naftali Tishby, Sara A. Solla. **A Statistical Approach to Learning and Generalization in Layered Neural Networks.** *COLT 1989*, pp. 245–260. DOI: 10.1016/B978-0-08-094829-4.50020-9.

- [ML Anthology official entry and original text](https://mlanthology.org/colt/1989/levin1989colt-statistical/)
- **Role on this site:** Long ago wrote fixed-architecture networks as a Gibbs ensemble and connected prediction, free energy, and predictive MDL; this site does not claim "neural-network free energy" as a first.

## 4. Flatness, local entropy, and singular learning theory

### R19 · Hochreiter and Schmidhuber (1997)

Sepp Hochreiter, Jürgen Schmidhuber. **Flat Minima.** *Neural Computation* 9(1):1–42 (1997). DOI: 10.1162/neco.1997.9.1.1.

- [DOI / MIT Press page](https://doi.org/10.1162/neco.1997.9.1.1)
- **Role on this site:** Connects wide low-error weight regions with MDL/generalization; this site emphasizes that parameter flatness is coordinate-dependent and cannot directly substitute for function volume.

### R20 · Chaudhari et al. (2017)

Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, Riccardo Zecchina. **Entropy-SGD: Biasing Gradient Descent Into Wide Valleys.** *ICLR 2017*.

- [arXiv:1611.01838](https://arxiv.org/abs/1611.01838)
- **Role on this site:** Uses local entropy to favor wide valleys; adjacent to this site's static parameter mass, but the measured object and the optimization protocol are different.

### R21 · Watanabe (2009)

Sumio Watanabe. **Algebraic Geometry and Statistical Learning Theory.** Cambridge University Press (2009). DOI: 10.1017/CBO9780511800474.

- [Cambridge DOI page](https://doi.org/10.1017/CBO9780511800474)
- **Role on this site:** The foundational source for Singular Learning Theory / RLCT; used to compare singular-model evidence and free-energy asymptotics, and is not equated with function Kolmogorov complexity.

### R22 · Lau et al. (2025)

Edmund Lau, Zach Furman, George Wang, Daniel Murfet, Susan Wei. **The Local Learning Coefficient: A Singularity-Aware Complexity Measure.** *AISTATS 2025*, PMLR 258:244–252.

- [PMLR official page](https://proceedings.mlr.press/v258/lau25a.html)
- **Role on this site:** Provides a scalably estimable local learning coefficient; a direct methodological neighbor for future comparisons of Neural K-profile local slopes, RLCT/LLC, and margin cores.

## 5. Classical sources of active learning

### R23 · Seung, Opper, and Sompolinsky (1992)

H. Sebastian Seung, Manfred Opper, Haim Sompolinsky. **Query by Committee.** *COLT 1992*. DOI: 10.1145/130385.130417.

- [ACM DOI page](https://doi.org/10.1145/130385.130417)
- **Role on this site:** E21 uses multi-seed disagreement to select informative unseen inputs; it is a neural function-distribution version of the version-space committee querying idea.

## 6. Modern statistical mechanics of finite-width neural networks

### R24 · Pacelli et al. (2023)

R. Pacelli, S. Ariosto, M. Pastore, F. Ginelli, M. Gherardi, P. Rotondo. **A Statistical Mechanics Framework for Bayesian Deep Neural Networks Beyond the Infinite-Width Limit.** *Nature Machine Intelligence* 5 (2023).

- [Nature Machine Intelligence original article](https://www.nature.com/articles/s42256-023-00767-6)
- **Role on this site:** Gives statistical-mechanics corrections for finite-width Bayesian deep networks beyond the NNGP/infinite-width limit; used to show that modern quantitative theory can solve particular models analytically, but it cannot automatically replace this site's experimental measurements of ordinary optimizers and complete function distributions.

## 7. AGI, free energy, and cognitive theories

### R25 · Hutter (2007)

Marcus Hutter. **Universal Algorithmic Intelligence: A Mathematical Top-Down Approach.** In *Artificial General Intelligence*, Springer, pp. 227–290 (2007).

- [arXiv:cs/0701125](https://arxiv.org/abs/cs/0701125)
- **Role on this site:** AIXI combines Solomonoff universal induction with sequential action and reward maximization; used to show that this site currently has only the induction layer, not a complete agent loop.

### R26 · Schmidhuber (2008)

Jürgen Schmidhuber. **Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes.** arXiv preprint (2008).

- [Author-provided original PDF](https://people.idsia.ch/~juergen/driven2008.pdf)
- **Role on this site:** Defines the curiosity reward as progress in compression ability rather than static compressibility; adjacent to this site's per-example free-energy increments and active disagreement queries.

### R27 · Friston and Kiebel (2009)

Karl Friston, Stefan Kiebel. **Predictive Coding under the Free-Energy Principle.** *Philosophical Transactions of the Royal Society B* 364:1211–1221 (2009). DOI: 10.1098/rstb.2008.0300.

- [PubMed Central original article](https://pmc.ncbi.nlm.nih.gov/articles/PMC2666703/)
- **Role on this site:** A representative entry point for the Free Energy Principle / predictive coding; it shares the variational free-energy mathematics with this site, but the hidden variables, perception, and action objects differ.

### R28 · Catoni (2007)

Olivier Catoni. **PAC-Bayesian Supervised Classification: The Thermodynamics of Statistical Learning.** IMS Lecture Notes–Monograph Series 56 (2007).

- [arXiv:0712.0248](https://arxiv.org/abs/0712.0248)
- **Role on this site:** Connects empirical risk and generalization through the Gibbs posterior, temperature, and KL complexity; this site further carries out a complete function-level decomposition, but does not automatically derive the profile from PAC-Bayes.

### R29 · Watanabe (2013)

Sumio Watanabe. **A Widely Applicable Bayesian Information Criterion.** *Journal of Machine Learning Research* 14:867–897 (2013).

- [JMLR original PDF](https://jmlr.csail.mit.edu/papers/volume14/watanabe13a/watanabe13a.pdf)
- **Role on this site:** WBIC/RLCT is a computable entry to the free-energy asymptotics of singular models; it may be used in the future to derive low-loss volume contraction exponents.

### R30 · Seung, Sompolinsky, and Tishby (1992)

H. Sebastian Seung, Haim Sompolinsky, Naftali Tishby. **Statistical Mechanics of Learning from Examples.** *Physical Review A* 45:6056–6091 (1992). DOI: 10.1103/PhysRevA.45.6056.

- [Physical Review A original page](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.45.6056)
- **Role on this site:** The classical teacher–student statistical mechanics of learning; used to delimit that rigorous precedents for "temperature, sample number, and generalization phase transitions" have long existed.

### R31 · Voita and Titov (2020)

Elena Voita, Ivan Titov. **Information-Theoretic Probing with Minimum Description Length.** *EMNLP 2020*.

- [ACL Anthology original article](https://aclanthology.org/2020.emnlp-main.14/)
- **Role on this site:** Replaces single-point probe accuracy with online codelength; adjacent to the future line of measuring conditional free energy / computational pressure by hidden layer.

### R32 · Tishby, Pereira, and Bialek (2000)

Naftali Tishby, Fernando C. Pereira, William Bialek. **The Information Bottleneck Method.** Allerton Conference / arXiv (2000).

- [arXiv:physics/0004057](https://arxiv.org/abs/physics/0004057)
- **Role on this site:** Defines the mutual-information tradeoff between compressing input information and retaining target-relevant information; the object being compressed differs from complete-function parameter volume.

### R33 · Saxe et al. (2018)

Andrew M. Saxe et al. **On the Information Bottleneck Theory of Deep Learning.** *ICLR 2018*.

- [OpenReview original](https://openreview.net/forum?id=ry_WPG-A-)
- **Role on this site:** Shows that the so-called layer-by-layer "compression phase" depends on the activation and measurement conditions and is not a universal phenomenon of all deep-network training.

### R34 · Kolchinsky, Tracey, and Van Kuyk (2018)

Artemy Kolchinsky, Brendan D. Tracey, Steven Van Kuyk. **Caveats for Information Bottleneck in Deterministic Scenarios.** arXiv (2018).

- [arXiv:1808.07593](https://arxiv.org/abs/1808.07593)
- **Role on this site:** Points out that the Information Bottleneck can degenerate under deterministic continuous mappings and deterministic labels; the exact-rule tasks of this project need precisely this boundary.

## 8. Bayesian surprise and active information gain

### R35 · Itti and Baldi (2009)

Laurent Itti, Pierre Baldi. **Bayesian Surprise Attracts Human Attention.** *Vision Research* 49(10):1295–1306 (2009). DOI: 10.1016/j.visres.2008.09.007.

- [PubMed Central original article](https://pmc.ncbi.nlm.nih.gov/articles/PMC2782645/)
- **Role on this site:** Defines Bayesian surprise through the KL between the model distributions before and after an update; Section 6 of the theory core specializes it to the complete hard-function distribution and proves that the realized label surprisal equals the hard-posterior information gain.

### R36 · Houlsby et al. (2011)

Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, Máté Lengyel. **Bayesian Active Learning for Classification and Preference Learning.** arXiv (2011).

- [arXiv:1112.5745](https://arxiv.org/abs/1112.5745)
- **Role on this site:** BALD selects queries by the mutual information between labels and parameters/hypotheses; in the deterministic-function limit, this site connects it to predictive entropy, low agreement, and E21's disagreement-based sample selection.

## 9. Supplement on the grokking entropy landscape

### R37 · Zhang et al. (2025)

Xiaotian Zhang, Yue Shang, Entao Yang, Ge Zhang. **Is Grokking a Computational Glass Relaxation?** *NeurIPS 2025*.

- [NeurIPS official paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/92f67b9047fa7a43d7506054b5f0ec6a-Paper-Conference.pdf)
- [arXiv:2505.11411](https://arxiv.org/abs/2505.11411)
- **Role on this site:** Uses the Wang--Landau method to map the training-loss--test-accuracy entropy landscape of modular-arithmetic Transformers, and finds high-norm generalizing solutions with WanD without explicit regularization; it supports the external comparison that "grokking does not require weight decay, and the generalizing state can have a static entropy advantage". E30 only clarifies, in a smaller system, how explicit L2 changes the same static function competition.
