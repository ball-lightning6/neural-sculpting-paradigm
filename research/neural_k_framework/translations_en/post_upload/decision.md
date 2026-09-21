# Upload decision table

Baseline: the 2026-09-02 short-paper submission, Git `4596b2b`.

Here "upload" is divided into three categories:

- **Main package**: should go into the public repository / the paper's accompanying materials; readers need it to understand the core claims;
- **Supplementary package**: worth publishing, but it should not crowd into the short paper's main line;
- **Archive**: kept locally or backed up separately; does not enter the public upload package.

## 1. Main package: should be uploaded

| Material | Path | Reason |
|---|---|---|
| Revised official short paper | `papers/compression_is_intelligence_{zh,en}.md` and the local v1.1 PDF | The current main narrative, joint output, and static/dynamic boundaries have been corrected |
| Theoretical core | `theory_core_zh.md` and the corresponding English core | States the final claims and the boundaries of Neural K, MDL, agreement, and optimization accessibility |
| Strict probabilistic language | `minimal_strict_statistical_language_tutorial_zh.md` and its audit results | Fixes the full-target/fixed-D, KL-direction, information-gain, and loss-event confusions |
| MDL–grokking bridge | `theory/mdl_identifiability_grokking_derivation_zh.md` | Formalizes the relationship among complexity, competing-function divergence, and the data transition |
| 4-bit rule curves / rule duels | Parent directories E19/E23/E24 and the corresponding result summaries | Core static volume and contraction evidence |
| E15/E27 agreement | `experiments/E15_grokking_agreement` | Shows that after a hard fit, agreement does not equal generalization, and that it co-varies with grokking |
| E23 Parity2/MUX3 | `experiments/E23_volume_to_data_transition` | Shows that full-target Neural K alone cannot determine the data transition; competing continuations and sample coverage are important |
| E31 raw MSE and amplitude alignment | `experiments/E32_loss_amplitude_grokking` | The most important new training intervention: grokking does not require a unique extremely low CE; the joint output shape is critical |
| E32 random-memory near-linearity | `experiments/E33_random_memory` | Provides independent capacity evidence for "complexity grows with the number of memorized samples" |
| E33 cross-architecture joint output | `experiments/E34_cross_architecture` | Shows that the contraction phenomenon is not an isolated case of a single MLP protocol, while preserving the ranking/coefficient boundaries |
| E34 conditional rule duel | `experiments/E37_conditional_rule_duel` | An independent frozen 4-bit complexity total order predicts 75 8-bit duels |
| E35 Mod97 frozen-function duels | `experiments/E38_mod97_frozen_function_duels` | Uses real pre-/post-grokking functions to test the dynamic preference for completed rules |
| Key evidence ledger and references | `evidence_ledger_zh.md`, `references_zh.md`, and the English versions | Makes every number, claim, and external theoretical interface traceable |

The main package should include scripts, protocols, result summaries, and the necessary figures; it need not include checkpoints, particle pools, or raw zips.

## 2. Supplementary package: recommended for upload

| Material | Path | Upload positioning |
|---|---|---|
| Pre-training architecture estimation | `evidence/architecture_bias` | Both the successful shallow region and the failed deep tail are worth publishing; it must not be packaged as a general-purpose predictor |
| E36 Peleg/G&C independent verification | `evidence/peleg_recheck` | An independent verification experiment: external comparison and the boundaries of the volume hypothesis; suitable as supplementary material |
| Verification of strictly defined finite probabilities | `theory/check_minimal_statistical_language.py` and JSON | Low-cost and reproducible; recommended to be uploaded together with the theory appendix |

The supplementary package may keep lightweight JSON, CSV, and PNG files; large raw results should be made into separate downloadable attachments with SHA256.

## 3. Not uploaded for now

| Material | Handling |
|---|---|
| `kimi_experiments` input-level / concat output-level experiments | Stable results already exist (64/64 completed), but it was decided they will not enter the public incremental package; the original scripts and results are kept in the workspace |
| Legacy rule-duel panels (physical directories E31/E36) | Duplicated with the public E34 or not closed; kept in the workspace; they do not represent the public E31/E36 numbering |
| Rule-space OOD, decoupled accessibility, early script audits | Kept in the workspace; not entering the public incremental package |
| addition8 static prediction | Kept in the workspace; not entering the public incremental package |
| AIT/Omega and the independent literature audit of the volume hypothesis | Kept in the workspace; not entering the public incremental package |
| `.tmp_*`, `tmp/`, `_analysis_*` | Serve as workspace provenance caches; most of `_analysis_*` predates this submission, and its formal results have been absorbed into the report |
| `.venv-gpu`, pip caches, GPU temp directories | Environment and caches, not research material |
| Checkpoints, SMC particle pools, `.pt/.npz/.zip/.jsonl` | Packed separately by topic only when someone needs a full recheck; they do not enter the main package |
| `chatbox_*`, chat recovery databases, screenshot caches | Privacy/engineering caches; not uploaded |
| Unexplained engineering logs of failed MSE seeds | Only result summaries and failure status are kept; large volumes of duplicate logs are not uploaded |
| Draft empirical formulas without independent result support | Kept as research notes; not written as paper conclusions |
| RSA and text2image changes unrelated to this research | Left in the original workspace; not included in the Neural K release |

## 4. Recommended upload order

1. First upload the main package: the short paper, the theoretical core, the E01–E30 basic experiments, and this main-package increment;
2. Then upload the supplementary package: E23, E31–E37, and the architecture-estimation report; the old physical E37/E38 directories remain local archive directories and are not the public E37 directory;
3. Finally, subject to platform attachment limits, separately upload the early script audits and the large raw results;
4. E35 and other undecided experiments receive versioned additions only after formal results exist; they should not occupy the main narrative in advance.

## 5. Qualifications that must be preserved when publishing

- Neural K is a protocol-relative implementation-quality code length; it is not equal to machine-independent Kolmogorov complexity;
- "Complex functions contract faster and the complexity distribution narrows" is an empirical trend and a conditional mechanism within finite protocols, not a cross-architecture theorem;
- The static function distribution, loss cross-sections, and the SGD final distribution must be kept separate;
- The grokking data transition, the training-time transition, and Neural K are not the same scalar;
- All failures, local reorderings, sampling degeneracies, and accessibility boundaries must be preserved together with the successful results.
