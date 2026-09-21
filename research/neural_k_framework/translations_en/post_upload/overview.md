# Incremental materials to be uploaded in this batch

## 1. Existing evidence: 4-bit rule curves and rule duels (E10/E19/E23/E24)

This is the foundational evidence of the main package in this batch: function mass curves over the full rule space, complex-function contraction at low loss, rule duels, and changes in the complexity distribution.

## 2. Existing experiments: E15/E27 Agreement around grokking

In the mod97 task, after the hard fit, agreement on unseen inputs remains close to random, and then rises together with target accuracy. This supports the picture that "many different training-set-compatible functions exist before grokking", and does not support the simple picture that "all seeds have long shared a single wrong function".

## 3. Existing experiment: E23 Parity2/MUX3 data transition

Full-target Neural K cannot by itself determine the data transition. MUX3's copy shortcut, selector-conflict coverage, and the fixed-D competition denominator can change `n50/n90`. Natural coverage, exogenous conflict intervention, and static SMC form a closed loop.

## 4. E31 raw MSE and magnitude alignment

raw 0/1 and 0/15 MSE can exhibit grokking; the BCE intervention with output-magnitude alignment can generalize at a higher mean CE. The conclusion is not that BCE is uniquely correct, but that the loss and the target magnitude jointly select the direction of the output joint distribution.

## 5. E32 random memorization

The memorization complexity of random labels grows approximately linearly with the number of samples, providing an external calibration for the complexity/capacity relationship.

## 6. E33 cross-architecture joint output

Rule-dependent joint output volume curves also appear across different architectures, but the coefficients and local ordering depend on the protocol.

## 7. E34 8-bit conditional rule duel

Also part of the rule duel family, and the group with the strongest signal within it: the 8-bit input is split into a front and a back 4-bit block, each fed to one of the two rules; the training set contains only inputs on which the two rules' outputs agree, and the conflict set tests the preference. The full complexity order (parity1<majority3<MUX3<parity2<parity3<parity4) was frozen in advance from the independent 4-bit curves; the complete round robin of 15 pairs × both directions × 5 loss cross-sections = 75 verdicts all hit; the only non-overwhelming verdict is MUX3 vs parity2 = 0.606 (the preference decreases from 0.684 to 0.606 as the loss deepens, without reversing).

## 8. E35 frozen-function duel of real grokking candidates on mod97

The actual full function B from before the original grokking trajectory completed, and the rule A from after completion, were placed into a unified small MLP to re-compete on the duel training set: all 192 models fully fit the training set, about 99.9% of conflict points favor A, and no model ultimately implements B, yet the full-A recovery rate is only about 16% — showing that "the later-arriving rule is preferred" is a dynamic fact at the function population level, not a net-by-net replication of the rule. The near/far tier comparison shows that a larger source-loss gap does not improve the exact recovery rate.

## 9. E36 Peleg/G&C independent verification

This is a standalone external verification experiment, not a literature summary: it reproduces Guess-and-Check under the small-sample MNIST setup of Peleg/Hein, and further tightens the static SMC constraint on raw BCE under the same initialization protocol. When G&C only requires all training classifications to be correct, it achieves about 82.13%; after tightening the BCE constraint, static samples reach a test level of about 96%, close to SGD. It therefore probes the boundary of "whether gradient-free static sampling must stop at a shallow interpolation surface", and does not conflate G&C with this project's function-level Neural K measurement as the same experiment.
