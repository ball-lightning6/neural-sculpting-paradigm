# Rule Preference Phase Transition

## Project Overview

This project designs an elegant "rule duel" experiment to investigate the internal decision-making mechanisms of neural networks when facing **ambiguous data**. The experiment reveals that neural networks exhibit an implicit "simplicity bias," and this preference demonstrates a **phase transition phenomenon**—as the complexity gap between competing rules changes, the model's preference behavior undergoes a qualitative shift.

### Experimental Design

#### Core Idea: Constructing "Indistinguishable" Training Data

We design two rules that produce identical outputs on the training set, making it impossible for the neural network to determine which rule to learn from the data alone:

**Input Construction**: 30-bit input = 15-bit front + 15-bit back

**Rule A (CA)**: `output = CA(front)` — Multi-layer evolution of Cellular Automaton Rule 110
**Rule B (XOR)**: `output = front ^ back` — Bitwise XOR of front and back halves

**Training Set Constraint**: `front ^ back = CA(front)`

By reverse-engineering the back half: `back = front ^ CA(front)`, we ensure both rules produce identical outputs on the training set.

#### Test Set Design: Breaking Constraints to Reveal Preferences

In the test set, `front` and `back` are generated independently and randomly, no longer satisfying the training set constraint. At this point:
- Rule A prediction: `CA(front)`
- Rule B prediction: `front ^ back`
- The two are no longer equal, creating "rule competition"

By observing which rule the model output aligns with, we can determine which interpretation the neural network "prefers."

### Key Finding: Phase Transition

By adjusting the number of CA layers (`CA_LAYERS`), we observe a clear **phase transition process**:

| CA Layers | XOR Accuracy | CA Accuracy | State Description |
|-----------|--------------|-------------|-------------------|
| **1 Layer** | ≈0% | ≈0% | **Superposition State** — The model does not "choose" any single rule, but maintains an "entangled state" that depends on both rules simultaneously |
| **2 Layers** | ≈5-6% | ≈0% | **Beginning Collapse** — Slight preference for the simpler rule (XOR), but not yet fully separated |
| **3 Layers** | **100%** | ≈0% | **Complete Collapse** — Clear selection of XOR, rule preference fully established |

### Key Insights

**1. Conditional Nature of Simplicity Bias**

The neural network's "Occam's Razor" does not always activate. Only when the complexity gap between competing rules is sufficiently large does the model exhibit preference for the simpler rule. When complexities are similar, the model does not make a choice, but maintains a "mixed computation" state.

**2. Counterintuitive "Superposition State"**

In the 1-layer CA experiment, the most counterintuitive finding is: **The model's OOD test accuracy for both rules approaches 0%**. This indicates that the neural network neither learns CA nor XOR, but instead learns a way of "using both information sources simultaneously" without following any single rule.

**3. Deceptiveness of Training Process**

Although the test shows rule preference (or lack thereof), the loss decrease during training remains normal. This demonstrates that the model can perfectly fit training data, but its internal representation may be "oscillating" or "mixing" between the two interpretations.

### Theoretical Implications

**Insights on Neural Network Inductive Bias**

This experiment explores how "simplicity bias" manifests in neural networks. The results show that this bias is not absolute, but has the following characteristics:
- **Gradual**: Strengthens as complexity gap increases
- **Conditional**: Requires sufficient "evidence accumulation" to trigger
- **May lead to generalization failure**: In the "superposition state," model generalization to new distributions may be extremely poor

**Connection to Bayesian Occam's Razor**

The experimental results show similarities to "simplicity priors" in the Bayesian framework, but reveal the specific mechanism by which neural networks implement this: through gradient descent dynamics, gradually "collapsing" to solutions that are easier to fit in parameter space.

### 1. rule_preference_phase_transition.py

- **Purpose:** Main program for rule preference phase transition experiment
- **Logic:**
    1. **Data Generation**: Construct training set satisfying `front ^ back = CA(front)` constraint
    2. **Model Training**: Use MLP to fit data, monitor training/validation loss
    3. **OOD Testing**: On out-of-distribution data, test adherence to Rule A and Rule B separately
    4. **Visualization**: Generate loss dynamics plots and accuracy competition plots
- **Key Parameters:**

```python
CA_LAYERS = 3  # Key variable: modify here (1, 2, 3) to observe phase transition
INPUT_DIM = 30   # Input dimension (15 bit front + 15 bit back)
CORE_DIM = 15    # Output dimension
HIDDEN_SIZE = 1024  # Model capacity
EPOCHS = 1000    # Training epochs
```

### Experimental Result Files

- `duel_layers_1_fixed.png` — Experimental results for CA=1 layer
- `duel_layers_2_fixed.png` — Experimental results for CA=2 layers
- `duel_layers_3_fixed.png` — Experimental results for CA=3 layers

Each figure contains:
- **Left**: Loss dynamics (training loss, in-distribution validation loss, OOD test loss vs Rule A/B)
- **Right**: Accuracy competition (in-distribution validation accuracy, OOD test adherence to Rule A/B)

---

**Last Updated**: 2026-01-29
