# NTK Batch Solver

## Project Overview

This project implements an analytical solver based on Neural Tangent Kernel (NTK) theory to verify whether deep learning models have truly surpassed the "lazy learning" regime.

### Core Theory

This benchmark computes the analytical solution for a **two-layer infinitely-wide ReLU network**, establishing a "lazy learning accuracy baseline" for deep learning tasks.

#### Why is this the Lazy Learning Upper Bound?

**1. Lazy Learning Essence**: NTK describes the linear approximation of networks near initialization. In this regime, the feature extractors (hidden layers) are **frozen**, and the model can only fit data through linear combinations of initial random features.

**2. Infinite-Width Representation**: According to the Universal Approximation Theorem (Hornik, 1991), a single-layer infinitely-wide network already possesses the capability to approximate any continuous function. Therefore, single-layer NTK already has the "theoretically perfect random feature combination".

**3. Depth and Kernel Degeneration** (Theoretical Rigor):
   - Theoretically, multi-layer networks have different kernel forms in the lazy regime
   - However, research (Hayou et al., 2019) shows that as depth increases, the NTK of ReLU networks gradually undergoes **kernel degeneration**, making the kernel function increasingly smooth and highly correlated
   - For non-smooth logical tasks like cellular automata and XOR that are extremely sensitive to input perturbations, kernel smoothing leads to a **significant decrease** in the ability to capture logical transitions
   - Thus, **single-layer NTK actually represents the "performance peak"** for ReLU kernel functions in maintaining local logical sensitivity

#### Inference Logic

If this "perfect lazy learner" (single-layer NTK) only achieves 1.4% exact match with 30,000 samples, while the real 4-layer network reaches 100%, this suggests:

- The additional 98.6% performance **may be difficult to explain through depth alone**
- It could result from the model escaping fixed kernel constraints and performing active "feature sculpting"
- This provides **empirical support for feature learning (Rich Training Regime)**

### NTK Analytical Formula

This is the **complete mathematical description of lazy learning**, computing the kernel matrix for a two-layer infinitely-wide ReLU network:

**Kernel Functions:**

```
K_σ(x, x') = (||x||·||x'||)/(2π) · [sin(θ) + (π - θ)cos(θ)]
K_dot(x, x') = (x·x')/(2π) · (π - θ)
Θ(x, x') = K_σ + K_dot

θ = arccos[(x·x')/(||x||·||x'||)]
```

**Prediction Equations:**

```
α = (K_train + λI)^(-1) · Y_train
Y_pred = K_test · α
```

In the NTK limit, network behavior is primarily determined by this **fixed kernel function**, with parameter evolution approximately frozen near initialization values. **Under the lazy learning regime, the model's performance may be difficult to easily exceed this analytical solution**. Therefore, this formula's prediction accuracy can be viewed as a **theoretical reference upper bound for lazy learning**.

### 1. ntk_batch_solver.py

- **Purpose:** Batch NTK Analytical Solver
- **Logic:** Computes all output bits at once, 30x efficiency improvement; Accurately statistics Exact Match (all bits correct simultaneously); GPU-accelerated, supports large-scale kernel matrix computation

**Key Parameters:**

```python
DATA_PATH = "ca_rule110_layer3_30.jsonl"  # Dataset path
N_TRAIN = 30000  # Training samples (up to 50k supported)
N_TEST = 1000    # Test samples
LAMBDA = 1e-5    # Regularization (recommended: 1e-5 to 1e-3)
BITS_TO_TEST = None  # None = test all bits
```

## Hardware Requirements

- **GPU**: RTX 3080 10GB or higher recommended
- **VRAM**: N=30,000 requires ~6-7GB VRAM
- **CPU**: Any modern CPU

## Experimental Validation

Experiments on some tasks have shown significant performance gaps between NTK analytical solutions and real training, suggesting that neural networks may surpass lazy learning limits through feature learning. More experimental evidence to be supplemented later.

## Experimental Value

Single-layer NTK can be viewed as a theoretical reference upper bound for lazy learning. If NTK fails but real training succeeds, it provides strong evidence for surpassing lazy learning.

---

**Last Updated**: 2026-01-27
