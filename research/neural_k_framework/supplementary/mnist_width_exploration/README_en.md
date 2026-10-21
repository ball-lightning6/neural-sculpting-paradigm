# MNIST width exploration: a literature-inspired phenomenon-reproduction attempt

This supplement preserves the existing MNIST/MLP training script for inspecting
the limited training comparison in the double-descent account. It is neither a
new Neural K measurement nor a completed validation of the double-descent
mechanism and receives no separate formal experiment number. No new training was
run for this release, and the missing condition was not filled in.

## 1. Relation to the literature

The script comments refer to the setting family of
[Nakkiran et al., Deep Double Descent](https://arxiv.org/abs/1912.02292), but use
MNIST, a two-hidden-layer ReLU MLP, and 300 epochs. This is a reproduction attempt
with a different task and architecture, not a configuration-by-configuration
reproduction of the original paper.

[Stephenson and Lee (2021)](https://arxiv.org/abs/2108.12006) also study epochwise
double descent. Section 5 of their paper uses CIFAR10, ResNet18, label noise from
0 to 10%, and 10 random seeds, and explicitly excludes CIFAR100. The local script
uses a different task and architecture and is not labeled as a faithful
reproduction of that paper.

## 2. Status of saved results

The existing record contains 13 conditions completed to 300 epochs out of a plan
of seven widths and two seeds; W4096 seed1 is missing. Every saved endpoint still
has nonzero training error. The two-seed mean test error falls from 16.140% at
width 64 to 8.145% at width 2048. W4096 has only one seed, with 8.420% test error.
These data alone establish neither a complete capacity-wise double descent nor
a second descent after strict interpolation.

The current script defaults to attempting uniform relabeling on 15% of examples
and allows drawing the true class again, so that proportion is not the realized
mislabeling rate. The original JSONL lacks a complete run-configuration snapshot;
current defaults alone cannot establish that historical environment variables
were unchanged. Agreement, gzip length, logit sensitivity, and weight norm are
proxies, not Neural K.

## 3. Reproduction entry point

Configuration is at the top of the script, without argparse. GPU-enabled PyTorch,
NumPy, and Matplotlib are required. Before running from this directory, set
NSP_MNIST_ROOT to a data root containing the four uncompressed IDX files under
MNIST/raw: train-images-idx3-ubyte, train-labels-idx1-ubyte,
t10k-images-idx3-ubyte, and t10k-labels-idx1-ubyte. The default root is the local
data directory. NSP_NOISE_RATE controls the relabeling proportion; NSP_DD_OUT
distinguishes output-directory suffixes.

Running the [existing training script](experiment_mnist_double_descent.py)
performs long training; it does not merely display historical results. First
check device availability, data provenance, and budget. This release only makes
the data path configurable and explicitly refuses a long CPU fallback when CUDA
is unavailable. It changes neither the network, optimizer, random seeds, nor
numerical training core. The original script, weights, and prediction records
remain local; results reproducible from the script are not uploaded with this
supplement.
