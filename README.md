# Mutual Information Estimation via Score-to-Fisher Bridge

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2510.05496-b31b1b.svg)](https://arxiv.org/abs/2510.05496)

This repository contains the implementation code for reproducing the figures in the paper:

**["Mutual Information Estimation via Score-to-Fisher Bridge for Nonlinear Gaussian Noise Channels"](https://arxiv.org/abs/2510.05496)**
*Tadashi Wadayama, Nagoya Institute of Technology*

## Overview

We present a numerical method to evaluate mutual information (MI) in nonlinear Gaussian noise channels by using **Denoising Score Matching (DSM)** learning. The key contributions include:

- **Score-only MI estimation** for arbitrary deterministic nonlinear front-ends
- **Posterior-free approach** that only requires forward channel simulation
- **Fisher integral representation** for accurate MI computation
- **Validation** on Gaussian inputs, BPSK, linear Gaussian channels, and composite nonlinear channels

### Channel Model

We consider additive Gaussian noise channels with a deterministic nonlinear front-end:

```
Y_t = f(X) + Z_t,    Z_t ~ N(0, t*I_n)
```

where `f: R^n -> R^n` is a deterministic function (e.g., tanh, saturation, learned encoder).

### Methodology

The method leverages the **de Bruijn identity** and **I-MMSE relation**:

```
d/dt I(X; Y_t) = 1/2 [J(Y_t) - n/t]
```

where `J(Y_t)` is the Fisher information estimated from the score function learned via DSM.

## Repository Structure

```
score-to-fisher-mi/
├── Gaussian_J/              # Figure 1: Fisher information validation
│   └── dsm_gaussian_multi_n.py
├── Gaussian_MI/             # Figure 2: Mutual information for Gaussian input
│   └── MI.py
├── Discrete/                # Figure 4: BPSK (discrete input) validation
│   └── bpsk.py
├── tanh_linear/             # Figure 6: Composite nonlinear channel
│   └── tanh_linear.py
├── Gaussian_Linear_MI/      # Figure 5: Linear Gaussian channel
│   └── Linear_MI.py
├── Noise_Conditional/       # Figure 3: Noise-conditional model
│   └── Noise_Conditional.py
├── pyproject.toml           # Project dependencies
└── README.md                # This file
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Prerequisites

- Python 3.13+
- uv (install via: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Setup

```bash
# Clone the repository
git clone https://github.com/wadayama/score-to-fisher-mi.git
cd score-to-fisher-mi

# Install dependencies using uv
uv sync
```

## Usage

### Running Experiments

Each directory contains standalone scripts for reproducing specific figures from the paper.

#### Figure 1: Fisher Information Validation (Gaussian Input)

```bash
uv run python Gaussian_J/dsm_gaussian_multi_n.py
```

#### Figure 2: Mutual Information (Gaussian Input)

```bash
uv run python Gaussian_MI/MI.py
```

#### Figure 4: BPSK Input (Discrete Distribution)

```bash
uv run python Discrete/bpsk.py
```

#### Figure 6: Composite Nonlinear Channel

```bash
uv run python tanh_linear/tanh_linear.py
```

#### Figure 5: Linear Gaussian Channel

```bash
uv run python Gaussian_Linear_MI/Linear_MI.py
```

#### Figure 3: Noise-Conditional Model

```bash
uv run python Noise_Conditional/Noise_Conditional.py
```

### Output Files

Each script generates:
- **PDF plots**: Visualization of MI estimates vs. ground truth
- **PT files**: PyTorch saved results for further analysis

## Implementation Details

- **Neural Network**: 3 hidden layers, 128 units, SiLU activation
- **Optimization**: Adam optimizer, lr=1e-3, gradient clipping
- **Integration**: Log-domain trapezoid rule for numerical stability
- **Tail Correction**: Asymptotic behavior for t -> infinity

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wadayama2024mi,
  title={Mutual Information Estimation via Score-to-Fisher Bridge for Nonlinear Gaussian Noise Channels},
  author={Wadayama, Tadashi},
  journal={arXiv preprint arXiv:2510.05496},
  year={2024},
  url={https://arxiv.org/abs/2510.05496}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by JST, CRONOS, Japan Grant Number JPMJCS25N5.

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Tadashi Wadayama (wadayama@nitech.ac.jp)
- Nagoya Institute of Technology
