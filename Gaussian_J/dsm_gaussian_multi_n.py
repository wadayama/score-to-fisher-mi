"""
Fisher information estimation for Gaussian input via DSM - Multiple n values overlay
---------------------------------------------------------------------------------
This script estimates J(Y_t) for Y_t = X + sqrt(t)*eps with X ~ N(0, P I_n)
for multiple dimensions n = 4, 8, 16 and plots them in overlay format.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------
# Configuration
# ------------------------------
@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Dimensions to compare
    n_list: Tuple[int, ...] = (4, 8, 16)

    # Fixed power setting
    P: float = 1.0

    # t grid: geometric (log-spaced)
    M: int = 10  # Number of t points
    t_min_scale: float = 1.0 / 200.0
    t_max_scale: float = 200.0

    # Training per-t
    steps_per_t: int = 300  # Reduced for faster testing
    batch_size: int = 4096  # Reduced batch size
    lr: float = 1e-3
    grad_clip: float = 1.0

    # Evaluation per-t
    eval_samples_per_t: int = 100_000

    # Network (MLP)
    hidden: int = 128
    layers: int = 3

    # Output files
    output_prefix: str = "fisher_multi_n"


# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_t_grid(P: float, M: int, t_min_scale: float, t_max_scale: float) -> torch.Tensor:
    t_min = P * t_min_scale
    t_max = P * t_max_scale
    return torch.logspace(math.log10(t_min), math.log10(t_max), M)


# ------------------------------
# Score network s_θ(y) at fixed t
# ------------------------------
class ScoreNetFixedT(nn.Module):
    """MLP: y -> s(y) in R^n (no t input)."""
    def __init__(self, n: int, hidden: int, layers: int):
        super().__init__()
        dims = [n] + [hidden] * layers + [n]
        net = []
        for i in range(len(dims) - 2):
            net += [nn.Linear(dims[i], dims[i + 1]), nn.SiLU()]
        net += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*net)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


# ------------------------------
# Samplers
# ------------------------------
@torch.no_grad()
def sample_batch_fixed_t(P: float, n: int, batch_size: int, t_value: float, device: str):
    t = torch.full((batch_size,), float(t_value), device=device)
    x = torch.randn(batch_size, n, device=device) * math.sqrt(P)
    eps = torch.randn(batch_size, n, device=device)
    y = x + eps * torch.sqrt(t).unsqueeze(-1)
    target = eps / torch.sqrt(t).unsqueeze(-1)  # DSM target ε/√t
    return y, target


@torch.no_grad()
def sample_y_only(P: float, n: int, num: int, t_value: float, device: str):
    t = torch.full((num,), float(t_value), device=device)
    x = torch.randn(num, n, device=device) * math.sqrt(P)
    eps = torch.randn(num, n, device=device)
    y = x + eps * torch.sqrt(t).unsqueeze(-1)
    return y


# ------------------------------
# Training one model at a fixed t
# ------------------------------

def train_one_fixed_t(cfg: Config, n: int, P: float, t_value: float) -> ScoreNetFixedT:
    device = cfg.device

    # Create the score network model
    model = ScoreNetFixedT(n=n, hidden=cfg.hidden, layers=cfg.layers).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for step in range(1, cfg.steps_per_t + 1):
        model.train()
        y, target = sample_batch_fixed_t(P, n, cfg.batch_size, t_value, device)
        pred = model(y)
        loss = ((pred + target) ** 2).mean()  # DSM loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        opt.step()

        if step % 200 == 0 or step == 1:
            print(f"[n={n} | P={P:.3g} | t={t_value:.3g}] step {step:4d} | loss {loss.item():.6f}")

    return model


# ------------------------------
# Evaluate J_hat(Y_t) and compare to J_true(Y_t)
# ------------------------------
@torch.no_grad()
def estimate_fisher_fixed_t(model: ScoreNetFixedT, n: int, P: float, t_value: float, cfg: Config) -> Tuple[float, float]:
    device = cfg.device
    y = sample_y_only(P, n, cfg.eval_samples_per_t, t_value, device)
    s = model(y)
    J_hat_t = (s.pow(2).sum(dim=1)).mean().item()
    J_true_t = n / (P + float(t_value))
    return J_hat_t, J_true_t


# ------------------------------
# Plot overlay for multiple n values
# ------------------------------

def plot_overlay_curves(results: Dict, cfg: Config):
    """Plot J(Y_t) curves for different n values on the same plot"""

    # Colors and markers for different n values
    colors = ['b', 'r', 'g']  # Use single character color codes
    markers_true = ['o', '^', 's']
    markers_hat = ['v', 'd', 'p']

    # 1) Fisher Information overlay plot
    plt.figure(figsize=(8, 6))

    for idx, n in enumerate(cfg.n_list):
        t_grid = results[n]["t_grid"]
        J_hat = results[n]["J_hat"]
        J_true = results[n]["J_true"]

        t = torch.tensor(t_grid)
        J_hat_t = torch.tensor(J_hat)
        J_true_t = torch.tensor(J_true)

        # Plot true values
        plt.plot(t, J_true_t, colors[idx]+'-',
                label=f"J_true (n={n})",
                marker=markers_true[idx], markersize=5, alpha=0.8)

        # Plot estimated values
        plt.plot(t, J_hat_t, colors[idx]+'--',
                label=f"J_hat (n={n})",
                marker=markers_hat[idx], markersize=5, alpha=0.8)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Noise variance t")
    plt.ylabel("Fisher Information J(Y_t)")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(loc='upper right')

    fname1 = f"{cfg.output_prefix}_fisher.pdf"
    plt.savefig(fname1, dpi=150, bbox_inches="tight")
    plt.close()

    # 2) Relative error overlay plot
    plt.figure(figsize=(8, 6))

    for idx, n in enumerate(cfg.n_list):
        t_grid = results[n]["t_grid"]
        rel_err = results[n]["rel_err"]

        t = torch.tensor(t_grid)
        rel = torch.tensor(rel_err)

        plt.plot(t, rel * 100, colors[idx]+'-',
                label=f"n={n}",
                marker=markers_true[idx], markersize=5)

    plt.xscale("log")
    plt.xlabel("Noise variance t")
    plt.ylabel("Relative error (%)")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(loc='upper left')

    fname2 = f"{cfg.output_prefix}_error.pdf"
    plt.savefig(fname2, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {fname1}, {fname2}")


# ------------------------------
# Main orchestration
# ------------------------------

def run():
    cfg = Config()
    print(cfg)
    set_seed(cfg.seed)

    results: Dict[int, Dict[str, object]] = {}

    # Generate t_grid (same for all n values)
    t_grid = make_t_grid(cfg.P, cfg.M, cfg.t_min_scale, cfg.t_max_scale).to(cfg.device)

    for n in cfg.n_list:
        print(f"\n=== Processing n={n} (P={cfg.P}) ===")

        J_hat_list = []
        J_true_list = []

        for tval in t_grid:
            t_scalar = float(tval.item())
            # Train one model at this fixed t
            model_t = train_one_fixed_t(cfg, n, cfg.P, t_scalar)
            # Estimate Fisher
            J_hat_t, J_true_t = estimate_fisher_fixed_t(model_t, n, cfg.P, t_scalar, cfg)
            J_hat_list.append(J_hat_t)
            J_true_list.append(J_true_t)

        J_hat = torch.tensor(J_hat_list)
        J_true = torch.tensor(J_true_list)
        rel_err = (J_hat - J_true).abs() / J_true

        results[n] = {
            "t_grid": t_grid.cpu().numpy(),
            "J_hat": J_hat.cpu().numpy(),
            "J_true": J_true.cpu().numpy(),
            "rel_err": rel_err.cpu().numpy(),
        }

        # Quick summary
        print(f"Summary for n={n}:")
        print(f"  median rel. err: {rel_err.median().item():.4f}")
        print(f"  90th pct rel. err: {rel_err.quantile(0.9).item():.4f}")

    # Create overlay plots
    plot_overlay_curves(results, cfg)

    # Save all results
    torch.save({"config": cfg.__dict__, "results": results}, f"{cfg.output_prefix}_results.pt")
    print(f"Saved results to {cfg.output_prefix}_results.pt")


if __name__ == "__main__":
    run()