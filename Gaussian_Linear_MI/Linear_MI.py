# ------------------------------
# Mutual Information via DSM for Linear Gaussian Channel
# ------------------------------
"""
Extension of MI.py to the linear Gaussian channel:
    Y_t = A X + Z_t,   Z_t ~ N(0, t I_n)

Closed-form MI:
    I(X;Y_t) = 0.5 * log det( I + (P/t) A A^T )

DSM estimates J(Y_t), then g(t), then MI via log-domain integration + tail correction.
"""

from dataclasses import dataclass
import math
from typing import Dict
import torch
from torch import nn
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
@dataclass
class MICfg:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    n: int = 4
    P: float = 1.0
    M: int = 10
    t_min_scale: float = 1/200
    t_max_scale: float = 50
    steps_per_t: int = 300
    batch_size: int = 8192
    lr: float = 1e-3
    grad_clip: float | None = 5.0
    eval_samples_per_t: int = 100_000
    hidden: int = 128
    layers: int = 3
    save_path: str = "mi_results_linear.pt"


def set_seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Model and sampling
# -----------------------------
class ScoreNetFixedT(nn.Module):
    def __init__(self, n: int, hidden: int, layers: int):
        super().__init__()
        dims = [n] + [hidden] * layers + [n]
        modules = []
        for i in range(len(dims) - 2):
            modules += [nn.Linear(dims[i], dims[i+1]), nn.SiLU()]
        modules += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*modules)
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


@torch.no_grad()
def sample_batch_fixed_t(P: float, A: torch.Tensor, n: int, batch: int, t: float, device: str):
    x = torch.randn(batch, n, device=device) * math.sqrt(P)
    eps = torch.randn(batch, n, device=device)
    y = (x @ A.T) + eps * math.sqrt(t)
    # True score of noise part: eps/sqrt(t)
    target = eps / math.sqrt(t)
    return y, target


@torch.no_grad()
def sample_y_only(P: float, A: torch.Tensor, n: int, num: int, t: float, device: str):
    x = torch.randn(num, n, device=device) * math.sqrt(P)
    eps = torch.randn(num, n, device=device)
    y = (x @ A.T) + eps * math.sqrt(t)
    return y


# -----------------------------
# Training and estimation
# -----------------------------
def train_one_fixed_t(cfg: MICfg, P: float, A: torch.Tensor, t_value: float) -> ScoreNetFixedT:
    device = cfg.device
    model = ScoreNetFixedT(cfg.n, cfg.hidden, cfg.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for step in range(1, cfg.steps_per_t + 1):
        y, target = sample_batch_fixed_t(P, A, cfg.n, cfg.batch_size, t_value, device)
        pred = model(y)
        loss = ((pred + target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
    return model


@torch.no_grad()
def estimate_fisher_fixed_t(model: ScoreNetFixedT, P: float, A: torch.Tensor, t_value: float, cfg: MICfg) -> float:
    model.eval()
    y = sample_y_only(P, A, cfg.n, cfg.eval_samples_per_t, t_value, cfg.device)
    s = model(y)
    return (s.pow(2).sum(dim=1)).mean().item()


def g_from_J(n: int, t: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    return 0.5 * (n / t - J)


def integrate_tail_gaussian(n: int, P: float, t_max: float) -> float:
    return 0.5 * n * P / t_max


def cumulative_mi_from_g_log(t_grid: torch.Tensor, g_vals: torch.Tensor, tail: float) -> torch.Tensor:
    M = t_grid.numel()
    I = torch.zeros(M, dtype=g_vals.dtype)
    u_grid = torch.log(t_grid)
    du = (u_grid[1] - u_grid[0]).item()
    integrand = g_vals * t_grid
    area = 0.0
    I[M-1] = tail
    for k in range(M-2, -1, -1):
        area += 0.5 * (integrand[k+1].item() + integrand[k].item()) * du
        I[k] = area + tail
    return I


# -----------------------------
# Run experiment
# -----------------------------
def run_mi_linear():
    cfg = MICfg()
    set_seed_all(cfg.seed)
    results: Dict[str, object] = {}

    # Example A: random orthogonal
    Q, _ = torch.linalg.qr(torch.randn(cfg.n, cfg.n))
    A = Q.to(cfg.device)

    t_grid = torch.logspace(math.log10(cfg.P*cfg.t_min_scale),
                            math.log10(cfg.P*cfg.t_max_scale), cfg.M).to(cfg.device)

    J_hat = []
    for tval in t_grid:
        model_t = train_one_fixed_t(cfg, cfg.P, A, float(tval.item()))
        J_hat_t = estimate_fisher_fixed_t(model_t, cfg.P, A, float(tval.item()), cfg)
        J_hat.append(J_hat_t)
    J_hat = torch.tensor(J_hat, device="cpu")

    g_hat = g_from_J(cfg.n, t_grid.cpu(), J_hat)
    tail = integrate_tail_gaussian(cfg.n, cfg.P, float(t_grid[-1].item()))
    I_hat = cumulative_mi_from_g_log(t_grid.cpu(), g_hat, tail)

    # Closed form
    eigvals = torch.linalg.eigvalsh(A @ A.T).real
    I_true = 0.5 * torch.sum(torch.log1p(cfg.P / t_grid.cpu().unsqueeze(1) * eigvals), dim=1)

    # Plot
    plt.figure()
    plt.plot(t_grid.cpu(), I_true / cfg.n, 'o-', label="I_true/n", markersize=5)
    plt.plot(t_grid.cpu(), I_hat / cfg.n, "^--", label="I_hat/n", markersize=5)
    plt.xscale("log")
    plt.xlabel("Noise variance t")
    plt.ylabel("I(X;Y_t)/n")
    # Add grid with thin lines
    plt.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5)
    plt.legend()
    plt.savefig("MI_linear_vs_t.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved results to MI_linear_vs_t.pdf")


if __name__ == "__main__":
    run_mi_linear()
