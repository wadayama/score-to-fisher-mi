import matplotlib.pyplot as plt
import torch
from torch import nn
from dataclasses import dataclass
import math
from typing import Dict, Tuple
import numpy as np
from scipy import integrate

# -----------------------------------------------------------------------------
# Configuration for the BPSK MI Comparison Experiment
# -----------------------------------------------------------------------------
@dataclass
class Cfg:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    # BPSK is a 1-dimensional constellation
    n: int = 1
    P: float = 1.0 # Power, so symbols are at +/- sqrt(P)
    # Evaluation grid
    M: int = 12
    t_min_scale: float = 1/200
    t_max_scale: float = 50.0
    # Per-t training params
    steps_per_t: int = 1000 # Sufficient for good accuracy
    # Common params
    batch_size: int = 4096
    lr: float = 1e-3
    grad_clip: float | None = 1.0
    # Evaluation
    eval_samples_per_t: int = 200_000
    # Model architecture
    hidden: int = 128
    layers: int = 3
    t_embed_dim: int = 64
    # Output
    save_path: str = "bpsk_mi_comparison.pt"
    plot_path_pdf: str = "bpsk_mi_comparison.pdf"


# -----------------------------------------------------------------------------
# Seeding and t-grid
# -----------------------------------------------------------------------------
def set_seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_t_grid(P: float, M: int, a: float, b: float) -> torch.Tensor:
    t_min = P * a
    t_max = P * b
    return torch.logspace(math.log10(t_min), math.log10(t_max), M)

# -----------------------------------------------------------------------------
# BPSK Sampling (NEW)
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_batch_bpsk(P: float, n: int, batch: int, t: torch.Tensor, device: str, for_kde=False):
    """Samples a batch using BPSK input."""
    assert n == 1, "BPSK is defined for n=1"
    # Equiprobable symbols from {-sqrt(P), +sqrt(P)}
    x = torch.randint(0, 2, (batch, n), device=device, dtype=torch.float32) * 2 - 1
    x *= math.sqrt(P)
    
    eps = torch.randn(batch, n, device=device)

    # Handle both scalar t and tensor t for different schemes
    if isinstance(t, torch.Tensor):
        sqrt_t = torch.sqrt(t)
        if sqrt_t.numel() > 1:
            sqrt_t = sqrt_t.unsqueeze(-1)
    else:
        # Convert scalar t to tensor
        sqrt_t = math.sqrt(t)

    y = x + eps * sqrt_t
    
    if for_kde:
        return y, None, x # Return y, None, w (where w=x for BPSK)

    target = eps / sqrt_t
    return y, target

# -----------------------------------------------------------------------------
# Models: Scheme A (Fixed-t) and Scheme B (Conditional)
# -----------------------------------------------------------------------------
class ScoreNetFixedT(nn.Module): # For Scheme A
    def __init__(self, n: int, hidden: int, layers: int):
        super().__init__()
        dims = [n] + [hidden] * layers + [n]
        mods = []
        for i in range(len(dims) - 2):
            mods += [nn.Linear(dims[i], dims[i+1]), nn.SiLU()]
        mods += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*mods)
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)

# Conditional model classes removed (not needed for per-t training)

# -----------------------------------------------------------------------------
# Training and Estimation Logic
# -----------------------------------------------------------------------------
def train_per_t_model(cfg: Cfg, t_value: float) -> ScoreNetFixedT:
    model = ScoreNetFixedT(cfg.n, cfg.hidden, cfg.layers).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for step in range(1, cfg.steps_per_t + 1):
        y, target = sample_batch_bpsk(cfg.P, cfg.n, cfg.batch_size, t_value, cfg.device)
        pred = model(y)
        loss = ((pred + target) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        if cfg.grad_clip: nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
    return model

# Scheme B (conditional model) removed due to poor performance on discrete inputs

@torch.no_grad()
def estimate_fisher(model: nn.Module, t_grid: torch.Tensor, cfg: Cfg) -> torch.Tensor:
    model.eval()
    J_hat = []
    for t_val in t_grid:
        t_scalar = float(t_val.item())
        y, _ = sample_batch_bpsk(cfg.P, cfg.n, cfg.eval_samples_per_t, t_scalar, cfg.device)
        s = model(y)  # Only per-t models now
        J_hat.append((s.pow(2).sum(dim=1)).mean().item())
    return torch.tensor(J_hat, device="cpu")

# -----------------------------------------------------------------------------
# MI Reconstruction and KDE Baseline
# -----------------------------------------------------------------------------
def tail_from_mmse_hat(t_grid: torch.Tensor, J_hat: torch.Tensor, n: int, P: float) -> float:
    """Universal tail correction using estimated MMSE at t_max."""
    # For large t, mmse(t) -> Var(X). For BPSK, Var(X) = P.
    # We use a data-driven version for generality.
    t_max = t_grid[-1].item()
    J_max = J_hat[-1].item()
    mmse_max = n * t_max - t_max**2 * J_max
    return 0.5 * mmse_max / t_max

def cumulative_mi_from_g_log(t_grid: torch.Tensor, J_hat: torch.Tensor, n: int, P: float) -> torch.Tensor:
    g_vals = 0.5 * (n / t_grid - J_hat)
    tail = tail_from_mmse_hat(t_grid, J_hat, n, P)
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

def compute_true_mi_bpsk(P: float, t: float) -> float:
    """Compute true MI for BPSK via numerical integration: I(X;Y_t) = H(Y_t) - H(Z_t)"""
    sqrt_P = math.sqrt(P)
    sqrt_t = math.sqrt(t)

    def p_y_given_x_plus(y):
        return math.exp(-(y - sqrt_P)**2 / (2*t)) / math.sqrt(2*math.pi*t)

    def p_y_given_x_minus(y):
        return math.exp(-(y + sqrt_P)**2 / (2*t)) / math.sqrt(2*math.pi*t)

    def p_y(y):
        return 0.5 * (p_y_given_x_plus(y) + p_y_given_x_minus(y))

    def h_y_integrand(y):
        p_val = p_y(y)
        if p_val <= 1e-15:
            return 0.0
        return -p_val * math.log(p_val)

    # Integration bounds: use wider range for numerical stability
    y_max = 6 * sqrt_t + sqrt_P
    H_Y, _ = integrate.quad(h_y_integrand, -y_max, y_max, epsabs=1e-10, epsrel=1e-8)

    # H(Z_t) where Z_t ~ N(0, t)
    H_Z = 0.5 * math.log(2 * math.pi * math.e * t)

    return H_Y - H_Z

@torch.no_grad()
def compute_true_mi_grid(t_grid: torch.Tensor, cfg: Cfg) -> torch.Tensor:
    I_true = []
    print("Computing True MI via Numerical Integration...")
    for t_val in t_grid:
        t_scalar = t_val.item()
        mi_true = compute_true_mi_bpsk(cfg.P, t_scalar)
        I_true.append(mi_true)
        print(f"  t={t_scalar:.4g} | I_true/n = {mi_true/cfg.n:.6f}")
    return torch.tensor(I_true)

# -----------------------------------------------------------------------------
# Main Experiment Runner
# -----------------------------------------------------------------------------
def run_bpsk_dsm_validation():
    cfg = Cfg()
    print("Starting BPSK MI Estimation via DSM:")
    print(cfg)
    set_seed_all(cfg.seed)
    
    t_grid = make_t_grid(cfg.P, cfg.M, cfg.t_min_scale, cfg.t_max_scale)
    t_cpu = t_grid.cpu()

    # --- Per-t DSM Training ---
    print("\n--- Running Per-t DSM Training ---")
    J_hat = []
    for t_val in t_grid:
        print(f"Training DSM model for t = {t_val.item():.4g}")
        model = train_per_t_model(cfg, t_val.item())
        J_hat.append(estimate_fisher(model, t_val.unsqueeze(0), cfg).item())
    J_hat = torch.tensor(J_hat)
    I_hat = cumulative_mi_from_g_log(t_cpu, J_hat, cfg.n, cfg.P)

    # Scheme B removed due to poor performance on discrete inputs

    # --- True MI via Numerical Integration ---
    I_true = compute_true_mi_grid(t_cpu, cfg)

    # --- Save and Plot ---
    I_gaussian = 0.5 * cfg.n * torch.log1p(cfg.P / t_cpu)
    results = {
        "config": cfg.__dict__, "t_grid": t_cpu.numpy(),
        "I_hat": I_hat.numpy(), "I_true": I_true.numpy(), "I_gaussian": I_gaussian.numpy()
    }
    torch.save(results, cfg.save_path)
    print(f"\nSaved all results to {cfg.save_path}")

    plt.figure(figsize=(8, 6))
    # Gaussian input theoretical baseline for comparison
    I_gaussian = 0.5 * cfg.n * torch.log1p(cfg.P / t_cpu)
    plt.plot(t_cpu, I_gaussian / cfg.n, ':', color='blue', label="Gaussian input (theoretical)", linewidth=2)
    plt.plot(t_cpu, I_true / cfg.n, 'o-', color='green', label="BPSK exact MI (numerical integration)", markersize=5)
    plt.plot(t_cpu, I_hat / cfg.n, 's--', color='red', label="BPSK DSM estimate (per-t training)", markersize=5)
    plt.xscale("log")
    plt.xlabel("Noise variance t")
    plt.ylabel("I(X;Y_t)/n")
    plt.ylim(0, 1.5)
    plt.grid(True, which='both', linestyle='-', linewidth=0.3)
    plt.legend()
    plt.savefig(cfg.plot_path_pdf, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Generated plot: {cfg.plot_path_pdf}")

    # Plot relative error
    rel_error = torch.abs(I_hat - I_true) / torch.clamp(I_true, 1e-9) * 100
    plt.figure(figsize=(8, 6))
    plt.plot(t_cpu, rel_error, 'o-', color='red', markersize=5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Noise variance t")
    plt.ylabel("Relative error (%)")
    plt.grid(True, which='both', linestyle='-', linewidth=0.3)
    plt.savefig("bpsk_error_analysis.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("Generated error analysis plot: bpsk_error_analysis.pdf")

if __name__ == "__main__":
    run_bpsk_dsm_validation()