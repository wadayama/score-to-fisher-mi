# --------------------------------------------
# Nonlinear channel (tanh with linear mixing): MI via DSM (proposed) vs KDE-LOO (baseline)
# Base: MI.py (fixed-t DSM, log-domain integration, tail correction)
# --------------------------------------------
# Model:
#   X ~ N(0, P I_n),  W = f(AX) with f(x)=tanh(x) (elementwise), A is random orthogonal matrix
#   Y_t = W + sqrt(t) * eps,  eps ~ N(0, I_n)
#
# DSM (proposed):
#   Learn s_theta(y) ≈ ∇_y log p_{Y_t}(y) for each fixed t (no t-conditioning).
#   Estimate J(Y_t) = E||s_theta(Y_t)||^2, then
#     g(t) = 0.5*(n/t - J(Y_t)),  I(T) = ∫_T^∞ g(t) dt
#   Integration: log-domain trapezoid on a geometric t-grid + tail correction.
#   Tail: distribution-agnostic, from mmse_hat at t_max: tail ≈ 0.5 * mmse(t_max) / t_max
#         where mmse(t) = n t - t^2 J(Y_t).
#
# KDE-LOO baseline (simulation-based, plug-in from definition):
#   With W samples {w_j}, for each y_i = w_i + sqrt(t) eps_i:
#     I ≈ (1/N) Σ_i [ log q_t(y_i - w_i) - log((1/(N-1)) Σ_{j≠i} q_t(y_i - w_j)) ]
#   For Gaussian noise, the normalization cancels; implement with LOO + kNN (top-K) neighbors,
#   log-sum-exp stabilized, float64 for numerical stability.
#
# Outputs:
#   - Plots MI vs t (DSM vs KDE), and relative difference
#   - Saves tensors to mi_nonlinear_tanh.pt for reuse
# --------------------------------------------

from dataclasses import dataclass
import math
from typing import Dict, Tuple
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
    # No alpha parameter, A matrix provides mixing
    M: int = 12                  # number of t grid points
    t_min_scale: float = 1/200   # t_min = P * t_min_scale
    t_max_scale: float = 50.0    # t_max = P * t_max_scale
    steps_per_t: int = 400       # DSM steps per t (increase for higher accuracy)
    batch_size: int = 8192
    lr: float = 1e-3
    grad_clip: float | None = 5.0
    eval_samples_per_t: int = 100_000
    hidden: int = 128
    layers: int = 3
    # KDE baseline
    kde_N: int = 20000           # number of samples for KDE baseline (per t)
    kde_k: int = 300             # kNN neighbors used in mixture sum
    kde_chunk: int = 2000        # process y in chunks for distance computation
    # Save
    save_path: str = "mi_nonlinear_tanh_linear_fullsum.pt"


def set_seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Nonlinearity and sampling
# -----------------------------
def f_nonlinear(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Apply linear transformation followed by tanh: tanh(Ax)"""
    return torch.tanh(x @ A.T)  # x is [batch, n], A is [n, n]

@torch.no_grad()
def sample_batch_fixed_t(cfg: MICfg, A: torch.Tensor, t: float, batch: int, for_kde: bool = False):
    """Return (y, target, w) for DSM; or (y, None, w) if for_kde=True."""
    n, P, device = cfg.n, cfg.P, cfg.device
    x = torch.randn(batch, n, device=device) * math.sqrt(P)
    w = f_nonlinear(x, A)
    eps = torch.randn(batch, n, device=device)
    y = w + eps * math.sqrt(t)
    if for_kde:
        return y, None, w
    target = eps / math.sqrt(t)  # DSM target (score of Gaussian noise)
    return y, target, w

@torch.no_grad()
def sample_y_only(cfg: MICfg, A: torch.Tensor, t: float, num: int):
    n, P, device = cfg.n, cfg.P, cfg.device
    x = torch.randn(num, n, device=device) * math.sqrt(P)
    w = f_nonlinear(x, A)
    eps = torch.randn(num, n, device=device)
    y = w + eps * math.sqrt(t)
    return y


# -----------------------------
# DSM model and training
# -----------------------------
class ScoreNetFixedT(nn.Module):
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


def train_one_fixed_t(cfg: MICfg, A: torch.Tensor, t_value: float) -> ScoreNetFixedT:
    device = cfg.device
    model = ScoreNetFixedT(cfg.n, cfg.hidden, cfg.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for step in range(1, cfg.steps_per_t + 1):
        y, target, _ = sample_batch_fixed_t(cfg, A, t_value, cfg.batch_size)
        pred = model(y)
        loss = ((pred + target) ** 2).mean()  # DSM: '+' sign is important
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"[t={t_value:.4g}] step {step:4d} | loss {loss.item():.6f}")
    return model


@torch.no_grad()
def estimate_fisher_fixed_t(model: ScoreNetFixedT, cfg: MICfg, A: torch.Tensor, t_value: float) -> float:
    model.eval()
    y = sample_y_only(cfg, A, t_value, cfg.eval_samples_per_t)
    s = model(y)
    return (s.pow(2).sum(dim=1)).mean().item()


# -----------------------------
# MI reconstruction (log-domain + tail from mmse_hat)
# -----------------------------
def g_from_J(n: int, t: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    return 0.5 * (n / t - J)

def tail_from_mmse_hat(t_grid: torch.Tensor, J_hat: torch.Tensor, n: int, K: int = 3) -> float:
    """Universal tail: tail ≈ (1/2) * mmse(t_max) / t_max, using last K points average."""
    t_tail = t_grid[-K:]
    J_tail = J_hat[-K:]
    mmse_tail = (n * t_tail - (t_tail**2) * J_tail).mean().item()
    tmax = float(t_grid[-1])
    return 0.5 * mmse_tail / tmax

def cumulative_mi_from_g_log(t_grid: torch.Tensor, g_vals: torch.Tensor, tail: float) -> torch.Tensor:
    """I_hat(t_k) = ∫_{t_k}^{t_max} g(t) dt + tail, with u=log t trapezoid."""
    M = t_grid.numel()
    I = torch.zeros(M, dtype=g_vals.dtype)
    u = torch.log(t_grid)
    du = (u[1] - u[0]).item()
    integrand = g_vals * t_grid  # g(e^u) * e^u
    area = 0.0
    I[-1] = tail
    for k in range(M-2, -1, -1):
        area += 0.5 * (integrand[k+1].item() + integrand[k].item()) * du
        I[k] = area + tail
    return I


# -----------------------------
# KDE-LOO baseline (kNN-accelerated)
# -----------------------------
@torch.no_grad()
def kde_mi_loo_full(y: torch.Tensor, w: torch.Tensor, t: float, chunk: int = 2000) -> float:
    """
    Full-sum LOO plug-in MI estimator (no kNN approximation):
      I ≈ (1/N) Σ_i [ -||y_i - w_i||^2/(2t) - log( (1/(N-1)) Σ_{j≠i} exp(-||y_i - w_j||^2/(2t)) ) ]
    y, w: [N, n], on CPU float64 for stability.
    """
    assert y.shape == w.shape
    N, n = y.shape
    two_t = 2.0 * t
    w2 = (w**2).sum(dim=1)  # [N]

    def batch_compute(i0: int, i1: int) -> torch.Tensor:
        Y = y[i0:i1]                                   # [B, n]
        Wdot = Y @ w.T                                 # [B, N]
        y2 = (Y**2).sum(dim=1, keepdim=True)           # [B, 1]
        d2 = y2 + w2.unsqueeze(0) - 2.0 * Wdot         # [B, N]
        # exclude self index for exact LOO
        rows = torch.arange(i0, i1)
        d2[torch.arange(i1-i0), rows] = float("inf")

        # full sum over all neighbors
        a = -d2 / two_t
        a_max, _ = a.max(dim=1, keepdim=True)
        lse = a_max.squeeze(1) + torch.log(torch.exp(a - a_max).sum(dim=1))  # [B]
        # mixture average (1/(N-1)) factor -> subtract log(N-1)
        log_mix = lse - math.log(N-1)

        # first term: -||y_i - w_i||^2/(2t)
        diag_exact = ((Y - w[i0:i1])**2).sum(dim=1)
        term = -diag_exact / two_t - log_mix
        return term

    acc = 0.0
    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk)
        acc += batch_compute(i0, i1).sum().item()
    return acc / N

@torch.no_grad()
def kde_mi_loo_adaptive(y: torch.Tensor, w: torch.Tensor, t: float,
                        k: int = 300, chunk: int = 2000, t_switch: float = 0.5) -> float:
    # y,w: [N,n] float64 on CPU
    assert y.shape == w.shape
    N, n = y.shape
    two_t = 2.0 * t
    w2 = (w**2).sum(dim=1)

    def block(i0, i1, full):
        Y = y[i0:i1]                                 # [B,n]
        y2 = (Y**2).sum(dim=1, keepdim=True)         # [B,1]
        Wdot = Y @ w.T                                # [B,N]
        d2 = y2 + w2.unsqueeze(0) - 2.0*Wdot         # [B,N]
        rows = torch.arange(i0, i1)
        d2[torch.arange(i1-i0), rows] = float("inf") # LOO
        if full:
            a = -d2 / two_t
            a_max, _ = a.max(dim=1, keepdim=True)
            lse = a_max.squeeze(1) + torch.log(torch.exp(a - a_max).sum(dim=1))
        else:
            k_eff = min(max(k, 300), N-1)            # 小tはkNN、kは十分大きく
            vals, _ = torch.topk(-d2, k=k_eff, dim=1)
            d2_knn = -vals
            a = -d2_knn / two_t
            a_max, _ = a.max(dim=1, keepdim=True)
            lse = a_max.squeeze(1) + torch.log(torch.exp(a - a_max).sum(dim=1))
        log_mix = lse - math.log(N-1)
        diag = ((Y - w[i0:i1])**2).sum(dim=1)
        return (-diag / two_t - log_mix).sum().item()

    full = (t >= t_switch)
    acc = 0.0
    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk)
        acc += block(i0, i1, full)
    return acc / N

@torch.no_grad()
def kde_mi_loo_knn(y: torch.Tensor, w: torch.Tensor, t: float, k: int = 300, chunk: int = 2000) -> float:
    """
    LOO plug-in MI estimator for additive Gaussian noise with variance t I:
      I ≈ (1/N) Σ_i [ -||y_i - w_i||^2/(2t) - log( (1/(N-1)) Σ_{j≠i} exp(-||y_i - w_j||^2/(2t)) ) ]
    Use top-k nearest neighbors for the mixture sum (excluding i).
    y, w: [N, n], on CPU float64 for stability is recommended.
    """
    assert y.shape == w.shape
    N, n = y.shape
    y = y.contiguous()
    w = w.contiguous()
    two_t = 2.0 * t

    # Precompute ||w_j||^2
    w2 = (w**2).sum(dim=1)  # [N]

    def batch_compute(i0: int, i1: int) -> torch.Tensor:
        Y = y[i0:i1]                                   # [B, n]
        Wdot = Y @ w.T                                 # [B, N]
        y2 = (Y**2).sum(dim=1, keepdim=True)           # [B, 1]
        # dist^2 = ||y||^2 + ||w||^2 - 2 y·w
        d2 = y2 + w2.unsqueeze(0) - 2.0 * Wdot         # [B, N]
        # exclude self index for exact LOO
        rows = torch.arange(i0, i1)
        d2[torch.arange(i1-i0), rows] = float("inf")

        # take k nearest neighbors
        k_eff = min(k, N-1)
        vals, _ = torch.topk(-d2, k=k_eff, dim=1)      # largest of (-d2) = smallest distances
        d2_knn = -vals                                 # [B, k_eff]

        # log-sum-exp over exp(-d2/(2t))
        a = -d2_knn / two_t
        a_max, _ = a.max(dim=1, keepdim=True)
        lse = a_max.squeeze(1) + torch.log(torch.exp(a - a_max).sum(dim=1))  # [B]
        # mixture average (1/(N-1)) factor -> subtract log(N-1)
        log_mix = lse - math.log(N-1)

        # first term: -||y_i - w_i||^2/(2t)
        diag = d2[torch.arange(i1-i0), rows]  # this should be inf (we set), so compute directly:
        # compute exact diag distance:
        diag_exact = ((Y - w[i0:i1])**2).sum(dim=1)
        term = -diag_exact / two_t - log_mix
        return term

    acc = 0.0
    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk)
        acc += batch_compute(i0, i1).sum().item()
    return acc / N


# -----------------------------
# Experiment runner
# -----------------------------
def make_t_grid(P: float, M: int, a: float, b: float) -> torch.Tensor:
    t_min = P * a
    t_max = P * b
    return torch.logspace(math.log10(t_min), math.log10(t_max), M)

def run_experiment():
    cfg = MICfg()
    print(cfg)
    set_seed_all(cfg.seed)

    device = cfg.device

    # Generate random orthogonal matrix A
    Q, _ = torch.linalg.qr(torch.randn(cfg.n, cfg.n))
    A = Q.to(device)
    print(f"Generated random orthogonal matrix A with shape {A.shape}")

    # t-grid (geometric)
    t_grid = make_t_grid(cfg.P, cfg.M, cfg.t_min_scale, cfg.t_max_scale).to(device)

    # --- DSM path: train per t, estimate J_hat
    J_hat = []
    for tval in t_grid:
        t_scalar = float(tval.item())
        model_t = train_one_fixed_t(cfg, A, t_scalar)
        J_hat_t = estimate_fisher_fixed_t(model_t, cfg, A, t_scalar)
        J_hat.append(J_hat_t)
    J_hat = torch.tensor(J_hat, dtype=torch.float64)  # use float64 for accumulations
    t_cpu = t_grid.cpu().to(torch.float64)

    # g, tail, I_hat (DSM)
    g_hat = 0.5 * (cfg.n / t_cpu - J_hat)
    tail = tail_from_mmse_hat(t_cpu, J_hat, cfg.n, K=3)
    I_hat_dsm = cumulative_mi_from_g_log(t_cpu, g_hat, tail)

    # --- KDE baseline on the same t-grid
    I_hat_kde = []
    torch.set_default_dtype(torch.float64)
    A_cpu = A.cpu().to(torch.float64)  # Convert A to float64 for KDE
    for tval in t_cpu:
        t_scalar = float(tval.item())
        # Generate samples for KDE baseline
        y, _, w = sample_batch_fixed_t(cfg, A_cpu, t_scalar, cfg.kde_N, for_kde=True)
        y_cpu = y.cpu().to(torch.float64)
        w_cpu = w.cpu().to(torch.float64)
        mi_est = kde_mi_loo_full(y_cpu, w_cpu, t_scalar, chunk=cfg.kde_chunk)
        I_hat_kde.append(mi_est)
        print(f"[KDE] t={t_scalar:.4g} | I_hat_kde/n = {mi_est/cfg.n:.6f}")
    I_hat_kde = torch.tensor(I_hat_kde, dtype=torch.float64)

    # --- Save and plots
    torch.save({
        "config": cfg.__dict__,
        "A_matrix": A.cpu().numpy(),
        "t_grid": t_cpu.numpy(),
        "J_hat": J_hat.numpy(),
        "g_hat": g_hat.numpy(),
        "I_hat_dsm": I_hat_dsm.numpy(),
        "I_hat_kde": I_hat_kde.numpy(),
        "tail": tail,
    }, cfg.save_path)
    print(f"Saved results to {cfg.save_path}")

    # Plot MI (per-dimension)
    plt.figure()
    plt.plot(t_cpu, I_hat_kde / cfg.n, 'o-', label="KDE-LOO baseline (per dim)", markersize=5)
    plt.plot(t_cpu, I_hat_dsm / cfg.n, "^--", label="DSM (proposed, per dim)", markersize=5)
    plt.xscale("log")
    plt.xlabel("Noise variance t")
    plt.ylabel("I(X;Y_t)/n")
    # Add grid with thin lines
    plt.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5)
    plt.legend()
    plt.savefig("MI_tanh_linear_DSM_vs_KDE_fullsum.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    # Relative difference |DSM-KDE| / KDE
    rel = (I_hat_dsm - I_hat_kde).abs() / torch.clamp(I_hat_kde.abs(), min=1e-12)
    plt.figure()
    plt.plot(t_cpu, rel, 's-', markersize=4)
    plt.xscale("log")
    plt.xlabel("Noise variance t")
    plt.ylabel("relative difference vs KDE")
    # Add grid with thin lines
    plt.grid(True, which='both', linestyle='-', linewidth=0.3, alpha=0.5)
    plt.savefig("RelDiff_tanh_linear_fullsum.pdf", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_experiment()
