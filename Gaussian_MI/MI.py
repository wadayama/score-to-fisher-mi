# ------------------------------
# Mutual Information via de Bruijn / I–MMSE (fixed-t DSM)
# ------------------------------
"""
Extend the fixed-t DSM pipeline to compute I(X;Y_t) from J(Y_t):
  dI/dt = 0.5 * ( n/t - J(Y_t) )  := g(t)
Thus, for each grid point t_k we estimate
  I_hat(t_k) = ∫_{t_k}^{∞} g_hat(u) du
Numerical integration over [t_k, t_max] uses a composite trapezoid on the t-grid.
Tail correction (t in (t_max, ∞)) for Gaussian input (f = Id, X~N(0, P I)):
  g(t) ~ 0.5 * n * P / t^2  ⇒  ∫_{t_max}^{∞} g(t) dt ≈ 0.5 * n * P / t_max
This matches the paper’s asymptotic (O(t^{-2})) and is exact for the Gaussian tail.
"""

from dataclasses import dataclass
import math
from typing import Dict, Tuple
import torch
from torch import nn
import matplotlib.pyplot as plt

# Reuse/assume ScoreNetFixedT, sampling, training from the fixed-t script above.
# For a self-contained MI run, we replicate the minimal definitions.

@dataclass
class MICfg:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    n: int = 4
    P_list: Tuple[float, ...] = (1.0,)  # Single P for quick test
    M: int = 10  # Reduced for quick test
    t_min_scale: float = 1/200
    t_max_scale: float = 50  # Reduced range
    steps_per_t: int = 300  # Reduced for quick test
    batch_size: int = 8192
    lr: float = 1e-3
    grad_clip: float | None = 5.0
    eval_samples_per_t: int = 100_000
    hidden: int = 128
    layers: int = 3
    save_path: str = "mi_results_fixed_t.pt"


def set_seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass


def make_t_grid(P: float, M: int, a: float, b: float) -> torch.Tensor:
    t_min = P * a
    t_max = P * b
    return torch.logspace(math.log10(t_min), math.log10(t_max), M)


class ScoreNetFixedT(nn.Module):
    def __init__(self, n: int, hidden: int, layers: int):
        super().__init__()
        dims = [n] + [hidden] * layers + [n]
        layers_ = []
        for i in range(len(dims) - 2):
            layers_ += [nn.Linear(dims[i], dims[i+1]), nn.SiLU()]
        layers_ += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers_)
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


@torch.no_grad()
def sample_batch_fixed_t(P: float, n: int, batch: int, t: float, device: str):
    x = torch.randn(batch, n, device=device) * math.sqrt(P)
    eps = torch.randn(batch, n, device=device)
    y = x + eps * math.sqrt(t)
    target = eps / math.sqrt(t)
    return y, target


@torch.no_grad()
def sample_y_only(P: float, n: int, num: int, t: float, device: str):
    x = torch.randn(num, n, device=device) * math.sqrt(P)
    eps = torch.randn(num, n, device=device)
    y = x + eps * math.sqrt(t)
    return y


def train_one_fixed_t(cfg: MICfg, P: float, t_value: float) -> ScoreNetFixedT:
    device = cfg.device
    model = ScoreNetFixedT(cfg.n, cfg.hidden, cfg.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for step in range(1, cfg.steps_per_t + 1):
        y, target = sample_batch_fixed_t(P, cfg.n, cfg.batch_size, t_value, device)
        pred = model(y)
        loss = ((pred + target) ** 2).mean()  # DSM loss (sign is '+')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"[P={P:.3g} | t={t_value:.3g}] step {step:4d} | loss {loss.item():.6f}")
    return model


@torch.no_grad()
def estimate_fisher_fixed_t(model: ScoreNetFixedT, P: float, t_value: float, cfg: MICfg) -> float:
    model.eval()
    y = sample_y_only(P, cfg.n, cfg.eval_samples_per_t, t_value, cfg.device)
    s = model(y)
    return (s.pow(2).sum(dim=1)).mean().item()


def g_from_J(n: int, t: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """Compute g(t) = 0.5 * (n/t - J(t)) on matching shapes.
    t, J are 1D tensors aligned to the t-grid.
    """
    return 0.5 * (n / t - J)


def integrate_tail_gaussian(n: int, P: float, t_max: float) -> float:
    """Tail correction ∫_{t_max}^∞ g(t) dt for Gaussian input.
    Asymptotic: g(t) ~ 0.5 * n * P / t^2 ⇒ integral = 0.5 * n * P / t_max.
    """
    return 0.5 * n * P / t_max


def cumulative_mi_from_g_log(t_grid: torch.Tensor, g_vals: torch.Tensor, tail: float) -> torch.Tensor:
    """For each index k, compute I_hat(t_k) = ∫_{t_k}^{t_max} g + tail.
    Using log-domain trapezoid integration for better numerical stability.

    Variable substitution: u = log(t), so ∫g(t)dt = ∫g(e^u)e^u du
    This converts to uniform spacing in u and reduces convexity bias.
    Returns tensor of shape [M] with decreasing values.
    """
    M = t_grid.numel()
    I = torch.zeros(M, dtype=g_vals.dtype)

    # Convert to log domain: u = log(t)
    u_grid = torch.log(t_grid)  # This should be equally spaced since t_grid is log-spaced
    du = (u_grid[1] - u_grid[0]).item()  # uniform spacing in log domain

    # Integrand in u-domain: g(e^u) * e^u
    integrand = g_vals * t_grid  # g(t) * t = g(e^u) * e^u

    # Build cumulative integral from the right using uniform trapezoid rule
    area = 0.0
    I[M-1] = tail  # no panel to the right; only tail
    for k in range(M-2, -1, -1):
        # Trapezoid rule with uniform spacing du
        area += 0.5 * (integrand[k+1].item() + integrand[k].item()) * du
        I[k] = area + tail
    return I


def run_mi():
    cfg = MICfg()
    print(cfg)
    set_seed_all(cfg.seed)

    results: Dict[float, Dict[str, object]] = {}

    for P in cfg.P_list:
        print(f"\n=== MI via DSM for P={P} (n={cfg.n}) ===")
        t_grid = make_t_grid(P, cfg.M, cfg.t_min_scale, cfg.t_max_scale).to(cfg.device)

        # Train per-t, estimate J_hat
        J_hat = []
        for tval in t_grid:
            t_scalar = float(tval.item())
            model_t = train_one_fixed_t(cfg, P, t_scalar)
            J_hat_t = estimate_fisher_fixed_t(model_t, P, t_scalar, cfg)
            J_hat.append(J_hat_t)
        J_hat = torch.tensor(J_hat, device="cpu")

        # Compute g_hat and numerical integral + tail correction
        t_cpu = t_grid.cpu()
        g_hat = g_from_J(cfg.n, t_cpu, J_hat)
        tail = integrate_tail_gaussian(cfg.n, P, float(t_cpu[-1].item()))
        I_hat = cumulative_mi_from_g_log(t_cpu, g_hat, tail)

        # Ground truth for comparison (Gaussian input): I_true(t) = 0.5 * n * log(1 + P/t)
        I_true = 0.5 * cfg.n * torch.log1p(P / t_cpu)

        rel_err = (I_hat - I_true).abs() / I_true

        results[P] = {
            "t_grid": t_cpu.numpy(),
            "J_hat": J_hat.numpy(),
            "g_hat": g_hat.numpy(),
            "I_hat": I_hat.numpy(),
            "I_true": I_true.numpy(),
            "rel_err_I": rel_err.numpy(),
        }

        print(f"Summary P={P}: median rel err(I)={rel_err.median().item():.4f} | 90th pct={rel_err.quantile(0.9).item():.4f}")

        # Plots for this P
        plt.figure()
        plt.plot(t_cpu, I_true / cfg.n, 'b-', label="I_true/n = 0.5 log(1+P/t)", marker='o', markersize=4)
        plt.plot(t_cpu, I_hat / cfg.n, 'r--', label="I_hat/n (DSM + tail)", marker='s', markersize=4)
        plt.xscale("log")
        plt.xlabel("Noise variance t"); plt.ylabel("I(X;Y_t)/n")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(); plt.savefig(f"MI_vs_t_P{str(P).replace('.', 'p')}.pdf", dpi=150, bbox_inches="tight"); plt.close()

        plt.figure()
        plt.plot(t_cpu, rel_err * 100, 'g-', marker='o', markersize=4)  # Convert to percentage
        plt.xscale("log"); plt.xlabel("Noise variance t"); plt.ylabel("Relative error (%)")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.savefig(f"RelErr_MI_P{str(P).replace('.', 'p')}.pdf", dpi=150, bbox_inches="tight"); plt.close()

    torch.save({"config": cfg.__dict__, "results": results}, cfg.save_path)
    print(f"Saved MI results to {cfg.save_path}")


if __name__ == "__main__":
    run_mi()
