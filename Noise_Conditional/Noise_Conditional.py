import matplotlib.pyplot as plt
import torch
from torch import nn
from dataclasses import dataclass
import math
from typing import Dict, Tuple

# -----------------------------------------------------------------------------
# Configuration for Noise-Conditional Model
# -----------------------------------------------------------------------------
@dataclass
class Cfg:
    # CPUで実行するように変更 (実行環境にGPUがない場合を想定)
    device: str = "cpu"
    seed: int = 42
    n: int = 4
    P_list: Tuple[float, ...] = (1.0,)
    M: int = 12  # Number of grid points for evaluation
    t_min_scale: float = 1/200
    t_max_scale: float = 50
    # Training is now done with a single loop
    total_steps: int = 20000  # Total training steps for the single model
    batch_size: int = 4096
    lr: float = 1e-3
    grad_clip: float | None = 1.0
    # Loss weighting lambda(t)=t as suggested in paper
    use_loss_weighting: bool = True
    # Evaluation
    eval_samples_per_t: int = 100_000
    # Model architecture
    hidden: int = 128
    layers: int = 3
    t_embed_dim: int = 64 # Dimension for time embedding
    save_path: str = "mi_results_conditional.pt"


# -----------------------------------------------------------------------------
# Seeding and t-grid generation
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
# Noise-Conditional Score Model
# -----------------------------------------------------------------------------
class GaussianFourierProjection(nn.Module):
    """Encodes scalar `t` into a vector using Gaussian Fourier projection."""
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

class ScoreNetConditional(nn.Module):
    """A score model that takes y and t as input."""
    def __init__(self, n: int, hidden: int, layers: int, t_embed_dim: int):
        super().__init__()
        self.t_embed = GaussianFourierProjection(t_embed_dim)
        self.t_map = nn.Sequential(nn.Linear(t_embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

        self.y_layers = nn.ModuleList()
        self.y_layers.append(nn.Linear(n, hidden))
        for _ in range(layers - 1):
            self.y_layers.append(nn.Linear(hidden, hidden))
        self.final_layer = nn.Linear(hidden, n)

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t is a scalar per batch item, so we use log(t) for better scaling
        t_embedding = self.t_map(self.t_embed(torch.log(t)))

        h = y
        for i, layer in enumerate(self.y_layers):
            h = layer(h)
            h += t_embedding # Add time embedding to hidden activations
            if i < len(self.y_layers) - 1:
                h = nn.functional.silu(h)
        return self.final_layer(h)

# -----------------------------------------------------------------------------
# Sampling and Training for Conditional Model
# -----------------------------------------------------------------------------
@torch.no_grad()
def sample_batch_conditional(P: float, n: int, batch: int, t: torch.Tensor, device: str):
    """Samples a batch for a given tensor of t values (one per batch item)."""
    x = torch.randn(batch, n, device=device) * math.sqrt(P)
    eps = torch.randn(batch, n, device=device)
    y = x + eps * torch.sqrt(t).unsqueeze(-1)
    target = eps / torch.sqrt(t).unsqueeze(-1)
    return y, target

def train_conditional_model(cfg: Cfg, P: float, t_min: float, t_max: float) -> ScoreNetConditional:
    device = cfg.device
    model = ScoreNetConditional(cfg.n, cfg.hidden, cfg.layers, cfg.t_embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    log_t_min, log_t_max = math.log(t_min), math.log(t_max)

    for step in range(1, cfg.total_steps + 1):
        # Sample t from a log-uniform distribution
        log_t = torch.rand(cfg.batch_size, device=device) * (log_t_max - log_t_min) + log_t_min
        t = torch.exp(log_t)

        y, target = sample_batch_conditional(P, cfg.n, cfg.batch_size, t, device)
        pred = model(y, t)
        
        sq_err = ((pred + target) ** 2).mean(dim=1)
        if cfg.use_loss_weighting:
            loss = (sq_err * t).mean()
        else:
            loss = sq_err.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % 2000 == 0 or step == 1:
            print(f"[P={P:.3g}] step {step:5d} | loss {loss.item():.6f}")
    return model

@torch.no_grad()
def estimate_fisher_conditional(model: ScoreNetConditional, P: float, t_grid: torch.Tensor, cfg: Cfg) -> torch.Tensor:
    model.eval()
    J_hat = []
    for t_val in t_grid:
        t_scalar = float(t_val.item())
        t_batch = torch.full((cfg.eval_samples_per_t,), t_scalar, device=cfg.device)
        
        x = torch.randn(cfg.eval_samples_per_t, cfg.n, device=cfg.device) * math.sqrt(P)
        eps = torch.randn(cfg.eval_samples_per_t, cfg.n, device=cfg.device)
        y = x + eps * math.sqrt(t_scalar)
        
        s = model(y, t_batch)
        J_hat.append((s.pow(2).sum(dim=1)).mean().item())
    return torch.tensor(J_hat, device="cpu")

# -----------------------------------------------------------------------------
# MI Calculation
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Main Experiment Runner
# -----------------------------------------------------------------------------
def run_mi_conditional():
    cfg = Cfg()
    print("Running with Noise-Conditional Model:")
    print(cfg)
    set_seed_all(cfg.seed)

    results: Dict[float, Dict[str, object]] = {}

    for P in cfg.P_list:
        print(f"\n=== MI via Conditional DSM for P={P} (n={cfg.n}) ===")
        t_grid_eval = make_t_grid(P, cfg.M, cfg.t_min_scale, cfg.t_max_scale).to(cfg.device)
        t_min_train = P * cfg.t_min_scale
        t_max_train = P * cfg.t_max_scale

        print("Training single conditional score model...")
        trained_model = train_conditional_model(cfg, P, t_min_train, t_max_train)

        print("\nEstimating Fisher Information across the t-grid...")
        J_hat = estimate_fisher_conditional(trained_model, P, t_grid_eval, cfg)

        t_cpu = t_grid_eval.cpu()
        g_hat = g_from_J(cfg.n, t_cpu, J_hat)
        tail = integrate_tail_gaussian(cfg.n, P, float(t_cpu[-1].item()))
        I_hat = cumulative_mi_from_g_log(t_cpu, g_hat, tail)
        I_true = 0.5 * cfg.n * torch.log1p(P / t_cpu)
        rel_err = (I_hat - I_true).abs() / torch.clamp(I_true, 1e-9)
        
        results[P] = {
            "t_grid": t_cpu.numpy(), "J_hat": J_hat.numpy(), "g_hat": g_hat.numpy(),
            "I_hat": I_hat.numpy(), "I_true": I_true.numpy(), "rel_err_I": rel_err.numpy(),
        }

        print(f"Summary P={P}: median rel err(I)={rel_err.median().item():.4f} | 90th pct={rel_err.quantile(0.9).item():.4f}")

        plt.figure()
        plt.plot(t_cpu, I_true / cfg.n, 'b-', label="I_true/n = 0.5 log(1+P/t)", marker='o', markersize=4)
        plt.plot(t_cpu, I_hat / cfg.n, 'r--', label="I_hat/n (Conditional DSM)", marker='s', markersize=4)
        plt.xscale("log")
        plt.xlabel("Noise variance t"); plt.ylabel("I(X;Y_t)/n")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(); plt.savefig(f"MI_vs_t_Conditional_P{str(P).replace('.', 'p')}.pdf", dpi=150, bbox_inches="tight"); plt.close()

        plt.figure()
        plt.plot(t_cpu, rel_err * 100, 'g-', marker='o', markersize=4)
        plt.xscale("log"); plt.xlabel("Noise variance t"); plt.ylabel("Relative error (%)")
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.savefig(f"RelErr_MI_Conditional_P{str(P).replace('.', 'p')}.pdf", dpi=150, bbox_inches="tight"); plt.close()

    torch.save({"config": cfg.__dict__, "results": results}, cfg.save_path)
    print(f"\nSaved conditional MI results to {cfg.save_path}")


if __name__ == "__main__":
    run_mi_conditional()