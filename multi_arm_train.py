"""
Multi-Armed Bandit PINN — Three Algorithms
===========================================

Each arm k is a fixed weight vector (λ_pde^k, λ_bc^k, λ_mse^k).
Weights act on *normalised* losses:

    L̃_i(θ) = L_i(θ) / L_i(θ_0)          (scale-invariant)

    L_total  = Σ_i  λ_i^k · L̃_i(θ)      (what gets minimised during pilot)

Reward after each pilot run (minimax objective):

    r_k  =  − max{ L̃_pde, L̃_bc, L̃_mse }   (negative worst-case residual)

Three bandit policies evaluated:
  1. UCB  — optimism under uncertainty
  2. Thompson Sampling (TS)  — Bayesian posterior sampling (Normal–Normal)
  3. Posterior Sampling via Perturbed History (PSPH)  — add Gaussian noise to
     observed rewards, refit a linear model, sample from it

Usage
-----
python mab_pinn_v2.py \
    --Re 1000 --U 10 --num 81 \
    --dataset_base_path /projects/bfth/rhe4/PINN/data_pinn \
    --output_base_path  /projects/bfth/rhe4/PINN/results_mab_v2 \
    --pilot_epochs 300 \
    --total_rounds 40  \
    --algorithm    all          # 'ucb' | 'ts' | 'psph' | 'all'
"""

import argparse, os, math, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ══════════════════════════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════════════════════════

class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=10.0):
        super().__init__()
        B = torch.randn(input_dim, mapping_size // 2) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = x @ self.B * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FCN(nn.Module):
    def __init__(self, layers, uvp_mean=None, uvp_std=None,
                 fourier_mapping_size=None, fourier_scale=None, U_lid=2):
        super().__init__()
        self.U_lid = U_lid
        self.fourier_mapping = None
        if fourier_mapping_size and fourier_scale:
            self.fourier_mapping = FourierFeatureMapping(
                layers[0], fourier_mapping_size, fourier_scale)
            layers = list(layers)
            layers[0] = fourier_mapping_size

        self.activation = nn.GELU()
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )
        for lin in self.linears:
            nn.init.xavier_normal_(lin.weight, gain=1.0)
            nn.init.zeros_(lin.bias)

        if uvp_mean is not None and uvp_std is not None:
            self.register_buffer('uvp_mean',
                                 torch.tensor(uvp_mean, dtype=torch.float32))
            self.register_buffer('uvp_std',
                                 torch.tensor(uvp_std,  dtype=torch.float32))
        else:
            self.uvp_mean = self.uvp_std = None

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = x.to(next(self.parameters()).device).float()
        if self.fourier_mapping is not None:
            a = self.fourier_mapping(a)
        residual = None
        for i, lin in enumerate(self.linears[:-1]):
            a = self.activation(lin(a))
            if i == 2:
                residual = a
            elif i == 5 and residual is not None:
                a = a + residual
        return self.linears[-1](a)

    def denormalize(self, z):
        if self.uvp_mean is not None:
            return z * self.uvp_std + self.uvp_mean
        return z

    def predict(self, x):
        return self.denormalize(self.forward(x))


# ══════════════════════════════════════════════════════════════════════════════
#  Individual loss functions
# ══════════════════════════════════════════════════════════════════════════════

def _loss_pde(model, x_col, rho=1.0, nu=0.01):
    torch.set_grad_enabled(True)
    g = x_col.clone().detach().requires_grad_(True)
    out = model.predict(g)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def grad1(f):
        return autograd.grad(f, g, torch.ones_like(f), create_graph=True)[0]

    ug = grad1(u); u_x, u_y = ug[:, 0:1], ug[:, 1:2]
    vg = grad1(v); v_x, v_y = vg[:, 0:1], vg[:, 1:2]
    pg = grad1(p); p_x, p_y = pg[:, 0:1], pg[:, 1:2]
    u_xx = grad1(u_x)[:, 0:1]; u_yy = grad1(u_y)[:, 1:2]
    v_xx = grad1(v_x)[:, 0:1]; v_yy = grad1(v_y)[:, 1:2]

    mse = nn.MSELoss(reduction='mean')
    z   = lambda t: torch.zeros_like(t)
    cont  = u_x + v_y
    x_mom = u*u_x + v*u_y + p_x/rho - nu*(u_xx + u_yy)
    y_mom = u*v_x + v*v_y + p_y/rho - nu*(v_xx + v_yy)
    return mse(cont, z(cont)) + mse(x_mom, z(x_mom)) + mse(y_mom, z(y_mom))


def _loss_bc(model, x_bc, y_bc):
    pred = model.predict(x_bc)
    return nn.MSELoss(reduction='mean')(pred[:, 0:2], y_bc[:, 0:2])


def _loss_mse(model, x_data, y_data):
    return nn.MSELoss(reduction='mean')(model.forward(x_data), y_data)


def raw_losses(model, x_col, x_bc, y_bc, x_data, y_data, rho=1.0, nu=0.01):
    """Return (L_pde, L_bc, L_mse) as a detached numpy array."""
    with torch.enable_grad():             # ← 改成这个，不用 no_grad
        l_pde = _loss_pde(model, x_col, rho=rho, nu=nu)
        l_bc  = _loss_bc (model, x_bc,  y_bc)
        l_mse = _loss_mse(model, x_data, y_data)
    return np.array([l_pde.item(), l_bc.item(), l_mse.item()])


# ══════════════════════════════════════════════════════════════════════════════
#  Pilot training
# ══════════════════════════════════════════════════════════════════════════════

def pilot_train(arm_weights, data_tensors, device,
                uvp_mean, uvp_std, U_lid, L0,
                pilot_epochs=300, lr=5e-4, rho=1.0, nu=0.01):
    """
    Train a fresh model for `pilot_epochs` with the given arm weights.

    arm_weights : (λ_pde, λ_bc, λ_mse) — applied to normalised losses
    L0          : anchor losses from initialisation (shape (3,), numpy)
    Returns     : reward (float), L_tilde_final (numpy (3,)), model
    """
    lam = np.array(arm_weights, dtype=np.float32)
    lam = lam / lam.sum()                          # ensure simplex

    x_bc, y_bc, x_col, x_all, y_all = data_tensors

    model = FCN([2, 256, 256, 256, 256, 256, 256, 3],
                uvp_mean=uvp_mean, uvp_std=uvp_std,
                fourier_mapping_size=64, fourier_scale=3.0,
                U_lid=U_lid).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = pilot_epochs // 2

    for ep in range(pilot_epochs):
        pde_scale = min(1.0, ep / max(warmup, 1))
        opt.zero_grad()

        raw = raw_losses(model, x_col, x_bc, y_bc, x_all, y_all, rho=rho, nu=nu)
        L_tilde = raw / (L0 + 1e-12)

        # Recompute with graph for backward
        l_pde_t = _loss_pde(model, x_col, rho=rho, nu=nu)
        l_bc_t  = _loss_bc (model, x_bc,  y_bc)
        l_mse_t = _loss_mse(model, x_all, y_all)

        L_tilde_t = torch.stack([
            l_pde_t / (L0[0] + 1e-12),
            l_bc_t  / (L0[1] + 1e-12),
            l_mse_t / (L0[2] + 1e-12),
        ])
        lam_t = torch.tensor(
            [lam[0] * pde_scale, lam[1], lam[2]],
            dtype=torch.float32, device=device
        )
        loss = (lam_t * L_tilde_t).sum()
        loss.backward()
        opt.step()

    # Final evaluation
    raw_final = raw_losses(model, x_col, x_bc, y_bc,
                       x_all, y_all, rho=rho, nu=nu)

    L_tilde_final = raw_final / (L0 + 1e-12)

    # Minimax reward: negative worst-case normalised residual
    reward = -float(np.max(L_tilde_final))

    return reward, L_tilde_final, model


# ══════════════════════════════════════════════════════════════════════════════
#  Bandit algorithms
# ══════════════════════════════════════════════════════════════════════════════

class ArmPool:
    """Shared discrete set of arms for all bandit policies."""
    def __init__(self, pde_values, bc_values, mse_values):
        from itertools import product
        raw = list(product(pde_values, bc_values, mse_values))
        # Normalise to simplex
        self.arms = []
        for arm in raw:
            a = np.array(arm, dtype=np.float32)
            self.arms.append(tuple(a / a.sum()))
        self.K = len(self.arms)

    def __len__(self):
        return self.K


# ── 1. UCB ────────────────────────────────────────────────────────────────────

class UCBBandit:
    """
    UCB-1: select arm i* = argmax_i  Q(i) + sqrt(2 ln t / n_i)
    """
    name = "UCB"

    def __init__(self, pool: ArmPool):
        self.pool = pool
        self.K    = pool.K
        self.n    = np.zeros(self.K)          # pull counts
        self.Q    = np.zeros(self.K)          # empirical mean reward
        self.t    = 0
        self.history = []                     # (round, arm_idx, reward)

    def select(self):
        self.t += 1
        # Warm-up: pull each arm once
        for i in range(self.K):
            if self.n[i] == 0:
                return i
        ucb = self.Q + np.sqrt(2 * np.log(self.t) / (self.n + 1e-12))
        return int(np.argmax(ucb))

    def update(self, arm_idx, reward):
        self.n[arm_idx] += 1
        self.Q[arm_idx] += (reward - self.Q[arm_idx]) / self.n[arm_idx]
        self.history.append((self.t, arm_idx, reward))

    def best_arm(self):
        return int(np.argmax(self.Q))


# ── 2. Thompson Sampling ──────────────────────────────────────────────────────

class ThompsonSamplingBandit:
    """
    Normal–Normal conjugate model.
    Prior: reward ~ N(μ_0, σ_0²);  likelihood: N(μ, σ²)
    Posterior after n pulls: N(μ_n, σ_n²) where
        σ_n² = 1 / (1/σ_0² + n/σ²)
        μ_n  = σ_n² · (μ_0/σ_0² + Σr / σ²)
    At each round sample θ_i ~ posterior and pick argmax.
    """
    name = "Thompson Sampling"

    def __init__(self, pool: ArmPool, mu0=0.0, sigma0=1.0, sigma_lik=0.5):
        self.pool      = pool
        self.K         = pool.K
        self.mu0       = mu0
        self.sigma0_sq = sigma0 ** 2
        self.sigma_sq  = sigma_lik ** 2   # assumed observation noise
        # Sufficient statistics
        self.n         = np.zeros(self.K)
        self.sum_r     = np.zeros(self.K)
        self.t         = 0
        self.history   = []

    def _posterior(self, i):
        s2_n = 1.0 / (1.0 / self.sigma0_sq + self.n[i] / self.sigma_sq)
        mu_n = s2_n * (self.mu0 / self.sigma0_sq +
                       self.sum_r[i] / self.sigma_sq)
        return mu_n, s2_n

    def select(self):
        self.t += 1
        samples = np.zeros(self.K)
        for i in range(self.K):
            mu_n, s2_n = self._posterior(i)
            samples[i] = np.random.normal(mu_n, math.sqrt(s2_n))
        return int(np.argmax(samples))

    def update(self, arm_idx, reward):
        self.n[arm_idx]     += 1
        self.sum_r[arm_idx] += reward
        self.history.append((self.t, arm_idx, reward))

    def best_arm(self):
        means = np.array([self._posterior(i)[0] for i in range(self.K)])
        return int(np.argmax(means))


# ── 3. Posterior Sampling via Perturbed History (PSPH) ───────────────────────

class PSPHBandit:
    """
    Posterior Sampling via Perturbed History.

    We maintain a history of (arm_feature_vector, reward) pairs.
    At each round:
      1. Add Gaussian noise ~ N(0, 1/4) to every past reward.
      2. Solve the resulting ridge-regression linear system to get θ̂_*.
      3. Select arm i* = argmax_i  x_i · θ̂_*

    The arm feature vector x_i is the normalised weight tuple itself
    (dim=3), so θ̂_* ∈ R³ is an estimate of "which loss dimension matters".
    This simulates a posterior sample while handling the discrete,
    non-Gaussian reward noise — exactly as described in the Methods section.
    """
    name = "PSPH"

    def __init__(self, pool: ArmPool, noise_var=0.25, ridge=1.0):
        self.pool      = pool
        self.K         = pool.K
        self.noise_std = math.sqrt(noise_var)   # = 0.5  (σ of N(0,1/4))
        self.ridge     = ridge
        self.X_hist    = []    # list of feature vectors (length-3 tuples)
        self.r_hist    = []    # list of observed rewards
        self.t         = 0
        self.history   = []
        # Current parameter estimate
        self.theta_hat = np.zeros(3)

    def _refit(self):
        if len(self.X_hist) == 0:
            return
        X = np.array(self.X_hist)                         # (n, 3)
        r = np.array(self.r_hist)                         # (n,)
        # Add Gaussian noise to rewards
        r_perturbed = r + np.random.normal(0, self.noise_std, size=r.shape)
        # Ridge regression: θ̂ = (XᵀX + ridge·I)⁻¹ Xᵀ r_perturbed
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        b = X.T @ r_perturbed
        self.theta_hat = np.linalg.solve(A, b)

    def select(self):
        self.t += 1
        # Warm-up
        if self.t <= self.K:
            return (self.t - 1) % self.K
        self._refit()
        scores = np.array([
            np.dot(self.pool.arms[i], self.theta_hat)
            for i in range(self.K)
        ])
        return int(np.argmax(scores))

    def update(self, arm_idx, reward):
        self.X_hist.append(self.pool.arms[arm_idx])
        self.r_hist.append(reward)
        self.history.append((self.t, arm_idx, reward))

    def best_arm(self):
        if len(self.X_hist) == 0:
            return 0
        scores = np.array([
            np.dot(self.pool.arms[i], self.theta_hat)
            for i in range(self.K)
        ])
        return int(np.argmax(scores))


# ══════════════════════════════════════════════════════════════════════════════
#  Run one bandit experiment
# ══════════════════════════════════════════════════════════════════════════════

def run_bandit(bandit, pool, data_tensors, device, uvp_mean, uvp_std,
               U_lid, L0, total_rounds, pilot_epochs, lr, rho, nu):
    """
    Run `total_rounds` of select → pilot_train → update.
    Returns a log dict with per-round statistics.
    """
    log = {
        'algorithm':    bandit.name,
        'chosen_arm':   [],
        'reward':       [],
        'L_tilde_pde':  [],
        'L_tilde_bc':   [],
        'L_tilde_mse':  [],
        'worst_case':   [],   # max(L̃)  — the minimax objective
    }

    best_reward      = -np.inf
    best_arm_idx     = None
    best_model_state = None

    for rnd in range(total_rounds):
        arm_idx = bandit.select()
        arm_cfg = pool.arms[arm_idx]

        print(f"  [{bandit.name}] Round {rnd+1}/{total_rounds}  "
              f"arm={arm_idx}  λ=({arm_cfg[0]:.3f},{arm_cfg[1]:.3f},"
              f"{arm_cfg[2]:.3f})")

        reward, L_tilde, model = pilot_train(
            arm_cfg, data_tensors, device, uvp_mean, uvp_std,
            U_lid=U_lid, L0=L0,
            pilot_epochs=pilot_epochs, lr=lr, rho=rho, nu=nu
        )

        bandit.update(arm_idx, reward)

        log['chosen_arm' ].append(arm_idx)
        log['reward'     ].append(reward)
        log['L_tilde_pde'].append(L_tilde[0])
        log['L_tilde_bc' ].append(L_tilde[1])
        log['L_tilde_mse'].append(L_tilde[2])
        log['worst_case' ].append(float(np.max(L_tilde)))

        if reward > best_reward:
            best_reward      = reward
            best_arm_idx     = arm_idx
            best_model_state = copy.deepcopy(model.state_dict())

        print(f"           reward={reward:.4f}  "
              f"worst_case_L̃={np.max(L_tilde):.4f}")

    log['best_arm_idx']    = best_arm_idx
    log['best_arm_config'] = pool.arms[best_arm_idx]
    log['best_reward']     = best_reward
    log['best_model_state'] = best_model_state

    return log


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

ALGO_COLORS = {
    "UCB":               "#378ADD",
    "Thompson Sampling": "#1D9E75",
    "PSPH":              "#D85A30",
}


def plot_comparison(logs, save_dir):
    """
    3-panel figure mirroring the original diagnostics layout.

    Panel (a) — cumulative best reward per algorithm
    Panel (b) — worst-case normalised loss  max(L̃)  per round
    Panel (c) — chosen arm index per round (scatter)
    """
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for log in logs:
        name   = log['algorithm']
        color  = ALGO_COLORS.get(name, "gray")
        rounds = np.arange(1, len(log['reward']) + 1)

        # Cumulative best reward
        cum_best = np.maximum.accumulate(log['reward'])
        ax1.plot(rounds, cum_best, color=color, linewidth=1.8, label=name)

        # Worst-case L̃ per round
        ax2.plot(rounds, log['worst_case'], color=color,
                 linewidth=1.2, alpha=0.85, label=name)

        # Chosen arm scatter
        ax3.scatter(rounds, log['chosen_arm'], color=color,
                    s=20, alpha=0.7, label=name)

    ax1.set_xlabel("Round", fontsize=11)
    ax1.set_ylabel("Cumulative best reward", fontsize=10)
    ax1.set_title("(a) Cumulative best reward", fontsize=11)
    ax1.legend(fontsize=9); ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.set_xlabel("Round", fontsize=11)
    ax2.set_ylabel(r"$\max(\tilde{L})$  [worst-case residual]", fontsize=10)
    ax2.set_title("(b) Worst-case normalised loss per round", fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, linestyle="--", alpha=0.4)

    ax3.set_xlabel("Round", fontsize=11)
    ax3.set_ylabel("Arm index", fontsize=10)
    ax3.set_title("(c) Arm selection history", fontsize=11)
    ax3.legend(fontsize=9); ax3.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(save_dir, "mab_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved: {path}")


def plot_per_algorithm(log, save_dir):
    """
    Per-algorithm 3-panel plot (mirrors your original panel-3 style).

    Panel (a) — normalised losses L̃_pde, L̃_bc, L̃_mse per round
    Panel (b) — reward per round
    Panel (c) — worst-case L̃ per round
    """
    name   = log['algorithm']
    color  = ALGO_COLORS.get(name, "gray")
    rounds = np.arange(1, len(log['reward']) + 1)

    SMOOTH_W = max(3, len(rounds) // 10)
    kernel   = np.ones(SMOOTH_W) / SMOOTH_W

    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Panel (a): normalised losses
    ax1.plot(rounds, log['L_tilde_pde'], color="#D85A30",
             linewidth=1.2, label=r"$\tilde{L}_{pde}$")
    ax1.plot(rounds, log['L_tilde_bc'],  color="#378ADD",
             linewidth=1.2, label=r"$\tilde{L}_{bc}$")
    ax1.plot(rounds, log['L_tilde_mse'], color="#1D9E75",
             linewidth=1.2, label=r"$\tilde{L}_{mse}$")
    ax1.axhline(1.0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
    ax1.set_yscale("log")
    ax1.set_xlabel("Round", fontsize=11)
    ax1.set_ylabel("Normalised loss  L̃", fontsize=10)
    ax1.set_title(f"(a) Normalised losses — {name}", fontsize=11)
    ax1.legend(fontsize=9); ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel (b): reward per round
    reward_smooth = np.convolve(log['reward'], kernel, mode='same')
    ax2.plot(rounds, log['reward'],    color="lightgray",
             linewidth=0.7, alpha=0.7, label="raw")
    ax2.plot(rounds, reward_smooth,    color=color,
             linewidth=1.8, label=f"smoothed (w={SMOOTH_W})")
    ax2.set_xlabel("Round", fontsize=11)
    ax2.set_ylabel("Reward  (−max L̃)", fontsize=10)
    ax2.set_title(f"(b) Reward trajectory — {name}", fontsize=11)
    ax2.legend(fontsize=9); ax2.grid(True, linestyle="--", alpha=0.4)

    # Panel (c): worst-case L̃
    wc_smooth = np.convolve(log['worst_case'], kernel, mode='same')
    ax3.plot(rounds, log['worst_case'], color="lightgray",
             linewidth=0.7, alpha=0.7, label="raw")
    ax3.plot(rounds, wc_smooth,         color=color,
             linewidth=1.8, label=f"smoothed (w={SMOOTH_W})")
    ax3.set_xlabel("Round", fontsize=11)
    ax3.set_ylabel(r"$\max(\tilde{L})$", fontsize=10)
    ax3.set_title(f"(c) Worst-case residual — {name}", fontsize=11)
    ax3.legend(fontsize=9); ax3.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    path  = os.path.join(save_dir, f"mab_{fname}_diagnostics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-algorithm plot saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_txt(path):
    print(f"Loading {path}")
    return np.loadtxt(path)


def split_boundary_interior(data, tol=1e-6):
    x    = torch.tensor(data[:, 0:2], dtype=torch.float32)
    y    = torch.tensor(data[:, 2:5], dtype=torch.float32)
    mask = ((x[:, 0] < tol) | (x[:, 0] > 1-tol) |
            (x[:, 1] < tol) | (x[:, 1] > 1-tol))
    return x[mask], y[mask], x[~mask]


def get_anchor_losses(uvp_mean, uvp_std, U_lid, data_tensors,
                      device, rho, nu):
    """Compute L0 = losses at random initialisation."""
    x_bc, y_bc, x_col, x_all, y_all = data_tensors
    probe = FCN([2, 256, 256, 256, 256, 256, 256, 3],
                uvp_mean=uvp_mean, uvp_std=uvp_std,
                fourier_mapping_size=64, fourier_scale=3.0,
                U_lid=U_lid).to(device)
    L0 = raw_losses(probe, x_col, x_bc, y_bc, x_all, y_all, rho=rho, nu=nu)
    print(f"Anchor L0:  pde={L0[0]:.4e}  bc={L0[1]:.4e}  mse={L0[2]:.4e}")
    return L0


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Armed Bandit PINN — UCB / TS / PSPH")
    parser.add_argument('--Re',   type=int,   default=1000)
    parser.add_argument('--U',    type=int,   default=10)
    parser.add_argument('--num',  type=int,   default=81)
    parser.add_argument('--dataset_base_path', type=str,
                        default='/projects/bfth/rhe4/PINN/data_pinn')
    parser.add_argument('--output_base_path',  type=str,
                        default='/projects/bfth/rhe4/PINN/results_mab_v2')
    parser.add_argument('--pilot_epochs', type=int,   default=300)
    parser.add_argument('--total_rounds', type=int,   default=40)
    parser.add_argument('--lr',           type=float, default=5e-4)
    parser.add_argument('--algorithm',    type=str,   default='all',
                        help="'ucb' | 'ts' | 'psph' | 'all'")
    # Arm grid
    parser.add_argument('--pde_values',  type=str, default='0.1,1,5,10,20')
    parser.add_argument('--bc_values',   type=str, default='10,40,60,80')
    parser.add_argument('--mse_values',  type=str, default='1,5,10,20')
    # PSPH hyper-params
    parser.add_argument('--psph_noise_var', type=float, default=0.25,
                        help='Variance of perturbation noise (default: 1/4)')
    parser.add_argument('--psph_ridge',     type=float, default=1.0)
    # TS hyper-params
    parser.add_argument('--ts_sigma0',   type=float, default=1.0)
    parser.add_argument('--ts_sigma_lik',type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_base_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── data ─────────────────────────────────────────────────────────────────
    subfolder = f"U{args.U}"
    fname     = f"processed_Re{args.Re}_U{args.U}_NX{args.num}.txt"
    data      = load_txt(os.path.join(args.dataset_base_path, subfolder, fname))

    uvp_mean          = np.mean(data[:, 2:5], axis=0)
    uvp_std           = np.std( data[:, 2:5], axis=0) + 1e-12
    data_norm         = data.copy()
    data_norm[:, 2:5] = (data[:, 2:5] - uvp_mean) / uvp_std
    nu                = 1.0 / args.Re

    x_bc, y_bc, x_col = split_boundary_interior(data_norm)
    x_all = torch.tensor(data_norm[:, 0:2], dtype=torch.float32)
    y_all = torch.tensor(data_norm[:, 2:5], dtype=torch.float32)

    x_bc  = x_bc.to(device);  y_bc  = y_bc.to(device)
    x_col = x_col.to(device)
    x_all = x_all.to(device);  y_all = y_all.to(device)

    data_tensors = (x_bc, y_bc, x_col, x_all, y_all)
    print(f"BC pts: {x_bc.shape[0]}  Collocation: {x_col.shape[0]}  "
          f"Total: {x_all.shape[0]}\n")

    # ── anchor losses ─────────────────────────────────────────────────────────
    L0 = get_anchor_losses(uvp_mean, uvp_std, args.U,
                           data_tensors, device, rho=1.0, nu=nu)

    # ── arm pool ──────────────────────────────────────────────────────────────
    pde_vals = [float(v) for v in args.pde_values.split(',')]
    bc_vals  = [float(v) for v in args.bc_values.split(',')]
    mse_vals = [float(v) for v in args.mse_values.split(',')]
    pool     = ArmPool(pde_vals, bc_vals, mse_vals)
    print(f"Arm pool: {pool.K} arms  |  "
          f"Rounds: {args.total_rounds}  |  "
          f"Pilot epochs: {args.pilot_epochs}\n")

    # ── select algorithms ─────────────────────────────────────────────────────
    algo = args.algorithm.lower()
    bandits = []
    if algo in ('ucb',  'all'):
        bandits.append(UCBBandit(pool))
    if algo in ('ts',   'all'):
        bandits.append(ThompsonSamplingBandit(
            pool,
            sigma0=args.ts_sigma0,
            sigma_lik=args.ts_sigma_lik
        ))
    if algo in ('psph', 'all'):
        bandits.append(PSPHBandit(
            pool,
            noise_var=args.psph_noise_var,
            ridge=args.psph_ridge
        ))

    # ── run experiments ───────────────────────────────────────────────────────
    all_logs = []
    for bandit in bandits:
        print(f"{'='*60}")
        print(f"Running: {bandit.name}")
        print(f"{'='*60}")
        log = run_bandit(
            bandit, pool, data_tensors, device,
            uvp_mean, uvp_std, U_lid=args.U, L0=L0,
            total_rounds=args.total_rounds,
            pilot_epochs=args.pilot_epochs,
            lr=args.lr, rho=1.0, nu=nu
        )
        all_logs.append(log)

        # Save best model
        best_path = os.path.join(
            args.output_base_path,
            f"best_model_{bandit.name.lower().replace(' ','_')}.pth"
        )
        torch.save(log['best_model_state'], best_path)
        print(f"Best model saved: {best_path}")

        # Per-algorithm plot
        plot_per_algorithm(log, args.output_base_path)

    # ── comparison plot ───────────────────────────────────────────────────────
    if len(all_logs) > 1:
        plot_comparison(all_logs, args.output_base_path)

    # ── leaderboard ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("LEADERBOARD")
    print(f"{'='*60}")
    for log in sorted(all_logs, key=lambda l: l['best_reward'], reverse=True):
        arm  = log['best_arm_config']
        print(f"  {log['algorithm']:<25}  "
              f"best_reward={log['best_reward']:.4f}  "
              f"arm=(pde={arm[0]:.3f}, bc={arm[1]:.3f}, mse={arm[2]:.3f})")

    # ── save JSON summary ─────────────────────────────────────────────────────
    summary = []
    for log in all_logs:
        summary.append({
            'algorithm':      log['algorithm'],
            'best_reward':    log['best_reward'],
            'best_arm_idx':   log['best_arm_idx'],
            'best_arm_config': list(log['best_arm_config']),
            'reward_history': log['reward'],
            'worst_case':     log['worst_case'],
        })
    with open(os.path.join(args.output_base_path, 'mab_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_base_path}/mab_results.json")


if __name__ == '__main__':
    main()
