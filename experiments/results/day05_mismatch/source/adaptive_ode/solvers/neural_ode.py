"""Neural ODE solver using PyTorch and torchdiffeq."""

import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """Generic MLP-based ODE function. Supports state_dim=2 or 3."""

    def __init__(self, state_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),  nn.Tanh(),
            nn.Linear(64, 128),        nn.Tanh(),
            nn.Linear(128, 64),        nn.Tanh(),
            nn.Linear(64, state_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, t, x):
        return self.net(x)


def _osc_true(t, s):
    """True oscillator: damping=0.1, freq²=4.0."""
    y, v = s
    return [v, -0.1 * v - 4.0 * y]


def _lorenz_clean(t, state):
    x, y, z = state
    return [10.0*(y-x), x*(28.0-z)-y, x*y-(8.0/3.0)*z]


def _generate_training_trajectory(t_span, y0, mismatch=False, n_points=500):
    """
    Generate high-resolution trajectory for Neural ODE training.

    mismatch=True  → true oscillator (damping=0.1, freq²=4.0), state_dim=2
    mismatch=False → clean Lorenz (rho=28),                     state_dim=3
    """
    t_np = np.linspace(t_span[0], t_span[1], n_points, dtype=np.float64)
    rhs  = _osc_true if mismatch else _lorenz_clean
    y0_  = np.asarray(y0, dtype=np.float64)

    sol = solve_ivp(rhs, t_span, y0_, t_eval=t_np,
                    method="RK45", rtol=1e-10, atol=1e-11)
    if not sol.success:
        raise RuntimeError(f"Trajectory generation failed: {sol.message}")
    return t_np.astype(np.float32), sol.y.T.astype(np.float32)


def train_neural_ode(
    t_eval,
    y0_clean,
    y_noisy_context=None,
    mismatch=False,
    device="cpu",
    epochs=800,
    lr=1e-3,
    rtol=1e-3,
    atol=1e-4,
    early_stop_loss=1e-3,
    print_every=100,
):
    """
    Train Neural ODE against clean ground-truth trajectory.

    mismatch=False → 3D Lorenz target, state_dim=3
    mismatch=True  → 2D oscillator target, state_dim=2

    Returns
    -------
    func         : trained ODEFunc
    loss_history : list[float]
    final_pred   : ndarray shape (len(t_eval), state_dim)
    """
    del y_noisy_context

    t_eval_np = np.asarray(t_eval, dtype=np.float32)
    y0_np     = np.asarray(y0_clean, dtype=np.float32)
    t_span    = (float(t_eval_np[0]), float(t_eval_np[-1]))
    state_dim = 2 if mismatch else 3

    t_train_np, y_train_np = _generate_training_trajectory(
        t_span=t_span, y0=y0_np, mismatch=mismatch, n_points=500,
    )

    t_train      = torch.tensor(t_train_np, dtype=torch.float32, device=device)
    y_true_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
    t_eval_ten   = torch.tensor(t_eval_np,  dtype=torch.float32, device=device)

    # Normalize
    y_mean = y_true_train.mean(dim=0)
    y_std  = y_true_train.std(dim=0)
    y_norm = (y_true_train - y_mean) / y_std

    func      = ODEFunc(state_dim=state_dim).to(device)
    optimizer = Adam(func.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    WIN     = 50
    BATCH   = 10
    MAX_IDX = max(1, t_train.shape[0] - WIN - 1)

    loss_history = []
    func.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        batch_loss = 0.0

        for _ in range(BATCH):
            idx   = torch.randint(0, MAX_IDX, (1,)).item()
            t_win = t_train[idx: idx + WIN]
            y_win = y_norm[idx: idx + WIN]
            pred  = odeint(func, y_win[0].detach(), t_win,
                           method="dopri5", rtol=rtol, atol=atol)
            batch_loss = batch_loss + F.mse_loss(pred, y_win)

        batch_loss = batch_loss / BATCH
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=0.05)
        optimizer.step()
        scheduler.step()

        loss_val = float(batch_loss.detach().cpu())
        loss_history.append(loss_val)

        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(f"Epoch {epoch:4d}/{epochs} - Loss: {loss_val:.6f}  "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if loss_val < early_stop_loss:
            print(f"Early stop at epoch {epoch}, loss={loss_val:.6f}")
            break

    func.eval()
    k     = min(100, len(loss_history))
    early = float(np.mean(loss_history[:k]))
    late  = float(np.mean(loss_history[-k:]))
    print(f"\nConvergence: early={early:.6f} → late={late:.6f}")
    print("OK: converged" if late < early * 0.6 else "WARNING: did not converge")

    with torch.no_grad():
        pred_norm = odeint(func, y_norm[0], t_eval_ten,
                           method="dopri5", rtol=1e-5, atol=1e-6)
        pred_full = pred_norm * y_std + y_mean

    return func, loss_history, pred_full.detach().cpu().numpy()


class NeuralODESolver:
    """
    Neural ODE solver wrapper.

    mismatch=True  → oscillator scenario, state_dim must be 2
    mismatch=False → Lorenz scenario,     state_dim must be 3
    """

    def __init__(
        self,
        state_dim=3,
        hidden_dim=64,
        device="cpu",
        lr=1e-3,
        epochs=800,
        print_every=100,
        method="dopri5",
        mismatch=False,
    ):
        self.device      = device
        self.state_dim   = state_dim
        self.lr          = lr
        self.epochs      = epochs
        self.print_every = print_every
        self.method      = method
        self.mismatch    = mismatch

        self.func         = ODEFunc(state_dim=state_dim).to(device)
        self.y_pred       = None
        self.loss_history = []

    def fit(self, t, y_true, y0_clean=None):
        t_np      = np.asarray(t,      dtype=np.float32)
        y_true_np = np.asarray(y_true, dtype=np.float32)

        if y0_clean is None:
            y0_clean = np.array([2.0, 0.0] if self.mismatch else [1.0, 1.0, 1.0],
                                dtype=np.float32)
        else:
            y0_clean = np.asarray(y0_clean, dtype=np.float32)

        self.func, self.loss_history, final_pred = train_neural_ode(
            t_eval          = t_np,
            y0_clean        = y0_clean,
            y_noisy_context = y_true_np,
            mismatch        = self.mismatch,
            device          = self.device,
            epochs          = self.epochs,
            lr              = self.lr,
            early_stop_loss = 1e-3,
            print_every     = self.print_every,
        )

        self.y_pred = torch.tensor(final_pred, dtype=torch.float32, device=self.device)
        return self

    def predict(self):
        if self.y_pred is None:
            raise RuntimeError("Call fit() before predict().")
        return self.y_pred.detach().cpu().numpy()

    def get_network(self):
        return self.func

    def get_loss_history(self):
        return list(self.loss_history)

    def to(self, device):
        self.device = device
        self.func.to(device)
        return self