"""Neural ODE solver using PyTorch and torchdiffeq."""

import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """
    Neural network to represent ODE dynamics.
    
    Learns the vector field dy/dt = f(t, y) where y is the state.
    Simple architecture: Linear → Tanh → Linear → Tanh → Linear
    """
    
    def __init__(self, state_dim, hidden_dim=64):
        """
        Initialize ODE function network.
        
        Parameters
        ----------
        state_dim : int
            Dimension of state vector (e.g., 3 for Lorenz system)
        hidden_dim : int
            Hidden layer dimension (default: 64)
        """
        super(ODEFunc, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, t, x):
        """
        Compute dx/dt for state x at time t.
        
        Parameters
        ----------
        t : tensor
            Time (required by torchdiffeq, may not be used)
        x : tensor
            State vector (shape: [..., state_dim])
            
        Returns
        -------
        dxdt : tensor
            Derivative dx/dt (same shape as x)
        """
        return self.net(x)


class NeuralODESolver:
    """
    Neural ODE solver that learns dynamics from data.
    
    Trains a neural network to model ODE dynamics and can solve
    the learned ODE forward in time.
    """
    
    def __init__(
        self,
        state_dim,
    hidden_dim=64,
        device='cpu',
        lr=0.001,
    epochs=600,
        print_every=25,
        method='dopri5'
    ):
        """
        Initialize Neural ODE solver.
        
        Parameters
        ----------
        state_dim : int
            Dimension of state space
        hidden_dim : int
            Hidden layer dimension (default: 64)
        device : str
            Computation device ('cpu' or 'cuda')
        lr : float
            Learning rate for Adam optimizer (default: 0.001)
        epochs : int
            Number of training epochs (default: 600)
        print_every : int
            Print loss every N epochs (default: 25)
        method : str
            ODE integration method for torchdiffeq (default: 'dopri5')
        """
        self.device = device
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.print_every = print_every
        self.method = method

        self.func = ODEFunc(state_dim, hidden_dim).to(device)
        self.y_pred = None
        self.t_train = None
        self.y_mean = None
        self.y_std = None

    def fit(self, t, y_true):
        """
        Train Neural ODE parameters on observed trajectories.

        Parameters
        ----------
        t : tensor or ndarray
            Time points (shape: n_points)
        y_true : tensor or ndarray
            True trajectory (shape: n_points, state_dim)

        Returns
        -------
        self
            Returns self for method chaining
        """
        # Convert to tensors
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32)

        # Handle 1D case
        if y_true.ndim == 1:
            y_true = y_true.unsqueeze(-1)

        if t.shape[0] != y_true.shape[0]:
            raise ValueError("t and y_true must have the same number of time points")

        if y_true.shape[1] != self.state_dim:
            raise ValueError("y_true second dimension must match state_dim")

        t = t.to(self.device)
        y_true = y_true.to(self.device)

        # Normalize targets for stable training
        self.y_mean = torch.mean(y_true, dim=0, keepdim=True)
        self.y_std = torch.std(y_true, dim=0, keepdim=True) + 1e-8
        y_true_norm = (y_true - self.y_mean) / self.y_std

        # Reinitialize ODE function before training
        self.func = ODEFunc(self.state_dim, self.hidden_dim).to(self.device)

        y0 = y_true_norm[0]
        optimizer = torch.optim.Adam(self.func.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.func.train()
        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()

            y_pred_norm = odeint(self.func, y0, t, method=self.method)
            loss = criterion(y_pred_norm, y_true_norm)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.func.parameters(), max_norm=1.0)
            optimizer.step()

            if epoch == 1 or epoch % self.print_every == 0 or epoch == self.epochs:
                print(f"Epoch {epoch}/{self.epochs} - Loss: {loss.item():.6f}")

        self.func.eval()
        with torch.no_grad():
            y_pred_norm = odeint(self.func, y0, t, method=self.method)
            self.y_pred = y_pred_norm * self.y_std + self.y_mean
            self.t_train = t

        return self

    def predict(self):
        """
        Return trained trajectory prediction as a numpy array.

        Returns
        -------
        y_pred : ndarray
            Predicted trajectory (shape: n_points, state_dim)

        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if self.y_pred is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        return self.y_pred.detach().cpu().numpy()
    
    def solve(self, t, y0, method='dopri5'):
        """
        Solve the learned ODE forward in time.
        
        Parameters
        ----------
        t : tensor or ndarray
            Time points (shape: n_points)
        y0 : tensor or ndarray
            Initial condition (shape: state_dim)
        method : str
            ODE solver method ('dopri', 'adams', etc.)
            
        Returns
        -------
        y : tensor
            Solution at time points (shape: n_points, state_dim)
        """
        if not isinstance(y0, torch.Tensor):
            y0 = torch.tensor(y0, dtype=torch.float32, device=self.device)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            y = odeint(self.func, y0, t, method=method)
        
        return y
    
    def get_network(self):
        """
        Get the underlying neural network.
        
        Returns
        -------
        func : nn.Module
            The ODE function network
        """
        return self.func
    
    def to(self, device):
        """
        Move model to device.
        
        Parameters
        ----------
        device : str
            Target device ('cpu' or 'cuda')
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        self.device = device
        self.func.to(device)
        return self
