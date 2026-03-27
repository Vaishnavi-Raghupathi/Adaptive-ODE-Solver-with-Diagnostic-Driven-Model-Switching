"""Data loading and preprocessing utilities."""

import numpy as np
from scipy.integrate import solve_ivp


def generate_lorenz_data(t_span, y0, num_points, sigma=10.0, rho=28.0, 
                         beta=8.0/3.0, noise_std=0.0):
    """
    Generate Lorenz system trajectories.
    
    The Lorenz system is defined by:
        dx/dt = sigma(y - x)
        dy/dt = x(rho - z) - y
        dz/dt = xy - beta*z
    
    Parameters
    ----------
    t_span : tuple
        Time interval (t0, tf)
    y0 : array-like
        Initial conditions [x0, y0, z0]
    num_points : int
        Number of time points to evaluate
    sigma : float
        Lorenz parameter (default: 10.0)
    rho : float
        Lorenz parameter (default: 28.0)
    beta : float
        Lorenz parameter (default: 8/3)
    noise_std : float
        Standard deviation of Gaussian noise to add (default: 0.0)
        
    Returns
    -------
    t : ndarray
        Time points (shape: num_points)
    y : ndarray
        State values (shape: num_points, 3)
    """
    def lorenz(t, state):
        """Lorenz system ODE function."""
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]
    
    # Solve the ODE
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    solution = solve_ivp(lorenz, t_span, y0, t_eval=t_eval, method='RK45')
    
    t = solution.t
    y = solution.y.T  # Transpose to get shape (num_points, 3)
    
    # Add noise if requested
    if noise_std > 0.0:
        y = y + np.random.normal(0, noise_std, y.shape)
    
    return t, y


def load_data(filepath, delimiter=","):
    """
    Load data from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
    delimiter : str
        Delimiter used in the file (default: ",")
        
    Returns
    -------
    data : ndarray
        Loaded data as numpy array
    """
    data = np.loadtxt(filepath, delimiter=delimiter)
    return data


def normalize_data(data, axis=0):
    """
    Normalize data to zero mean and unit variance.
    
    Parameters
    ----------
    data : ndarray
        Input data
    axis : int
        Axis along which to normalize
        
    Returns
    -------
    normalized : ndarray
        Normalized data
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    normalized = (data - mean) / (std + 1e-8)
    return normalized
