"""Classical ODE solvers (RK45, BDF, etc.)."""

import numpy as np
from scipy.integrate import solve_ivp


class ClassicalSolver:
    """
    Classical ODE solver using scipy's RK45 integrator.
    
    This solver fits observed trajectories and provides predictions
    by solving the ODE forward from the initial condition.
    """
    
    def __init__(self, ode_func):
        """
        Initialize the solver with an ODE function.
        
        Parameters
        ----------
        ode_func : callable
            ODE system: dy/dt = ode_func(t, y)
        """
        self.ode_func = ode_func
        self.y_pred = None
        self.t = None
    
    def fit(self, t, y_true):
        """
        Fit the solver by solving the ODE forward from initial condition.
        
        Parameters
        ----------
        t : ndarray
            Time points (shape: n_points)
        y_true : ndarray
            True trajectory (shape: n_points, n_dims)
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        # Extract initial condition from first point
        y0 = y_true[0]
        t_span = (t[0], t[-1])
        
        # Solve ODE forward
        solution = solve_ivp(
            self.ode_func, 
            t_span, 
            y0, 
            t_eval=t, 
            method='RK45'
        )
        
        # Store prediction
        self.t = solution.t
        self.y_pred = solution.y.T  # Shape: (n_points, n_dims)
        
        return self
    
    def predict(self):
        """
        Return the predicted trajectory.
        
        Returns
        -------
        y_pred : ndarray
            Predicted trajectory (shape: n_points, n_dims)
            
        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if self.y_pred is None:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        return self.y_pred
    
    def compute_residuals(self, y_true):
        """
        Compute residuals between true and predicted trajectories.
        
        Parameters
        ----------
        y_true : ndarray
            True trajectory (shape: n_points, n_dims)
            
        Returns
        -------
        residuals : ndarray
            Residuals: y_true - y_pred (shape: n_points, n_dims)
            
        Raises
        ------
        RuntimeError
            If fit() has not been called yet
        """
        if self.y_pred is None:
            raise RuntimeError("Model must be fitted before computing residuals. Call fit() first.")
        
        residuals = y_true - self.y_pred
        return residuals


def solve_ode(func, t_span, y0, method="RK45", **kwargs):
    """
    Solve an ODE using scipy's solve_ivp wrapper.
    
    Parameters
    ----------
    func : callable
        ODE system: dy/dt = func(t, y)
    t_span : tuple
        Integration interval (t0, tf)
    y0 : array-like
        Initial condition
    method : str
        Integration method (default: RK45)
    **kwargs : dict
        Additional arguments to pass to solve_ivp
        
    Returns
    -------
    solution : scipy OdeSolution
        Solution object with t and y attributes
    """
    solution = solve_ivp(func, t_span, y0, method=method, **kwargs)
    return solution
