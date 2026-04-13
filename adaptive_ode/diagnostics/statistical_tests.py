"""Statistical tests for residual analysis and error diagnostics."""

import numpy as np
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


def _is_near_constant(x, tol=1e-12):
    """Return True when a signal has negligible variation."""
    x = np.asarray(x)
    if x.size == 0:
        return True
    return float(np.std(x)) < tol


def _residuals_are_negligible(residuals, tol=1e-12):
    """Return True when residual field is effectively zero/constant."""
    return _is_near_constant(np.asarray(residuals).flatten(), tol=tol)


def compute_residual_statistics(residuals):
    """
    Compute statistical metrics for residuals.
    
    Parameters
    ----------
    residuals : ndarray
        Residuals (shape: n_points, n_dims or n_points)
        
    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'mean': Mean of residuals
        - 'std': Standard deviation
        - 'max_abs': Maximum absolute value
        - 'l2_norm': L2 norm
    """
    stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'max_abs': np.max(np.abs(residuals)),
        'l2_norm': np.linalg.norm(residuals)
    }
    return stats


def compute_error_metrics(y_true, y_pred):
    """
    Compute error metrics between true and predicted values.
    
    Parameters
    ----------
    y_true : ndarray
        True values
    y_pred : ndarray
        Predicted values
        
    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'mae': Mean absolute error
        - 'mse': Mean squared error
        - 'rmse': Root mean squared error
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }
    return metrics


def test_heteroscedasticity(residuals, y_pred, alpha=0.05):
    """
    Test for heteroscedasticity using Breusch-Pagan test.
    
    Tests if the variance of residuals depends on predicted values.
    
    Parameters
    ----------
    residuals : ndarray
        Residuals (shape: n_points, n_dims or n_points)
    y_pred : ndarray
        Predicted values (same shape as residuals)
    alpha : float
        Significance level (default: 0.05)
        
    Returns
    -------
    has_heteroscedasticity : bool
        True if heteroscedasticity detected (p-value < alpha)
    """
    # Perfect/near-perfect fit: no variance pattern to diagnose
    if _residuals_are_negligible(residuals):
        return False

    # Flatten arrays if multi-dimensional
    residuals_flat = residuals.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Squared residuals
    squared_residuals = residuals_flat ** 2
    
    # Simple linear regression: squared_residuals ~ y_pred
    # Using least squares: y = a*x + b
    n = len(y_pred_flat)
    x = y_pred_flat
    y = squared_residuals
    
    # Add intercept
    X = np.column_stack([np.ones(n), x])
    
    # Fit model
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_fitted = X @ beta
        residuals_model = y - y_fitted
        
        # Sum of squared residuals
        ss_residual = np.sum(residuals_model ** 2)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        
        # R-squared
        r_squared = 1 - (ss_residual / (ss_total + 1e-10))
        
        # Test statistic
        bp_stat = n * r_squared
        
        # Chi-square test (1 degree of freedom)
        p_value = 1 - sp_stats.chi2.cdf(bp_stat, df=1)
        
        return p_value < alpha
    except:
        # If regression fails, assume no heteroscedasticity
        return False


def test_autocorrelation(residuals, alpha=0.05):
    """
    Test for autocorrelation using Ljung-Box test.
    
    Tests if residuals are serially correlated.
    
    Parameters
    ----------
    residuals : ndarray
        Residuals (shape: n_points, n_dims or n_points)
    alpha : float
        Significance level (default: 0.05)
        
    Returns
    -------
    has_autocorrelation : bool
        True if autocorrelation detected in any dimension
    """
    # Constant/near-zero residuals: treat as no actionable autocorrelation
    if _residuals_are_negligible(residuals):
        return False

    # Handle 1D case
    if residuals.ndim == 1:
        residuals_2d = residuals.reshape(-1, 1)
    else:
        residuals_2d = residuals
    
    n_dims = residuals_2d.shape[1]
    
    # Test each dimension
    for d in range(n_dims):
        res = residuals_2d[:, d]

        if _is_near_constant(res):
            continue
        
        # Ljung-Box test (lags=10)
        try:
            max_lag = min(10, max(1, len(res) - 1))
            lb_result = acorr_ljungbox(res, lags=[max_lag], return_df=True)
            p_value = float(lb_result["lb_pvalue"].iloc[-1])
            if p_value < alpha:
                return True
        except:
            pass
    
    return False


def test_stationarity(residuals, alpha=0.05):
    """
    Test for stationarity using Augmented Dickey-Fuller (ADF) test.
    
    Tests if residuals have a unit root (non-stationary).
    
    Parameters
    ----------
    residuals : ndarray
        Residuals (shape: n_points, n_dims or n_points)
    alpha : float
        Significance level (default: 0.05)
        
    Returns
    -------
    is_non_stationary : bool
        True if non-stationary (fails stationarity test)
    """
    # Constant/near-zero residuals are effectively stationary for diagnostics.
    if _residuals_are_negligible(residuals):
        return False

    # Handle 1D case
    if residuals.ndim == 1:
        residuals_2d = residuals.reshape(-1, 1)
    else:
        residuals_2d = residuals
    
    n_dims = residuals_2d.shape[1]
    non_stationary_count = 0
    
    # Test each dimension
    for d in range(n_dims):
        res = residuals_2d[:, d]

        if _is_near_constant(res):
            # Constant residual stream is treated as stationary.
            continue
        
        try:
            p_value = _adf_test(res)
            if p_value >= alpha:  # Fail to reject H0 (non-stationary)
                non_stationary_count += 1
        except:
            non_stationary_count += 1
    
    # True if any dimension is non-stationary
    return non_stationary_count > 0


def _adf_test(residuals):
    """
    Augmented Dickey-Fuller test wrapper using statsmodels.
    
    Parameters
    ----------
    residuals : ndarray
        1D residuals array
        
    Returns
    -------
    p_value : float
        P-value for ADF test
    """
    p_value = adfuller(residuals, autolag='AIC')[1]

    return p_value


def test_state_dependence(residuals, y_pred, alpha=0.05):
    """
    Test for state-dependent errors.
    
    Tests if absolute residuals depend on predicted state values
    using linear regression.
    
    Parameters
    ----------
    residuals : ndarray
        Residuals (shape: n_points, n_dims or n_points)
    y_pred : ndarray
        Predicted values (same shape as residuals)
    alpha : float
        Significance level (default: 0.05)
        
    Returns
    -------
    state_dependent : bool
        True if errors show state-dependence (R² > 0.1)
    """
    # Perfect/near-perfect fit: no state dependence in residual magnitude
    if _residuals_are_negligible(residuals):
        return False

    # Flatten arrays
    residuals_flat = residuals.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Absolute residuals
    abs_residuals = np.abs(residuals_flat)
    
    # Linear regression: |residual| ~ y_pred
    n = len(y_pred_flat)
    x = y_pred_flat
    y = abs_residuals
    
    # Add intercept
    X = np.column_stack([np.ones(n), x])
    
    # Fit model
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_fitted = X @ beta
        residuals_model = y - y_fitted
        
        # R-squared
        ss_residual = np.sum(residuals_model ** 2)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_residual / (ss_total + 1e-10))
        
        # Consider state-dependent if R² > 0.1
        return r_squared > 0.1
    except:
        # If regression fails, assume not state-dependent
        return False
