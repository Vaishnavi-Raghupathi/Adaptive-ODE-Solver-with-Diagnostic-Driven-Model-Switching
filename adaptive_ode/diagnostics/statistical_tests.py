"""Statistical tests for residual analysis and error diagnostics."""

import numpy as np
from scipy import stats as sp_stats


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
    # Handle 1D case
    if residuals.ndim == 1:
        residuals_2d = residuals.reshape(-1, 1)
    else:
        residuals_2d = residuals
    
    n_dims = residuals_2d.shape[1]
    
    # Test each dimension
    for d in range(n_dims):
        res = residuals_2d[:, d]
        
        # Ljung-Box test (lags=10)
        try:
            lb_stat, p_value = _ljung_box_test(res, lags=10)
            if p_value < alpha:
                return True
        except:
            pass
    
    return False


def _ljung_box_test(residuals, lags=10):
    """
    Simple Ljung-Box test implementation.
    
    Parameters
    ----------
    residuals : ndarray
        1D residuals array
    lags : int
        Number of lags to test
        
    Returns
    -------
    stat : float
        Test statistic
    p_value : float
        P-value
    """
    n = len(residuals)
    
    # Compute autocorrelations
    acf_values = np.zeros(lags)
    for k in range(1, lags + 1):
        c0 = np.sum(residuals * np.roll(residuals, k)) / n
        c_var = np.var(residuals)
        acf_values[k - 1] = c0 / (c_var + 1e-10)
    
    # Ljung-Box statistic
    stat = n * (n + 2) * np.sum(acf_values ** 2 / (n - np.arange(1, lags + 1)))
    
    # Chi-square test
    p_value = 1 - sp_stats.chi2.cdf(stat, df=lags)
    
    return stat, p_value


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
    Simple Augmented Dickey-Fuller test implementation.
    
    Parameters
    ----------
    residuals : ndarray
        1D residuals array
        
    Returns
    -------
    p_value : float
        P-value for ADF test
    """
    n = len(residuals)
    
    # ADF regression: Δy_t = α + β*y_{t-1} + ε_t
    y = residuals[1:]  # Δy_t
    y_lag = residuals[:-1]  # y_{t-1}
    
    # Add intercept
    X = np.column_stack([np.ones(len(y)), y_lag])
    
    # Fit model
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_fitted = X @ beta
    residuals_model = y - y_fitted
    
    # t-statistic for β
    ss_residual = np.sum(residuals_model ** 2)
    se_squared = ss_residual / (len(y) - 2)
    xx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(se_squared * xx_inv[1, 1])
    t_stat = beta[1] / (se + 1e-10)
    
    # Approximate p-value using t-distribution
    # (simplified; actual ADF uses special distribution)
    p_value = sp_stats.t.sf(abs(t_stat), df=len(y) - 2) * 2
    
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
