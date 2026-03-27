"""Diagnostic engine for model evaluation and error analysis."""

import numpy as np
from adaptive_ode.diagnostics.statistical_tests import (
    compute_residual_statistics,
    compute_error_metrics,
    test_heteroscedasticity,
    test_autocorrelation,
    test_stationarity,
    test_state_dependence
)


class DiagnosticEngine:
    """
    Engine for comprehensive model diagnostics and evaluation.
    
    Analyzes predictions, residuals, and error patterns to identify
    solver performance issues and statistical problems.
    """
    
    def __init__(self, y_true, y_pred):
        """
        Initialize diagnostic engine.
        
        Parameters
        ----------
        y_true : ndarray
            True trajectories (shape: n_points, n_dims)
        y_pred : ndarray
            Predicted trajectories (shape: n_points, n_dims)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.residuals = y_true - y_pred
    
    def run_all_tests(self):
        """
        Run all statistical tests on residuals and predictions.
        
        Tests performed:
        - Heteroscedasticity (Breusch-Pagan)
        - Autocorrelation (Ljung-Box)
        - Stationarity (Augmented Dickey-Fuller)
        - State-dependence (linear regression)
        
        Returns
        -------
        test_results : dict
            Dictionary containing test results:
            - 'heteroscedasticity': bool
            - 'autocorrelation': bool
            - 'non_stationary': bool
            - 'state_dependence': bool
        """
        test_results = {
            'heteroscedasticity': test_heteroscedasticity(self.residuals, self.y_pred),
            'autocorrelation': test_autocorrelation(self.residuals),
            'non_stationary': test_stationarity(self.residuals),
            'state_dependence': test_state_dependence(self.residuals, self.y_pred)
        }
        return test_results
    
    def get_residual_statistics(self):
        """
        Get statistical summary of residuals.
        
        Returns
        -------
        stats : dict
            Dictionary containing:
            - 'mean': Mean of residuals
            - 'std': Standard deviation
            - 'max_abs': Maximum absolute value
            - 'l2_norm': L2 norm
        """
        return compute_residual_statistics(self.residuals)
    
    def get_error_metrics(self):
        """
        Get error metrics comparing true and predicted values.
        
        Returns
        -------
        metrics : dict
            Dictionary containing:
            - 'mae': Mean absolute error
            - 'mse': Mean squared error
            - 'rmse': Root mean squared error
        """
        return compute_error_metrics(self.y_true, self.y_pred)
    
    def get_residuals(self):
        """
        Get residuals.
        
        Returns
        -------
        residuals : ndarray
            Residuals (y_true - y_pred)
        """
        return self.residuals
    
    def diagnose(self):
        """
        Run full diagnostic suite.
        
        Returns
        -------
        diagnosis : dict
            Comprehensive diagnostic results including:
            - test_results: All statistical tests
            - residual_stats: Statistical metrics for residuals
            - error_metrics: Error metrics (MAE, MSE, RMSE)
        """
        diagnosis = {
            'test_results': self.run_all_tests(),
            'residual_stats': self.get_residual_statistics(),
            'error_metrics': self.get_error_metrics()
        }
        return diagnosis
