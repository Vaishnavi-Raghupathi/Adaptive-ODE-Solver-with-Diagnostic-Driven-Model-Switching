"""Rule-based decision engine for adaptive solver selection."""


def decide_model(diagnostics):
    """
    Make decision on which model/solver to use based on diagnostics.
    
    Decision rules:
    1. If all diagnostic flags are False → classical solver is OK
    2. If autocorrelation OR state_dependence → switch to neural ODE
    3. If heteroscedasticity → switch to neural ODE
    4. Otherwise → keep classical solver
    
    Parameters
    ----------
    diagnostics : dict
        Dictionary with keys:
        - 'heteroscedasticity': bool
        - 'autocorrelation': bool
        - 'non_stationary': bool
        - 'state_dependence': bool
        
    Returns
    -------
    model : str
        Model selection: 'classical_ok' or 'neural_ode'
    """
    # Rule 1: All clean - classical solver is sufficient
    if not any(diagnostics.values()):
        return 'classical_ok'
    
    # Rule 2: Autocorrelation or state-dependence - need neural ODE
    if diagnostics.get('autocorrelation', False) or diagnostics.get('state_dependence', False):
        return 'neural_ode'
    
    # Rule 3: Heteroscedasticity - need neural ODE
    if diagnostics.get('heteroscedasticity', False):
        return 'neural_ode'
    
    # Rule 4: Default - keep classical
    return 'classical_ok'
