def decide_model(diagnostics, mse, mse_threshold=1.5):
    if mse < mse_threshold:
        return "classical_ok"
    
    n_flagged = sum(bool(v) for v in diagnostics.values())
    
    if n_flagged >= 1:
        return "neural_ode"
    
    return "classical_ok"