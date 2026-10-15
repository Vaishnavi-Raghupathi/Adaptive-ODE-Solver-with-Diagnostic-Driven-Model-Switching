"""Compatibility imports for the original frozen research workflow."""

from adaptive_ode.oscillator import (
    OscillatorParameters,
    analytical_trajectory,
    fit_parameters,
    integrate_oscillator,
    validate_initial_state,
    validate_times,
)

__all__ = ["OscillatorParameters", "analytical_trajectory", "fit_parameters",
           "integrate_oscillator", "validate_initial_state", "validate_times"]
