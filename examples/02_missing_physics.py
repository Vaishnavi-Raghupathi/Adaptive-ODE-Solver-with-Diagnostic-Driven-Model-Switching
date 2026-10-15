"""Tutorial 2: compare calibration with a state-based polynomial correction.

The polynomial library includes the true cubic term, an explicitly favorable
baseline. The target RHS generates evaluation data only; it is not passed to
the learned correction's fit callback.
"""

import argparse
from pathlib import Path

import numpy as np

from adaptive_ode import BenchmarkConfig, DynamicalSystem, benchmark
from adaptive_ode.oscillator import OscillatorParameters
from adaptive_ode.methods import classical_candidate, oscillator_calibration_candidate, polynomial_correction_candidate
from adaptive_ode.numerics import integrate


def true_rhs(time, state):
    x, v = state
    return [v, -.1*v - 4*x - .5*x**3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assumed = OscillatorParameters(.5, 3)
    system = DynamicalSystem("duffing_demo", "Oscillator with missing cubic force", "Synthetic tutorial", "non-stiff",
                              (0, 20), np.array([2., 0.]), ["position", "velocity"], true_rhs,
                              baseline_rhs=assumed.rhs)
    candidates = [classical_candidate(assumed.rhs), oscillator_calibration_candidate(assumed),
                  polynomial_correction_candidate(assumed, window=9), polynomial_correction_candidate(assumed, window=21)]
    result = benchmark(system, config=BenchmarkConfig(num_points=401), candidates=candidates)
    print("Selected on original validation:", result.selected_method)
    model = result.models.get("polynomial_correction")
    if model is not None:
        new_initial = [1., 0.]
        prediction = model.predict_from_initial(result.t, new_initial)
        target = integrate(true_rhs, result.t, new_initial, rtol=1e-11, atol=1e-13)
        print("Unseen initial-state MSE (no refitting):", np.mean((prediction-target)**2))
    result.save(args.output)


if __name__ == "__main__":
    main()
