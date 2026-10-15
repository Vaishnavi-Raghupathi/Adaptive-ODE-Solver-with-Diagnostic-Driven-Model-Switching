"""Record the existing app pipeline without overwriting its saved plots."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from adaptive_ode.experiments.artifacts import create_run, finish_run, write_json
from adaptive_ode.experiments.oscillator import OscillatorParameters, analytical_trajectory
from adaptive_ode.pipeline.run_pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    config = {
        "system_key": "mismatch", "noise_std": 0.0, "num_points": 300,
        "train_fraction": 0.7, "include_pinn": False,
        "save_plots": False, "seed": 7,
    }
    with threadpool_limits(limits=1):
        output, metadata = create_run(arguments.output, config)
        start = perf_counter()
        result = run_pipeline(config)
        reference = analytical_trajectory(result["t"], [2.0, 0.0], OscillatorParameters(0.1, 4.0))
        write_json(output / "baseline.json", {
            "recommended_solver": result["recommended_solver"],
            "recommendation_reason": result["recommendation_reason"],
            "candidate_results": result["candidate_results"],
            "reference_max_abs_vs_analytical": float(np.max(np.abs(reference - result["y_reference"]))),
            "evaluation_note": "Legacy two-way random split; selection and reported scores share held-out points. Not a final test estimate.",
            "pinn_note": "PINN disabled for the baseline capture; PyTorch/torchdiffeq are not required for Days 1–5.",
        })
        for name, buffer in result["plots"].items():
            if buffer is not None:
                (output / f"{name}.png").write_bytes(buffer.getvalue())
        np.savez_compressed(output / "trajectory.npz", t=result["t"], reference=result["y_reference"])
        finish_run(output, metadata, perf_counter() - start)
        print(result["recommended_solver"])


if __name__ == "__main__":
    main()
