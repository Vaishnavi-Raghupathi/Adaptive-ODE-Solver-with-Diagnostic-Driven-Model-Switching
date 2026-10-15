"""Small software fixtures; not used to tune the frozen production study."""

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from adaptive_ode.experiments.robustness import reference_trajectory, run_condition


def test_missing_physics_reference_and_transfer_use_frozen_models():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "experiments/configs/day06_07_robustness.json").read_text())
    config.update(t_span=[0, 4], num_points=81, held_out_initial_states=[[1, 0]], nn_max_iter=10,
                  nn_alphas=[1e-4], smoothing_windows=[5], acceleration_nn_window=5)
    system = "missing_cubic_force"
    references = {system: []}
    for initial in [config["training_initial_state"], *config["held_out_initial_states"]]:
        reference, check = reference_trajectory(config, system, initial)
        assert check["max_abs_difference"] < 1e-7
        references[system].append(reference)
    with threadpool_limits(limits=1):
        result, transfers, arrays = run_condition(config, system, .02, 1, 101, references)
    assert result.selected_method is not None
    assert len(result.rows) == 6
    assert len(transfers) == 6
    trajectory_row = next(row for row in transfers if row["method"] == "trajectory_residual_nn")
    assert trajectory_row["status"] == "not_supported"
    assert any(row["status"] == "ok" for row in transfers)
    assert np.array_equal(arrays["transfer_reference_0"], references[system][1])
    assert all(result.t[result.split["fit"]] < result.t[result.split["validation"]][0])
