"""Numerical and evaluation checks required before interpreting Day 5 results."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from adaptive_ode.evaluation.metrics import compute_metrics
from adaptive_ode.experiments import mismatch
from adaptive_ode.experiments.oscillator import (
    OscillatorParameters, analytical_trajectory, fit_parameters, integrate_oscillator,
)
from adaptive_ode.solvers.candidates import solve_rk4


CONFIG = Path(__file__).resolve().parents[1] / "experiments/configs/day05_mismatch.json"


@pytest.fixture
def small_config():
    config = json.loads(CONFIG.read_text())
    config.update(num_points=81, t_span=[0.0, 4.0], mismatch_levels=[0.0, 1.0], seeds=[7])
    config["nn"].update(hidden_layer_sizes=[8], max_iter=15)
    return config


@pytest.mark.parametrize("parameters,y0,start", [
    (OscillatorParameters(0.1, 4.0), [2.0, 0.0], 0.0),
    (OscillatorParameters(0.5, 3.0), [0.3, -1.2], 5.0),
    (OscillatorParameters(0.0, 2.0), [-0.5, 0.7], -2.0),
])
def test_analytical_reference_matches_independent_matrix_exponential(parameters, y0, start):
    t = np.linspace(start, start + 4, 41)
    matrix = np.array([[0.0, 1.0], [-parameters.stiffness, -parameters.damping]])
    expected = np.array([expm(matrix * (time - t[0])) @ y0 for time in t])
    # Matrix-exponential roundoff varies with BLAS/platform; this remains orders
    # of magnitude tighter than the numerical baseline tolerances.
    assert_allclose(analytical_trajectory(t, y0, parameters), expected, atol=1e-12, rtol=1e-12)


def test_dop853_matches_analytical_solution():
    parameters = OscillatorParameters(0.1, 4.0)
    t = np.linspace(0, 20, 401)
    reference = analytical_trajectory(t, [2, 0], parameters)
    assert_allclose(integrate_oscillator(t, [2, 0], parameters, rtol=1e-10, atol=1e-12),
                    reference, atol=2e-8, rtol=1e-8)


def test_rk4_has_fourth_order_convergence_on_oscillator():
    parameters = OscillatorParameters(0.1, 4.0)
    errors = []
    for points in [21, 41]:
        t = np.linspace(0, 2, points)
        numerical = solve_rk4(parameters.rhs, t, [2, 0])
        reference = analytical_trajectory(t, [2, 0], parameters)
        errors.append(np.max(np.abs(numerical - reference)))
    assert 12 < errors[0] / errors[1] < 20


def test_fit_recovers_parameters_and_predicts_unseen_future():
    true = OscillatorParameters(0.1, 4.0)
    wrong = OscillatorParameters(0.5, 3.0)
    t = np.linspace(0, 20, 201)
    reference = analytical_trajectory(t, [2, 0], true)
    fitted, details = fit_parameters(t[:120], reference[:120], [2, 0], wrong)
    assert_allclose([fitted.damping, fitted.stiffness], [0.1, 4], atol=1e-7)
    prediction = integrate_oscillator(t, [2, 0], fitted, rtol=1e-9, atol=1e-11)
    assert compute_metrics(reference[160:], prediction[160:])["mse"] < 1e-14
    assert details["nfev"] > 0


@pytest.mark.parametrize("parameters", [(-0.1, 4), (0.1, 0), (np.nan, 4), (0.1, np.inf)])
def test_invalid_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        OscillatorParameters(*parameters)


@pytest.mark.parametrize("times", [[0, 0, 1], [1, 0], [0, np.nan], [[0, 1]], [0]])
def test_invalid_time_grid_is_rejected(times):
    with pytest.raises(ValueError):
        integrate_oscillator(times, [2, 0], OscillatorParameters(0.1, 4))


def test_invalid_training_observations_are_rejected():
    with pytest.raises(ValueError, match="Training observations"):
        fit_parameters([0, 1], [[2, 0], [np.nan, 0]], [2, 0], OscillatorParameters(0.5, 3))


def test_chronological_split_is_exhaustive_disjoint_and_ordered():
    split = mismatch.chronological_split(401, 0.6, 0.2)
    assert [len(split[key]) for key in ["train", "validation", "test"]] == [240, 80, 81]
    assert_allclose(np.concatenate(list(split.values())), np.arange(401))
    assert split["train"][-1] < split["validation"][0]
    assert split["validation"][-1] < split["test"][0]


@pytest.mark.parametrize("arguments", [(10, 0.6, 0.2), (50, 0.8, 0.3), (50, 0, 0.2), (50, np.nan, 0.2), (12, 0.01, 0.2)])
def test_invalid_split_is_rejected(arguments):
    with pytest.raises(ValueError):
        mismatch.chronological_split(*arguments)


def test_selection_uses_validation_and_excludes_failures():
    rows = [
        {"method": "bad_test", "status": "ok", "validation_mse": 0.1, "test_mse": 999},
        {"method": "great_test", "status": "ok", "validation_mse": 0.2, "test_mse": 0},
        {"method": "failed", "status": "failed", "validation_mse": 0},
        {"method": "nan", "status": "ok", "validation_mse": np.nan},
    ]
    assert mismatch.choose_on_validation(rows)["method"] == "bad_test"
    assert mismatch.choose_on_validation(rows[2:]) is None


def test_final_test_observations_cannot_change_fitting_or_selection(small_config):
    t = np.linspace(*small_config["t_span"], small_config["num_points"])
    observed = analytical_trajectory(t, [2, 0], OscillatorParameters(0.1, 4))
    split = mismatch.chronological_split(len(t), 0.6, 0.2)
    poisoned = observed.copy()
    poisoned[split["test"]] = np.nan
    arguments = (split, np.array([2, 0]), OscillatorParameters(0.5, 3), small_config, 7)
    with threadpool_limits(limits=1):
        first = mismatch.fit_and_select(t, observed, *arguments)
        second = mismatch.fit_and_select(t, poisoned, *arguments)
    assert first[3] == second[3]
    assert first[2].keys() == second[2].keys()
    for key in first[2]:
        assert_allclose(first[2][key], second[2][key], atol=0, rtol=0)
    assert [row["validation_mse"] for row in first[0]] == [row["validation_mse"] for row in second[0]]


def test_validation_observations_do_not_enter_model_training(small_config):
    t = np.linspace(*small_config["t_span"], small_config["num_points"])
    observed = analytical_trajectory(t, [2, 0], OscillatorParameters(0.1, 4))
    split = mismatch.chronological_split(len(t), 0.6, 0.2)
    altered = observed.copy()
    altered[split["validation"]] += 50
    arguments = (split, np.array([2, 0]), OscillatorParameters(0.5, 3), small_config, 7)
    with threadpool_limits(limits=1):
        first = mismatch.fit_and_select(t, observed, *arguments)
        second = mismatch.fit_and_select(t, altered, *arguments)
    for key in first[2]:
        assert_allclose(first[2][key], second[2][key], atol=0, rtol=0)


def test_neural_failure_is_retained_and_other_methods_can_be_selected(small_config, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate training failure")
    monkeypatch.setattr(mismatch, "_fit_residual", fail)
    result, _ = mismatch.run_case(small_config, 1.0, 7)
    failed = [row for row in result["tuning_attempts"] if row["method"] == "rk4_residual_nn"]
    assert len(failed) == len(small_config["nn"]["alphas"])
    assert all(row["status"] == "failed" and "deliberate" in row["message"] for row in failed)
    assert result["selected_method"] != "rk4_residual_nn"
    oracle = next(row for row in result["methods"] if row["method"] == "oracle_dop853")
    assert not oracle["eligible_for_selection"]


def test_recorded_run_has_recoverable_artifacts_and_refuses_overwrite(small_config, tmp_path):
    small_config["mismatch_levels"] = [1.0]
    small_config["nn"]["alphas"] = [0.0001]
    output = tmp_path / "run"
    results = mismatch.run_study(small_config, output)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["status"] == "complete"
    assert provenance["cases"] == 1
    assert provenance["failed_attempts"] == 0
    for name in ["report.md", "metrics.csv", "summary.csv", "failures.json", "accuracy_cost.png", "trajectory.png", "environment.txt"]:
        assert (output / name).stat().st_size > 0
    assert (output / "source/adaptive_ode/experiments/mismatch.py").is_file()
    with np.load(output / "cases/level_00_seed_7.npz", allow_pickle=False) as arrays:
        assert arrays["reference"].shape == (81, 2)
    assert len(results[0]["methods"]) == 6
    with pytest.raises(FileExistsError):
        mismatch.run_study(small_config, output)


def test_study_config_valid_and_noise_requires_new_protocol(small_config):
    mismatch.validate_config(json.loads(CONFIG.read_text()))
    invalid = deepcopy(small_config)
    invalid["noise_std"] = 0.1
    with pytest.raises(ValueError, match="noiseless"):
        mismatch.validate_config(invalid)
