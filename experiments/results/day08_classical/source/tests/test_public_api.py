"""Tests of actual selection, isolation, numerical behavior, and exports."""

import json
import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest

from adaptive_ode import BenchmarkConfig, Candidate, FittedModel, benchmark, benchmark_data, evaluate
from adaptive_ode.experiments.oscillator import OscillatorParameters, analytical_trajectory
from adaptive_ode.methods import classical_candidate, polynomial_correction_candidate, smoothed_acceleration
from adaptive_ode.numerics import integrate


def test_public_import_does_not_load_optional_ml_or_demo_dependencies():
    code = "from adaptive_ode import benchmark; import sys; assert not any(x in sys.modules for x in ['sklearn','torch','streamlit','matplotlib'])"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_builtin_mismatch_selects_calibration_and_exports(tmp_path):
    result = benchmark(config=BenchmarkConfig(num_points=101))
    assert result.selected_method == "calibrated_oscillator"
    winner = result.ranking[0]
    assert winner["test_mse"] < 1e-12
    result.save(tmp_path / "saved")
    record = json.loads((tmp_path / "saved/results.json").read_text())
    assert record["selected_method"] == result.selected_method
    assert len(record["split_indices"]["fit"]) == 60
    with np.load(tmp_path / "saved/predictions.npz", allow_pickle=False) as arrays:
        assert arrays["target"].shape == (101, 2)
    with pytest.raises(FileExistsError):
        result.save(tmp_path / "saved")


def test_selection_and_ranking_ignore_final_test_scores():
    t = np.linspace(0, 1, 50)
    observed = np.ones((50, 1))
    observed[40:] = 0
    candidates = [Candidate("one", "test", lambda t, y: FittedModel(lambda times: np.ones((len(times), 1)))),
                  Candidate("zero", "test", lambda t, y: FittedModel(lambda times: np.zeros((len(times), 1))))]
    result = evaluate(t, observed, candidates)
    assert result.selected_method == "one"
    assert result.ranking[0]["test_mse"] > result.ranking[1]["test_mse"]
    observed[40:] = 500
    changed = evaluate(t, observed, candidates)
    assert changed.selected_method == result.selected_method
    for key in result.predictions:
        assert_allclose(result.predictions[key], changed.predictions[key], atol=0, rtol=0)


def test_variants_tuned_only_on_validation_and_search_cost_is_retained():
    t = np.linspace(0, 1, 50)
    y = np.ones((50, 1))
    candidates = [Candidate("bad", "test", lambda t, y: FittedModel(lambda times: np.zeros((len(times), 1))), "family"),
                  Candidate("good", "test", lambda t, y: FittedModel(lambda times: np.ones((len(times), 1))), "family")]
    result = evaluate(t, y, candidates)
    assert len(result.rows) == 1 and len(result.trials) == 2
    assert result.rows[0]["variant"] == "good"
    assert result.rows[0]["search_seconds"] >= result.rows[0]["fit_seconds"]


def test_only_retained_training_samples_are_passed_to_fit():
    seen = []
    def fit(t, y):
        seen.extend(t.tolist())
        return FittedModel(lambda times: np.ones((len(times), 1)))
    t = np.linspace(0, 1, 50)
    result = evaluate(t, np.ones((50, 1)), [Candidate("test", "test", fit)], config=BenchmarkConfig(train_stride=4))
    assert_allclose(seen, t[result.split["fit"]])
    assert max(seen) < t[result.split["validation"][0]]


def test_failures_and_warnings_are_visible():
    import warnings
    def fail(t, y):
        warnings.warn("expected warning", UserWarning)
        raise RuntimeError("expected failure")
    result = evaluate(np.arange(20), np.ones((20, 1)), [Candidate("bad", "test", fail)])
    assert result.selected_method is None
    assert result.rows[0]["status"] == "failed"
    assert "expected failure" in result.rows[0]["message"]
    assert "expected warning" in result.trials[0]["warnings"][0]


@pytest.mark.parametrize("settings", [{"train_fraction": .9, "validation_fraction": .2}, {"seed": -1},
    {"num_points": True}, {"nn_max_iter": 0}, {"train_stride": 0}, {"include_ml": "yes"}])
def test_config_validation(settings):
    with pytest.raises(ValueError):
        BenchmarkConfig(**settings)


def test_invalid_input_and_duplicate_names_rejected():
    candidate = classical_candidate(lambda t, y: -y)
    with pytest.raises(ValueError):
        evaluate([0, 2, 1], [[1], [1], [1]], [candidate])
    with pytest.raises(ValueError):
        evaluate(np.arange(20), np.ones((20, 1)), [candidate, candidate])
    with pytest.raises(ValueError):
        evaluate(np.arange(20), np.ones((20, 1)), [])
    with pytest.raises(ValueError):
        evaluate(np.arange(20), np.full((20, 1), np.nan), [candidate])


def test_rhs_budget_and_wrong_shape_are_failures():
    with pytest.raises(RuntimeError, match="budget"):
        integrate(lambda t, y: -y, np.linspace(0, 10, 50), [1], max_nfev=4)
    with pytest.raises(ValueError, match="RHS"):
        integrate(lambda t, y: [0, 0], [0, 1], [1])


def test_polynomial_correction_can_transfer_without_new_labels():
    parameters = OscillatorParameters(.1, 4)
    t = np.linspace(0, 12, 241)
    y = analytical_trajectory(t, [2, 0], parameters)
    candidate = polynomial_correction_candidate(window=9)
    fitted = candidate.fit(t[:145], y[:145])
    new_target = analytical_trajectory(t, [1, 0], parameters)
    prediction = fitted.predict_from_initial(t, [1, 0])
    assert np.mean((new_target-prediction)**2) < 1e-4
    with pytest.raises(NotImplementedError):
        FittedModel(lambda t: np.ones((len(t), 1))).predict_from_initial(t, [1])


def test_smoothing_rejects_irregular_or_too_short_data():
    with pytest.raises(ValueError, match="uniform"):
        smoothed_acceleration(np.arange(30)**2, np.ones((30, 2)), 9)
    with pytest.raises(ValueError, match="window"):
        smoothed_acceleration(np.arange(8), np.ones((8, 2)), 9)


def test_one_state_data_supported():
    t = np.linspace(0, 2, 80)
    result = benchmark_data(t, np.exp(-t))
    assert result.target.shape == (80, 1)
    assert result.selected_method is not None


def test_rk4_substepping_removes_training_sampling_step_change():
    rhs = OscillatorParameters(.5, 3).rhs
    dense = np.linspace(0, 4, 81)
    sparse = dense[::4]
    full = integrate(rhs, dense, [2, 0], method="RK4", rk4_max_step=.05)
    thin = integrate(rhs, sparse, [2, 0], method="RK4", rk4_max_step=.05)
    assert_allclose(full[::4], thin, atol=1e-12, rtol=1e-12)
