"""Dataset selection must preserve units, chronology and comparison scope."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from adaptive_ode import BenchmarkConfig, Candidate, FittedModel, benchmark_data, evaluate
from adaptive_ode.cli import main
from adaptive_ode.custom_system import build_observed_system


def test_equations_are_compared_against_observations_without_test_leakage(tmp_path):
    times = np.linspace(2, 5, 101)
    values = np.exp(-(times-2))
    config = BenchmarkConfig(selection_metric="normalized_rmse")
    result = benchmark_data(times, values, rhs=lambda t, y: -y, config=config)
    assert len(result.rows) == 11
    classical = [row for row in result.rows if row["family"] == "classical"]
    assert len(classical) == 6 and all(row["status"] == "ok" for row in classical)
    assert max(row["test_rmse"] for row in classical) < 1e-6
    assert result.assessment["best_forecast_baseline"] is not None
    changed = values.copy()
    changed[80:] = 1000
    altered = benchmark_data(times, changed, rhs=lambda t, y: -y, config=config)
    assert altered.selected_method == result.selected_method
    assert altered.assessment["profile"] == result.assessment["profile"]
    for key in result.predictions:
        assert_allclose(result.predictions[key], altered.predictions[key], rtol=0, atol=0)
    result.save(tmp_path / "run")
    record = json.loads((tmp_path / "run/results.json").read_text())
    assert record["schema_version"] == 2
    assert record["assessment"]["selected_method"] == result.selected_method
    assert "Final test" in (tmp_path / "run/report.md").read_text()


def test_normalized_selection_is_invariant_to_state_unit_changes():
    times = np.arange(50, dtype=float)

    def run(units, metric):
        def candidate(name, error):
            def fit(t, y):
                return FittedModel(lambda grid: (np.column_stack([grid, grid]) + error) * units)
            return Candidate(name, "test", fit, "variants")
        # One method group exercises variant selection as well as score scaling.
        return evaluate(times, np.column_stack([times, times]) * units,
                        [candidate("a", np.array([1, 10])), candidate("b", np.array([2, 0]))],
                        config=BenchmarkConfig(selection_metric=metric))

    ordinary = run(np.array([1, 1]), "normalized_rmse")
    rescaled = run(np.array([1000, 1]), "normalized_rmse")
    assert ordinary.rows[0]["variant"] == rescaled.rows[0]["variant"] == "b"
    assert_allclose(ordinary.rows[0]["validation_score"], rescaled.rows[0]["validation_score"])
    assert run(np.array([1, 1]), "mse").rows[0]["variant"] == "b"
    assert run(np.array([1000, 1]), "mse").rows[0]["variant"] == "a"


def test_simple_baseline_and_constant_state_are_reported():
    result = benchmark_data(np.linspace(0, 2, 50), np.ones(50))
    assert result.selected_method == "persistence"
    assert result.assessment["profile"]["normalization_scales"] == [1.0]
    assert result.assessment["profile"]["constant_training_states"] == [0]
    assert any("baseline won" in item for item in result.assessment["warnings"])
    assert "no supplied-equation" in result.assessment["comparison_scope"]


def test_all_failures_remain_exportable(tmp_path):
    def fail(t, y):
        raise RuntimeError("deliberate numerical failure")
    result = benchmark_data(np.arange(20), np.ones(20), candidates=[Candidate("failure", "test", fail)])
    assert result.selected_method is None
    assert result.assessment["accuracy_runtime_pareto_methods"] == []
    result.save(tmp_path / "failures")
    assert "deliberate numerical failure" in (tmp_path / "failures/report.md").read_text()


def test_cli_equations_file_and_state_order(tmp_path):
    times = np.linspace(0, 2, 61)
    csv = tmp_path / "data.csv"
    np.savetxt(csv, np.column_stack([times, np.exp(-times)]), delimiter=",", header="time,x", comments="")
    spec = {"variables": ["x"], "parameters": {"rate": 1}, "derivatives": ["-rate*x"]}
    equations = tmp_path / "equations.json"
    equations.write_text(json.dumps(spec))
    assert main(["data", str(csv), "--equations", str(equations), "--metric", "normalized_rmse",
                 "--output", str(tmp_path / "result")]) == 0
    assert (tmp_path / "result/report.md").exists()
    context = json.loads((tmp_path / "result/results.json").read_text())["context"]
    assert context["equation_spec"] == spec
    assert len(context["source_sha256"]) == 64
    with pytest.raises(ValueError, match="one variable"):
        build_observed_system(spec, times, np.ones((len(times), 2)))
    spec["derivatives"] = ["__import__('os')"]
    with pytest.raises(ValueError):
        build_observed_system(spec, times, np.ones(len(times)))


def test_invalid_configuration_and_equation_arguments():
    with pytest.raises(ValueError, match="selection_metric"):
        BenchmarkConfig(selection_metric="test_mse")
    with pytest.raises(TypeError, match="rhs"):
        benchmark_data(np.arange(20), np.ones(20), rhs="-x")
    with pytest.raises(ValueError, match="not both"):
        benchmark_data(np.arange(20), np.ones(20), rhs=lambda t, y: -y, candidates=[])
