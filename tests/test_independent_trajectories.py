"""Check scientific roles, initial-state transfer and accuracy/cost selection."""

import json

import numpy as np
from numpy.testing import assert_allclose
import pytest

from adaptive_ode import BenchmarkConfig, Candidate, FittedModel, Trajectory, benchmark_trajectories, evaluate
from adaptive_ode.methods import spline_candidate
from adaptive_ode.cli import main


def record(name, initial, start=0., end=1., samples=51):
    times = np.linspace(start, start+end, samples)
    return Trajectory(times, initial*np.exp(-(times-start)), name)


def datasets():
    return {"train": [record("fit-a", 1.), record("fit-b", 2., start=10.)],
            "validation": [record("val-a", 1.5, start=20.), record("val-b", .8, start=25.)],
            "test": [record("test-a", 2.5, start=30.), record("test-b", .7, start=40., samples=77)]}


class Clock:
    now = 0.
    def __call__(self):
        return self.now
    def advance(self, seconds):
        self.now += seconds


def timed_candidate(name, error, seconds, clock, group=None):
    def fit_many(times, states):
        clock.advance(.1)
        def transfer(grid, initial):
            clock.advance(seconds)
            values = initial[None, :] * np.exp(-(grid-grid[0]))[:, None]
            values[1:] += error
            return values
        return FittedModel(lambda grid: transfer(grid, states[0][0]), transfer=transfer)
    return Candidate(name, "test", lambda t, y: fit_many([t], [y]), group, fit_many)


def test_fastest_policy_selects_qualifying_variant_and_not_test_winner(monkeypatch):
    import adaptive_ode.trajectories as module
    clock = Clock()
    monkeypatch.setattr(module, "perf_counter", clock)
    candidates = [timed_candidate("precise", 0., 10., clock, "solver"),
                  timed_candidate("sufficient", .01, 1., clock, "solver"),
                  timed_candidate("too-inaccurate", .5, .1, clock)]
    config = BenchmarkConfig(selection_metric="mse", selection_policy="fastest_within_target",
                             accuracy_target=.001, timing_repeats=3)
    data = datasets()
    result = benchmark_trajectories(**data, candidates=candidates, config=config)
    assert result.selected_method == "solver"
    selected = next(row for row in result.rows if row["selected"])
    assert selected["variant"] == "sufficient"
    assert selected["prediction_timings"] == [2., 2., 2.]
    assert selected["test_score"] == pytest.approx(.0001)
    assert selected["test_score_interval"] is not None
    assert result.rows[0]["search_seconds"] >= 66.  # both variants and every repeat
    changed = []
    for old in data["test"]:
        values = old.states.copy()
        values[1:] = 100
        changed.append(Trajectory(old.t, values, old.name))
    altered = benchmark_trajectories(**{**data, "test": changed}, candidates=candidates, config=config)
    assert altered.selected_method == result.selected_method
    assert altered.scales.tolist() == result.scales.tolist()
    assert next(row for row in altered.rows if row["selected"])["variant"] == "sufficient"
    for key in result.predictions:
        assert_allclose(result.predictions[key], altered.predictions[key], rtol=0, atol=0)


def test_no_qualifying_method_does_not_silently_fall_back(monkeypatch):
    import adaptive_ode.trajectories as module
    clock = Clock()
    monkeypatch.setattr(module, "perf_counter", clock)
    result = benchmark_trajectories(**datasets(), candidates=[timed_candidate("miss", .1, 1, clock)],
                                   config=BenchmarkConfig(selection_metric="mse", selection_policy="fastest_within_target", accuracy_target=.001))
    assert result.selected_method is None
    assert result.rows[0]["status"] == "ok" and not result.rows[0]["meets_accuracy_target"]
    assert not result.rows[0]["selected"]
    assert "No successful" in result.report()


def test_full_training_experiments_only_and_no_test_reselection():
    seen = []
    def fit(times, states):
        seen.extend(times)
        def transfer(grid, initial):
            if grid[0] >= 30:
                raise RuntimeError("test-only integration failure")
            return initial[None, :] * np.exp(-(grid-grid[0]))[:, None]
        return FittedModel(lambda t: transfer(t, states[0][0]), transfer=transfer)
    data = datasets()
    result = benchmark_trajectories(**data, candidates=[Candidate("fitted", "test", lambda t,y: None, fit_trajectories=fit)])
    assert result.selected_method == "fitted"
    assert result.rows[0]["test_status"] == "failed"
    assert result.rows[0]["test_score"] is None
    assert len(seen) == 2
    assert_allclose(seen[0], data["train"][0].t)
    assert_allclose(seen[1], data["train"][1].t)
    assert "test-only integration failure" in result.report()


def test_default_models_transfer_and_export_unequal_length_trajectories(tmp_path):
    result = benchmark_trajectories(**datasets(), rhs=lambda t,y: -y,
                                   config=BenchmarkConfig(solver_sweep=True))
    assert len(result.rows) == 9 and len(result.trials) == 21
    learned = next(row for row in result.rows if row["method"] == "linear_learned_ode")
    assert learned["status"] == "ok" and learned["test_normalized_rmse"] < .02
    assert result.selected_method is not None
    assert result.context["selection_frozen_before_test"] is True
    result.save(tmp_path / "saved")
    exported = json.loads((tmp_path / "saved/results.json").read_text())
    assert exported["schema_version"] == 3
    assert len(exported["datasets"]["test"]) == 2
    with np.load(tmp_path / "saved/trajectories.npz", allow_pickle=False) as arrays:
        assert arrays["test_000_states"].shape == (51, 1)
        assert arrays["test_001_states"].shape == (77, 1)
    with pytest.raises(FileExistsError):
        result.save(tmp_path / "saved")


def test_equal_experiment_weight_and_initial_sample_exclusion():
    def fit(times, states):
        return FittedModel(lambda t: np.zeros((len(t), 1)),
                           transfer=lambda t,y: np.ones((len(t),1))*y)
    train = [record("train", 1)]
    validation = [Trajectory(np.arange(2), [5, 6], "short"),
                  Trajectory(np.arange(101), np.r_[5, np.full(100, 8)], "long")]
    result = benchmark_trajectories(train=train, validation=validation, test=[record("test",2)],
                                   candidates=[Candidate("hold", "test", lambda t,y: None, fit_trajectories=fit)],
                                   config=BenchmarkConfig(selection_metric="mse"))
    assert result.rows[0]["validation_score"] == pytest.approx((1+9)/2)


def test_overlap_and_unsupported_models_are_explicit():
    data = datasets()
    same = Trajectory(data["train"][0].t, data["train"][0].states, "copied-test")
    with pytest.raises(ValueError, match="Duplicate trajectory"):
        benchmark_trajectories(**{**data, "test": [same]})
    with pytest.raises(ValueError, match="unique"):
        benchmark_trajectories(**{**data, "test": data["train"]})
    result = benchmark_trajectories(**data, candidates=[spline_candidate()])
    assert result.selected_method is None
    assert "fit_trajectories" in result.trials[0]["message"]


def test_cli_manifest_runs_an_independent_comparison(tmp_path):
    data = datasets()
    manifest = {"time_column": "time", "state_columns": ["x"]}
    for role, records in data.items():
        manifest[role] = []
        for episode in records:
            path = tmp_path / f"{episode.name}.csv"
            np.savetxt(path, np.column_stack([episode.t, episode.states]), delimiter=",", header="time,x", comments="")
            manifest[role].append(path.name)
    path = tmp_path / "experiments.json"
    path.write_text(json.dumps(manifest))
    assert main(["trajectories", str(path), "--timing-repeats", "1", "--output", str(tmp_path / "result")]) == 0
    exported = json.loads((tmp_path / "result/results.json").read_text())
    assert exported["evaluation_kind"] == "independent_trajectories"
    assert exported["context"]["source_files"]["train"][0]["path"] == "fit-a.csv"


def test_single_series_accuracy_selection_uses_repeated_prediction_cost(monkeypatch):
    import adaptive_ode.core as module
    clock = Clock()
    monkeypatch.setattr(module, "perf_counter", clock)
    times = np.linspace(0, 1, 51)
    values = np.exp(-times)
    candidates = [timed_candidate("slow", 0, 10, clock), timed_candidate("fast", .01, 1, clock)]
    result = evaluate(times, values, candidates, config=BenchmarkConfig(selection_policy="fastest_within_target",
                      accuracy_target=.001, timing_repeats=3))
    assert result.selected_method == "fast"
    assert next(row for row in result.rows if row["selected"])["prediction_seconds"] == 1


@pytest.mark.parametrize("options", [{"selection_policy": "fastest_within_target"}, {"accuracy_target": -1},
    {"accuracy_target": float("nan")}, {"accuracy_target": True}, {"timing_repeats": 0},
    {"timing_repeats": 21}, {"solver_sweep": "yes"}, {"selection_cost": "test_score"}])
def test_invalid_policy_configuration(options):
    with pytest.raises(ValueError):
        BenchmarkConfig(**options)
