"""Exercise the installed-style CLI and the Streamlit API client."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from adaptive_ode.cli import main, read_csv


def test_cli_known_benchmark_and_no_overwrite(tmp_path):
    output = tmp_path / "run"
    assert main(["benchmark", "--points", "81", "--output", str(output)]) == 0
    assert json.loads((output / "results.json").read_text())["selected_method"] == "calibrated_oscillator"
    with pytest.raises(SystemExit) as error:
        main(["benchmark", "--output", str(output)])
    assert error.value.code == 2


def test_cli_csv_reader_and_data_run(tmp_path):
    csv = tmp_path / "input.csv"
    t = np.linspace(0, 2, 61)
    np.savetxt(csv, np.column_stack([t, np.exp(-t)]), delimiter=",", header="time,x", comments="")
    times, states = read_csv(csv)
    assert states.shape == (61, 1)
    assert np.allclose(times, t)
    assert main(["data", str(csv), "--output", str(tmp_path / "out")]) == 0
    with pytest.raises(ValueError):
        read_csv(csv, "missing")
    with pytest.raises(ValueError):
        read_csv(csv, state_columns=["time"])


def test_python_module_help():
    completed = subprocess.run([sys.executable, "-m", "adaptive_ode", "--help"], capture_output=True, text=True, check=True)
    assert "benchmark" in completed.stdout and "data" in completed.stdout


def test_streamlit_demo_runs_a_real_benchmark():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = streamlit_testing.AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Built-in system").run()
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert any(metric.value == "calibrated_oscillator" for metric in app.metric)
    assert not app.error


def test_dataset_demo_equations_and_stale_results():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = streamlit_testing.AppTest.from_file(str(app_path), default_timeout=30).run()
    next(item for item in app.sidebar.checkbox if item.label == "Use example oscillator dataset").check().run()
    next(item for item in app.sidebar.checkbox if item.label == "I have governing equations").check().run()
    app.sidebar.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.session_state["benchmark_result"].rows) == 11
    assert len(app.metric) == 3
    next(item for item in app.sidebar.selectbox if item.label == "Selection metric").set_value("mse").run()
    assert not app.exception
    assert len(app.metric) == 0  # old recommendations cannot masquerade as a new run


def test_independent_trajectory_demo_selects_and_exports_a_result():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = streamlit_testing.AppTest.from_file(str(app_path), default_timeout=60).run()
    app.sidebar.radio[0].set_value("Independent trajectories").run()
    app.sidebar.button[0].click().run()
    assert not app.exception and not app.error
    result = app.session_state["trajectory_result"]
    assert result.selected_method is not None
    assert len(result.rows) == 9 and len(result.trials) == 21
    assert result.assessment["trajectory_counts"] == {"train": 3, "validation": 3, "test": 3}
    assert "No fallback" not in result.report()
