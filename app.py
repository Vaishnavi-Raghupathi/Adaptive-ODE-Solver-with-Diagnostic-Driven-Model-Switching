import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from adaptive_ode.pipeline.run_pipeline import run_pipeline


SCENARIOS = {
    "Clean Data": {
        "noise_std": 0.0,
        "mismatch": False,
        "description": "No added noise and no model mismatch. This is the easiest case for the classical solver.",
    },
    "Noisy Data": {
        "noise_std": 0.5,
        "mismatch": False,
        "description": "Adds observation noise while keeping the same underlying dynamics. Useful to test robustness under measurement error.",
    },
    "Model Mismatch": {
        "noise_std": 0.0,
        "mismatch": True,
        "description": "Data is generated with a damped oscillator using true parameters. The classical solver uses wrong parameters, creating model mismatch. Neural ODE learns the true dynamics.",
    },
}

DIAGNOSTIC_MEANINGS = {
    "heteroscedasticity": "Checks whether residual variance changes with state/prediction level.",
    "autocorrelation":    "Checks whether residuals remain temporally correlated.",
    "non_stationary":     "Checks whether residual statistics drift over time.",
    "state_dependence":   "Checks whether error magnitude depends on predicted state.",
}


def _buf_to_bytes(buf):
    if buf is None:
        return None
    buf.seek(0)
    return buf.read()


def _show_plot(raw_bytes):
    if raw_bytes is None:
        return
    st.image(io.BytesIO(raw_bytes), use_container_width=True)


def _get_decision_message(results):
    decision     = results["decision"]
    tests        = results.get("diagnostics", {}).get("test_results", {})
    failed_tests = [name for name, flagged in tests.items() if flagged]

    if decision == "classical_ok":
        if not failed_tests:
            return "success", "Classical solver sufficient: residuals show no significant patterns."
        return "success", "Classical solver sufficient: overall error is low enough despite minor diagnostic flags."

    if failed_tests:
        reasons = ", ".join(failed_tests)
        return "warning", f"Neural ODE selected: residual diagnostics flagged: {reasons}."

    return "warning", "Neural ODE selected: diagnostics indicate classical solver is insufficient."


def _format_metrics_table(results):
    classical = results["metrics_classical"]
    neural    = results["metrics_neural"]
    rows = []
    for key, label in [("mse", "MSE"), ("rmse", "RMSE"), ("mae", "MAE")]:
        rows.append({
            "Metric":     label,
            "Classical":  f"{classical[key]:.6f}",
            "Neural ODE": f"{neural[key]:.6f}" if neural else "N/A",
        })
    return rows


def _format_diagnostics_table(results):
    tests = results.get("diagnostics", {}).get("test_results", {})
    rows  = []
    for name in ["heteroscedasticity", "autocorrelation", "non_stationary", "state_dependence"]:
        flagged = bool(tests.get(name, False))
        rows.append({
            "Test":    name,
            "Result":  "Fail" if flagged else "Pass",
            "Meaning": DIAGNOSTIC_MEANINGS[name],
        })
    return rows


def main():
    st.title("Adaptive ODE Solver")
    st.caption(
        "Simulates ODE trajectories, diagnoses residual behavior, "
        "and adaptively chooses classical or Neural ODE modeling."
    )

    st.sidebar.header("Simulation Settings")
    scenario = st.sidebar.selectbox(
        "Select Scenario",
        options=["Clean Data", "Noisy Data", "Model Mismatch"],
    )
    st.sidebar.write(SCENARIOS[scenario]["description"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("Advanced Settings")
    epochs = st.sidebar.slider(
        "Neural ODE Training Epochs",
        min_value=100,
        max_value=800,
        value=800,
        step=100,
        help="Reduce to 200-300 on cloud deployments for faster runs. Full 800 epochs recommended locally for best results.",
    )

    if epochs < 500:
        st.sidebar.warning(
            f"⚠️ {epochs} epochs may reduce Neural ODE accuracy. "
            "800 epochs recommended for best results (local only)."
        )

    run_clicked = st.sidebar.button("Run Simulation")

    if run_clicked:
        with st.spinner("Running simulation..."):
            try:
                config = {
                    "noise_std":  SCENARIOS[scenario]["noise_std"],
                    "mismatch":   SCENARIOS[scenario]["mismatch"],
                    "save_plots": False,
                    "show_plots": False,
                    "epochs":     epochs,
                }
                raw = run_pipeline(config)

                raw["plots"] = {
                    "trajectory":       _buf_to_bytes(raw["plots"]["trajectory"]),
                    "residuals":        _buf_to_bytes(raw["plots"]["residuals"]),
                    "model_comparison": _buf_to_bytes(raw["plots"]["model_comparison"]),
                }

                st.session_state["results"]  = raw
                st.session_state["scenario"] = scenario
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")
                return

    if "results" not in st.session_state:
        st.info("Choose a scenario in the sidebar and click Run Simulation to view results.")
        return

    results           = st.session_state["results"]
    selected_scenario = st.session_state.get("scenario", scenario)

    st.subheader(f"Results: {selected_scenario}")

    banner_type, decision_text = _get_decision_message(results)
    if banner_type == "success":
        st.success(decision_text)
    else:
        st.warning(decision_text)

    st.subheader("Metrics")
    st.table(_format_metrics_table(results))

    if results["metrics_neural"] is None:
        st.caption("Neural ODE metrics are N/A because the decision logic kept the classical solver.")

    if results["decision"] == "neural_ode" and results["improvement_percent"] is not None:
        st.metric("Improvement % (MSE)", f"{results['improvement_percent']:.2f}%")

    with st.expander("Diagnostics", expanded=False):
        st.table(_format_diagnostics_table(results))

    st.subheader("Plots")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Trajectory comparison**")
        _show_plot(results["plots"]["trajectory"])
    with col2:
        st.markdown("**Residuals**")
        _show_plot(results["plots"]["residuals"])

    if results["decision"] == "neural_ode" and results["plots"]["model_comparison"] is not None:
        st.markdown("**Model comparison (Classical vs Neural ODE)**")
        _show_plot(results["plots"]["model_comparison"])


main()