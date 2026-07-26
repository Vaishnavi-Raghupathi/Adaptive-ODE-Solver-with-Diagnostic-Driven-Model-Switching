import io

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import streamlit as st

from adaptive_ode.pipeline.run_pipeline import run_data_pipeline, run_pipeline
from adaptive_ode.systems import SYSTEMS


DIAGNOSTIC_MEANINGS = {
    "heteroscedasticity": "Residual variance changes with prediction level.",
    "autocorrelation": "Residuals remain temporally correlated.",
    "non_stationary": "Residual statistics drift over time.",
    "state_dependence": "Error magnitude depends on predicted state.",
}


def _buf_to_bytes(buf):
    if buf is None:
        return None
    buf.seek(0)
    return buf.read()


def _show_plot(raw_bytes):
    if raw_bytes is None:
        return
    st.image(io.BytesIO(raw_bytes), width="stretch")


def _format_candidate_table(rows):
    formatted = []
    for row in rows:
        formatted.append({
            "Recommended": "Yes" if row["Recommended"] else "",
            "Solver": row["Solver"],
            "Family": row["Family"],
            "Status": row["Status"],
            "MSE": f"{row['MSE']:.6g}" if row["MSE"] is not None else "N/A",
            "RMSE": f"{row['RMSE']:.6g}" if row["RMSE"] is not None else "N/A",
            "MAE": f"{row['MAE']:.6g}" if row["MAE"] is not None else "N/A",
            "Runtime (s)": f"{row['Runtime (s)']:.4f}",
        })
    return formatted


def _format_diagnostics_table(results):
    tests = (results.get("diagnostics") or {}).get("test_results", {})
    rows = []
    for name, meaning in DIAGNOSTIC_MEANINGS.items():
        flagged = bool(tests.get(name, False))
        rows.append({
            "Diagnostic": name,
            "Result": "Flagged" if flagged else "Pass",
            "Meaning": meaning,
        })
    return rows


def main():
    st.title("Adaptive Solver Selection")
    st.caption(
        "Benchmarks classical, hybrid, and physics-informed candidates, "
        "then recommends the solver with the best held-out performance."
    )

    mode = st.sidebar.radio("Input mode", ["Built-in system", "Upload data"])
    uploaded_file = None
    time_column = None
    state_columns = []
    system_key = None

    if mode == "Built-in system":
        system_options = {system.name: key for key, system in SYSTEMS.items()}
        st.sidebar.header("System")
        system_name = st.sidebar.selectbox("Dynamical system", list(system_options.keys()))
        system_key = system_options[system_name]
        selected_system = SYSTEMS[system_key]
        st.sidebar.write(selected_system.description)
        st.sidebar.caption(f"Regime: {selected_system.stiffness}")
    else:
        st.sidebar.header("Upload Data")
        uploaded_file = st.sidebar.file_uploader("CSV file", type=["csv"])
        if uploaded_file is not None:
            preview_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            numeric_columns = list(preview_df.select_dtypes(include="number").columns)
            if len(numeric_columns) >= 2:
                time_column = st.sidebar.selectbox("Time column", numeric_columns, index=0)
                default_states = [col for col in numeric_columns if col != time_column]
                state_columns = st.sidebar.multiselect(
                    "State columns",
                    numeric_columns,
                    default=default_states,
                )
                with st.expander("Uploaded Data Preview", expanded=False):
                    st.dataframe(preview_df.head(20), width="stretch")
            else:
                st.sidebar.warning("CSV needs at least one time column and one state column.")

    st.sidebar.header("Benchmark")
    num_points = st.sidebar.slider("Trajectory points", 80, 500, 240, step=20)
    train_fraction = st.sidebar.slider("Training fraction", 0.4, 0.85, 0.7, step=0.05)
    noise_std = st.sidebar.slider("Observation noise", 0.0, 1.0, 0.0, step=0.05, disabled=mode == "Upload data")
    include_pinn = st.sidebar.checkbox("Include PINN surrogate", value=False, disabled=mode == "Upload data")
    pinn_epochs = st.sidebar.slider("PINN epochs", 50, 600, 200, step=50, disabled=not include_pinn)

    run_clicked = st.sidebar.button("Run Solver Selection")

    if run_clicked:
        with st.spinner("Running solver candidates..."):
            try:
                if mode == "Upload data":
                    if uploaded_file is None:
                        st.error("Upload a CSV file first.")
                        return
                    if time_column is None or not state_columns:
                        st.error("Select one time column and at least one state column.")
                        return
                    data_frame = pd.read_csv(uploaded_file)
                    raw = run_data_pipeline(
                        data_frame[time_column].to_numpy(),
                        data_frame[state_columns].to_numpy(),
                        labels=state_columns,
                        config={
                            "train_fraction": train_fraction,
                            "save_plots": False,
                        },
                    )
                else:
                    raw = run_pipeline({
                        "system_key": system_key,
                        "noise_std": noise_std,
                        "num_points": num_points,
                        "train_fraction": train_fraction,
                        "include_pinn": include_pinn,
                        "pinn_epochs": pinn_epochs,
                        "save_plots": False,
                    })
                raw["plots"] = {
                    "trajectory": _buf_to_bytes(raw["plots"]["trajectory"]),
                    "residuals": _buf_to_bytes(raw["plots"]["residuals"]),
                    "model_comparison": _buf_to_bytes(raw["plots"]["model_comparison"]),
                }
                st.session_state["results"] = raw
            except Exception as exc:
                st.error(f"Solver selection failed: {exc}")
                return

    if "results" not in st.session_state:
        st.info("Choose a system and click Run Solver Selection.")
        return

    results = st.session_state["results"]
    recommended = results["recommended_solver"]
    system = results["system"]

    st.subheader(system["name"])
    st.write(system["description"])

    if recommended is None:
        st.error("No solver completed successfully.")
        return

    st.success(f"Recommended solver: {recommended['Solver']}")
    st.write(results["recommendation_reason"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Held-out MSE", f"{recommended['MSE']:.6g}")
    col2.metric("Held-out RMSE", f"{recommended['RMSE']:.6g}")
    col3.metric("Runtime", f"{recommended['Runtime (s)']:.4f}s")

    st.subheader("Candidate Ranking")
    st.table(_format_candidate_table(results["candidate_results"]))

    with st.expander("Diagnostics on Recommended Solver", expanded=False):
        st.table(_format_diagnostics_table(results))

    with st.expander("Candidate Notes", expanded=False):
        for row in results["candidate_results"]:
            st.markdown(f"**{row['Solver']}**: {row['Notes']}")

    st.subheader("Plots")
    st.markdown("**Recommended trajectory**")
    _show_plot(results["plots"]["trajectory"])
    st.markdown("**Residuals**")
    _show_plot(results["plots"]["residuals"])
    st.markdown("**Solver ranking**")
    _show_plot(results["plots"]["model_comparison"])


main()
