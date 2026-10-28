"""Streamlit view of the independent-experiment evaluation protocol."""

from dataclasses import asdict
import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from adaptive_ode import BenchmarkConfig, Trajectory, benchmark_trajectories


def example_trajectories():
    from adaptive_ode.oscillator import OscillatorParameters, analytical_trajectory
    parameters = OscillatorParameters(.1, 4)
    rng = np.random.default_rng(3107)
    datasets = {}
    for role, count in (("train", 3), ("validation", 3), ("test", 3)):
        records = []
        for index in range(count):
            times = np.linspace(0, 6, 81)
            initial = rng.uniform([.6, -.4], [2.2, .4])
            states = analytical_trajectory(times, initial, parameters)
            states += rng.normal(0, .002, states.shape)
            records.append(Trajectory(times, states, f"{role}-{index+1}"))
        datasets[role] = records
    return datasets


def render():
    st.subheader("Evaluate on independent experiments")
    st.write("Fit on training experiments, choose a configuration on validation experiments, then report untouched test experiments. Only each experiment's initial state initializes its prediction.")
    example = st.sidebar.checkbox("Use independent oscillator experiments", value=True)
    datasets, source_ids = None, []
    labels = None
    if example:
        datasets = example_trajectories()
        labels = ["position", "velocity"]
        st.info("Synthetic example: nine experiments with different initial states and fixed measurement noise. Three experiments per role; these are software demonstrations, not real-data validation.")
    else:
        uploaded = {role: st.sidebar.file_uploader(f"{role.title()} experiments (one CSV each)", type=["csv"],
                                                  accept_multiple_files=True, key=f"independent_{role}")
                    for role in ("train", "validation", "test")}
        if all(uploaded.values()):
            try:
                frames = {role: [pd.read_csv(io.BytesIO(item.getvalue())) for item in files] for role, files in uploaded.items()}
                common = list(frames["train"][0].select_dtypes(include="number").columns)
                for frame in [frame for batch in frames.values() for frame in batch]:
                    common = [column for column in common if column in frame.select_dtypes(include="number").columns]
                if len(common) < 2:
                    st.error("Every experiment needs a common numeric time column and at least one state column.")
                    return
                time_column = st.sidebar.selectbox("Experiment time column", common, index=common.index("time") if "time" in common else 0)
                labels = st.sidebar.multiselect("Experiment states (equation order)", [col for col in common if col != time_column],
                                                default=[col for col in common if col != time_column])
                if not labels:
                    st.info("Select state columns shared by all experiments.")
                    return
                datasets = {role: [Trajectory(frame[time_column].to_numpy(), frame[labels].to_numpy(), f"{role}:{file.name}")
                                   for frame, file in zip(frames[role], uploaded[role])] for role in frames}
            except (ValueError, pd.errors.ParserError) as error:
                st.error(str(error))
                return
        else:
            st.info("Provide separate training, validation and test experiments. Do not split overlapping windows from one experiment across these roles.")
    equations_enabled = st.sidebar.checkbox("Known equations for experiments", value=example)
    equation_text = None
    if datasets is not None:
        source_ids = [(role, record.name, record.fingerprint) for role, records in datasets.items() for record in records]
        st.write({role: len(records) for role, records in datasets.items()})
        if equations_enabled:
            variables = [f"y{i}" for i in range(len(labels))]
            st.caption("Equation mapping: " + ", ".join(f"{var} = {label}" for var, label in zip(variables, labels)))
            spec = {"variables": variables, "parameters": {"c": .1, "k": 4} if len(labels) == 2 else {},
                    "derivatives": ["y1", "-c*y1-k*y0"] if len(labels) == 2 else [f"-{var}" for var in variables]}
            equation_text = st.text_area("Experiment equation specification (JSON)", json.dumps(spec, indent=2), height=200)
            st.caption("These are example equations. Replace them for your data. Supplied parameters remain fixed.")
    policy_label = st.sidebar.selectbox("Recommendation rule", ["Fastest meeting accuracy target", "Lowest validation error"])
    policy = "fastest_within_target" if policy_label.startswith("Fastest") else "min_error"
    metric = st.sidebar.selectbox("Experiment score", ["normalized_rmse", "mse"])
    target = st.sidebar.number_input("Maximum validation score", value=.05, min_value=0., step=.01, format="%.4g") if policy == "fastest_within_target" else None
    cost = st.sidebar.selectbox("Runtime objective", ["prediction_seconds", "fit_plus_prediction_seconds"],
                                help="Prediction cost is the median time for the validation batch. Fit-plus-prediction adds one measured training fit.")
    repeats = st.sidebar.select_slider("Prediction timing repeats", options=[1, 3, 5], value=3)
    sweep = st.sidebar.checkbox("Sweep solver tolerances and RK4 steps", value=True, disabled=not equations_enabled)
    signature = json.dumps([source_ids, labels, equation_text, policy, metric, target, cost, repeats, sweep])
    if st.sidebar.button("Run independent comparison", type="primary"):
        st.session_state.pop("trajectory_result", None)
        if datasets is None:
            st.error("Provide experiments in all three roles.")
            return
        try:
            rhs, spec = None, None
            if equations_enabled:
                from adaptive_ode.custom_system import build_observed_system
                spec = json.loads(equation_text)
                first = datasets["train"][0]
                rhs = build_observed_system(spec, first.t, first.states).rhs
            config = BenchmarkConfig(selection_metric=metric, selection_policy=policy, accuracy_target=target,
                                     selection_cost=cost, timing_repeats=repeats, solver_sweep=sweep and equations_enabled)
            with st.spinner("Fitting training experiments, comparing validation predictions, then evaluating final test experiments…"):
                result = benchmark_trajectories(**datasets, rhs=rhs, config=config)
            result.context.update({"state_labels": labels, "equation_spec": spec, "source_ids": source_ids})
            st.session_state["trajectory_result"] = result
            st.session_state["trajectory_signature"] = signature
        except Exception as error:
            st.error(f"Comparison failed: {error}")
            return
    result = st.session_state.get("trajectory_result")
    if result is None or st.session_state.get("trajectory_signature") != signature:
        st.caption("Run the comparison to obtain a recommendation. Changes to data or evaluation settings require a new run.")
        return
    st.write(result.assessment["selection_rule"])
    selected = next((row for row in result.rows if row["selected"]), None)
    if selected:
        a, b, c = st.columns(3)
        a.metric("Selected before test", result.selected_method)
        b.metric("Validation score", f"{selected['validation_score']:.4g}")
        c.metric("Final test score", f"{selected['test_score']:.4g}" if selected["test_score"] is not None else "Failed")
        if selected["test_status"] != "ok":
            st.warning("The chosen method failed final test evaluation. Its selection remains frozen; no replacement was chosen using test outcomes.")
    else:
        st.warning(result.assessment["no_recommendation_reason"])
    if len(result.datasets["test"]) < 5:
        st.caption("There are fewer than five test experiments. Any bootstrap interval is exploratory and conditional on this small sample.")
    table = pd.DataFrame(result.ranking)
    columns = ["method", "variant", "selected", "meets_accuracy_target", "validation_score", "test_score", "test_status",
               "prediction_seconds", "fit_seconds", "search_seconds"]
    st.dataframe(table[columns], hide_index=True, width="stretch")
    valid = [trial for trial in result.trials if trial["status"] == "ok"]
    if valid:
        figure = go.Figure()
        for family in dict.fromkeys(trial["family"] for trial in valid):
            trials = [trial for trial in valid if trial["family"] == family]
            figure.add_trace(go.Scatter(x=[row[cost] for row in trials], y=[row["validation_score"] for row in trials],
                                        mode="markers", text=[row["variant"] for row in trials], name=family,
                                        hovertemplate="%{text}<br>seconds=%{x}<br>validation score=%{y}<extra></extra>"))
        if target is not None:
            figure.add_hline(y=target, line_dash="dash", annotation_text="accuracy target")
        figure.update_layout(title="Validation accuracy versus measured cost", xaxis_title=cost,
                             xaxis_type="log", yaxis_title=metric,
                             yaxis_type="log" if all(row["validation_score"] > 0 for row in valid) and target != 0 else "linear")
        st.plotly_chart(figure, width="stretch")
    role = st.selectbox("Inspect experiment role", ["test", "validation"])
    records = result.datasets[role]
    episode = st.selectbox("Experiment", list(range(len(records))), format_func=lambda i: records[i].name)
    dimension = st.selectbox("Experiment state", list(range(len(labels))), format_func=lambda i: labels[i])
    record = records[episode]
    figure = go.Figure(go.Scatter(x=record.t, y=record.states[:, dimension], name="observations", line={"color": "black"}))
    for row in result.rows:
        if row["selected"]:
            key = f"{row['prediction_key']}_{role}_{episode:03d}"
            if key in result.predictions:
                figure.add_trace(go.Scatter(x=record.t, y=result.predictions[key][:, dimension], name=row["method"]))
    figure.update_layout(xaxis_title="Time", yaxis_title=labels[dimension])
    st.plotly_chart(figure, width="stretch")
    st.download_button("Download experiment recommendation", result.report(), "trajectory-recommendation.md", "text/markdown")
    st.download_button("Download experiment metrics", table[columns].to_csv(index=False), "trajectory-metrics.csv", "text/csv")
    metadata = {"schema_version": 3, "config": asdict(result.config), "assessment": result.assessment,
                "context": result.context, "rows": result.rows, "trials": result.trials}
    st.download_button("Download experiment evaluation JSON", json.dumps(metadata, indent=2, allow_nan=False), "trajectory-results.json", "application/json")
    with st.expander("Independent-experiment protocol and diagnostics"):
        st.json(result.assessment)
        st.dataframe(pd.DataFrame(result.trials)[["variant", "status", "message"]], hide_index=True)
