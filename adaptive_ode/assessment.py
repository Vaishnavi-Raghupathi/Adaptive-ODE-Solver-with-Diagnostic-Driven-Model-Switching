"""Explain empirical method selection without inferring physics from a CSV."""

import numpy as np
from adaptive_ode.selection import describe


def training_scales(values):
    """Per-state training SD; constant states use their magnitude, or one at zero."""
    with np.errstate(over="raise", invalid="raise"):
        spread = values.std(axis=0)
        level = np.max(np.abs(values), axis=0)
        varying = spread > np.finfo(float).eps * level
        scales = np.where(varying, spread, np.where(level > 0, level, 1.0))
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Training scales are non-finite; rescale the input values.")
    return scales


def assess_result(result):
    """Use training values for profiling and validation scores for recommendations.

    Counts and split boundaries describe the supplied grid. No held-out state
    values enter the profile, baseline comparison, Pareto set or recommendation.
    """
    fit = result.split["fit"]
    times, values = result.t[fit], result.observed[fit]
    steps = np.diff(times)
    scales = training_scales(values)
    constant = np.flatnonzero(np.ptp(values, axis=0) <=
                              np.finfo(float).eps * np.max(np.abs(values), axis=0)).tolist()
    regular = bool(np.allclose(steps, np.median(steps), rtol=1e-3, atol=0))
    profile = {
        "samples": len(result.t), "states": values.shape[1],
        "retained_training_samples": len(fit),
        "split_samples": {key: len(result.split[key]) for key in ("train", "validation", "test")},
        "training_time_range": [float(times[0]), float(times[-1])],
        "training_step_min": float(steps.min()), "training_step_max": float(steps.max()),
        "training_sampling_regular": regular, "constant_training_states": constant,
        "normalization_scales": scales.tolist(),
    }
    successful = [row for row in result.ranking if row["status"] == "ok"]
    baseline = next((row for row in successful if row["family"] == "forecast baseline"), None)
    selected = next((row for row in successful if row["selected"]), None)
    cost = result.config.selection_cost
    fastest = min(successful, key=lambda row: row[cost]) if successful else None
    pareto = [row["method"] for row in successful if not any(
        other["validation_score"] <= row["validation_score"]
        and other[cost] <= row[cost]
        and (other["validation_score"] < row["validation_score"]
             or other[cost] < row[cost])
        for other in successful)]
    notes = []
    if len(fit) < 20:
        notes.append("Fewer than 20 retained training samples; learned dynamics and derivative estimates may be unreliable.")
    if not regular:
        notes.append("Training times are irregular. Finite-difference dynamics can be sensitive to short gaps and noise.")
    if constant:
        notes.append("Some training states are constant; their dynamics are not identified by this training block.")
    if float(scales.max() / scales.min()) > 1000:
        notes.append("State scales differ by more than 1000x; raw MSE can be dominated by units. Consider normalized_rmse.")
    if selected and selected["family"] == "forecast baseline":
        notes.append("A simple forecast baseline won under the selected policy; added complexity was not required by that rule.")
    if not selected and successful:
        notes.append("No successful method met the accuracy target. No fallback winner was chosen.")
    if result.config.timing_repeats == 1:
        notes.append("Timing has one repeat; repeat measurements before interpreting small runtime differences.")
    if any(row["status"] != "ok" for row in result.trials):
        notes.append("Some trials failed; the recommendation covers only successful candidates. Inspect failures before interpreting the ranking.")
    equations = any(row["family"] == "classical" for row in result.rows)
    scope = ("Supplied-equation integrators and forecasting models" if equations
             else "Forecasting models learned from observations; no supplied-equation integrator comparison")
    return {
        "profile": profile, "comparison_scope": scope,
        "selection_metric": result.config.selection_metric,
        "selection_rule": describe(result.config),
        "selection_cost": cost,
        "selected_method": result.selected_method,
        "selected_validation_score": selected["validation_score"] if selected else None,
        "fastest_measured_method": fastest["method"] if fastest else None,
        "accuracy_runtime_pareto_methods": pareto,
        "best_forecast_baseline": baseline["method"] if baseline else None,
        "baseline_validation_score": baseline["validation_score"] if baseline else None,
        "warnings": notes,
        "limitations": [
            "Recommendation applies to the tested candidates, this trajectory, and this forecast horizon.",
            "Final test targets never determine the selected method or normalization scales.",
            "Observations may contain noise; observed-data error is not guaranteed physical ground-truth error.",
            "CSV statistics do not establish stiffness, chaos, full observability, or a correct governing model.",
            "Search times include all variants and repeated predictions; the selection cost uses a single fit and median prediction time.",
            "A complete finite prediction on the full grid is required; future numerical failures affect eligibility.",
        ],
    }


def render_report(result):
    assessment = result.assessment
    profile = assessment["profile"]

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    def number(value):
        return "unavailable" if value is None else f"{value:.6g}"

    lines = ["# Dataset method recommendation", "", cell(result.name), "",
             f"Selected method: **{cell(result.selected_method or 'none — no method meets the selection rule')}**.",
             f"Selection rule: {assessment['selection_rule']}.", "",
             f"Comparison: {assessment['comparison_scope']}.", "",
             f"Data: {profile['samples']} samples, {profile['states']} states; "
             f"train/validation/test = {profile['split_samples']['train']}/"
             f"{profile['split_samples']['validation']}/{profile['split_samples']['test']}.",
             f"Training-derived normalization scales: {profile['normalization_scales']}.", "",
             "Normalized RMSE is sqrt(mean(((target - prediction) / training_scale)^2)). "
             "Training scale is each state's SD; constant states use their magnitude, or 1 at zero.", "",
             "| Method | Variant | Status | Meets target | Validation score | Final test RMSE | Final test normalized RMSE | Median prediction seconds | Search seconds |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in result.ranking:
        lines.append(f"| {cell(row['method'])} | {cell(row['variant'])} | {row['status']} | {row['meets_accuracy_target']} | "
                     f"{number(row['validation_score'])} | {number(row['test_rmse'])} | "
                     f"{number(row['test_normalized_rmse'])} | {number(row['prediction_seconds'])} | {number(row['search_seconds'])} |")
    lines.extend(["", "Accuracy/runtime trade-off methods: " +
                  (", ".join(map(cell, assessment["accuracy_runtime_pareto_methods"])) or "none") + ".",
                  f"The trade-off set uses validation error and {assessment['selection_cost']}. "
                  "Selection follows the stated rule and never uses final test errors.", ""])
    if assessment["best_forecast_baseline"]:
        lines.extend([f"Best simple forecast baseline: {assessment['best_forecast_baseline']} "
                      f"(validation score {number(assessment['baseline_validation_score'])}).", ""])
    lines.extend(["## Warnings and scope", ""])
    lines.extend(f"- {note}" for note in assessment["warnings"] + assessment["limitations"])
    failures = [trial for trial in result.trials if trial["message"] or trial["warnings"]]
    if failures:
        lines.extend(["", "## Trial diagnostics", ""])
        for trial in failures:
            lines.append(f"- {cell(trial['variant'])}: {cell(trial['message'])} "
                         f"{cell('; '.join(trial['warnings']))}")
    return "\n".join(lines) + "\n"
