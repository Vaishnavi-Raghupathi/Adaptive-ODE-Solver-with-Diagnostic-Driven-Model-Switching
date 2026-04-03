import streamlit as st
from adaptive_ode.pipeline.run_pipeline import run_pipeline


SCENARIOS = {
    "Clean Data": {
        "noise_std": 0.0,
        "mismatch": False,
    },
    "Noisy Data": {
        "noise_std": 0.5,
        "mismatch": False,
    },
    "Model Mismatch": {
        "noise_std": 0.0,
        "mismatch": True,
    },
}


def main():
    st.title("Adaptive ODE Solver Demo")

    scenario = st.selectbox(
        "Select Scenario",
        options=["Clean Data", "Noisy Data", "Model Mismatch"],
    )

    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            try:
                config = {
                    "noise_std": SCENARIOS[scenario]["noise_std"],
                    "mismatch": SCENARIOS[scenario]["mismatch"],
                    "save_plots": False,
                    "show_plots": False,
                }
                results = run_pipeline(config)
            except Exception as exc:
                st.error(f"Simulation failed: {exc}")
                return

        st.subheader("Decision")
        st.success(f"Selected Model: {results['decision']}")

        st.subheader("Metrics")
        col_classical, col_neural = st.columns(2)

        with col_classical:
            st.markdown("**Classical**")
            st.metric("MSE", f"{results['metrics_classical']['mse']:.6f}")
            st.metric("RMSE", f"{results['metrics_classical']['rmse']:.6f}")
            st.metric("MAE", f"{results['metrics_classical']['mae']:.6f}")

        with col_neural:
            st.markdown("**Neural**")
            if results["metrics_neural"] is not None:
                st.metric("MSE", f"{results['metrics_neural']['mse']:.6f}")
                st.metric("RMSE", f"{results['metrics_neural']['rmse']:.6f}")
                st.metric("MAE", f"{results['metrics_neural']['mae']:.6f}")
            else:
                st.metric("MSE", "N/A")
                st.metric("RMSE", "N/A")
                st.metric("MAE", "N/A")

        st.subheader("Plots")
        st.markdown("**residuals.png**")
        st.pyplot(results["plots"]["residuals"])

        st.markdown("**model_comparison.png**")
        if results["plots"]["model_comparison"] is not None:
            st.pyplot(results["plots"]["model_comparison"])
        else:
            st.info("model_comparison.png is unavailable because Neural ODE was not selected.")


if __name__ == "__main__":
    main()
