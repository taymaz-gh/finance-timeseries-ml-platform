"""Providing an interactive frontend for financial risk inference."""

from __future__ import annotations

import hashlib
import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st


API_URL = os.getenv(
    "FINANCE_ML_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

RISK_STATES = [
    "normal",
    "elevated_risk",
    "liquidity_stress",
    "delinquency_risk",
]

PROBABILITY_COLUMNS = [f"prob_{name}" for name in RISK_STATES]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def request_csv(
    endpoint: str,
    filename: str,
    content: bytes,
) -> dict:
    """Submitting a CSV to the existing FastAPI backend."""

    try:
        # Allowing sufficient time for Spark preprocessing and LSTM inference.
        response = httpx.post(
            f"{API_URL}{endpoint}",
            files={"file": (filename, content, "text/csv")},
            timeout=httpx.Timeout(600.0, connect=10.0),
        )

    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Unable to communicate with the prediction service: {exc}"
        ) from exc

    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"Prediction service returned HTTP {response.status_code}: " f"{detail}"
        )

    return response.json()


def render_report(result: dict) -> None:
    """Rendering actual model predictions and prediction coverage."""

    # Extracting the validated API prediction results.
    predictions = pd.DataFrame(result["predictions"])
    coverage = pd.DataFrame(result["coverage"])
    summary = result["summary"]

    st.subheader("Prediction summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accounts analyzed", summary["accounts_received"])
    col2.metric("Predictions generated", summary["predictions"])
    col3.metric(
        "Dates without sufficient context",
        summary["dates_insufficient_context"],
    )

    st.caption(f"Inference run: {result['run_id']}")

    if predictions.empty:
        st.warning(
            "No predictions were generated. The uploaded accounts may "
            "not have enough historical observations."
        )
        return

    # Preparing predictions for interactive visualizations.
    predictions["date"] = pd.to_datetime(predictions["date"])

    account_ids = sorted(predictions["account_id"].unique())

    selected_account = st.selectbox(
        "Select an account to inspect",
        account_ids,
    )

    account_data = (
        predictions.loc[predictions["account_id"] == selected_account]
        .sort_values("date")
        .copy()
    )

    latest = account_data.iloc[-1]

    col1, col2 = st.columns(2)
    col1.metric(
        "Latest predicted risk state",
        latest["predicted_state_name"].replace("_", " ").title(),
    )
    col2.metric(
        "Latest prediction date",
        latest["date"].strftime("%Y-%m-%d"),
    )

    # Visualizing the account's predicted risk-state timeline.
    st.subheader("Risk classification over time")

    timeline = px.scatter(
        account_data,
        x="date",
        y="predicted_class",
        color="predicted_state_name",
        labels={
            "date": "Date",
            "predicted_class": "Risk state",
            "predicted_state_name": "Classification",
        },
        category_orders={"predicted_state_name": RISK_STATES},
    )

    timeline.update_traces(marker_size=12)
    timeline.update_yaxes(
        tickvals=[0, 1, 2, 3],
        ticktext=[name.replace("_", " ").title() for name in RISK_STATES],
    )

    st.plotly_chart(timeline, width="stretch")

    # Visualizing predicted probabilities for all four risk states.
    st.subheader("Risk-class probabilities")

    probability_data = account_data.melt(
        id_vars=["date"],
        value_vars=PROBABILITY_COLUMNS,
        var_name="Risk state",
        value_name="Probability",
    )

    probability_data["Risk state"] = (
        probability_data["Risk state"]
        .str.removeprefix("prob_")
        .str.replace("_", " ")
        .str.title()
    )

    probability_chart = px.line(
        probability_data,
        x="date",
        y="Probability",
        color="Risk state",
        markers=True,
    )

    probability_chart.update_yaxes(range=[0, 1])

    st.plotly_chart(probability_chart, width="stretch")

    # Summarizing the account's predicted classifications.
    st.subheader("Risk-state distribution")

    distribution = (
        account_data["predicted_state_name"]
        .value_counts()
        .rename_axis("Risk state")
        .reset_index(name="Number of dates")
    )

    distribution_chart = px.bar(
        distribution,
        x="Risk state",
        y="Number of dates",
        color="Risk state",
    )

    st.plotly_chart(distribution_chart, width="stretch")

    # Reporting dates without sufficient sequence context.
    st.subheader("Prediction coverage")

    account_coverage = coverage.loc[coverage["account_id"] == selected_account]

    st.dataframe(
        account_coverage,
        hide_index=True,
        width="stretch",
    )

    # Providing downloadable results from the actual inference run.
    st.subheader("Download predictions")

    st.download_button(
        label="Download all predictions (CSV)",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name=f"{result['run_id']}_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    """Running the user-facing financial risk dashboard."""

    st.set_page_config(
        page_title="Financial Risk Dashboard",
        layout="wide",
    )

    st.title("Financial Risk Prediction")
    st.caption(
        "Upload financial account observations to classify risk states "
        "using the trained LSTM model."
    )

    # Checking that the prediction service and model are available.
    try:
        ready = httpx.get(f"{API_URL}/ready", timeout=5.0)
        model_response = httpx.get(f"{API_URL}/model", timeout=5.0)

        if ready.status_code != 200 or model_response.status_code != 200:
            st.error(
                "The prediction service is running, but no model "
                "is available. Please check the backend configuration."
            )
            st.stop()

        model_info = model_response.json()

    except httpx.RequestError:
        st.error(
            "Cannot connect to the prediction service. "
            f"Please start the FastAPI backend at {API_URL}."
        )
        st.stop()

    st.success("Prediction service is ready.")

    with st.expander("Model information"):
        st.json(model_info)

    # Accepting raw financial observations from the user.
    st.subheader("1. Upload your financial data")

    uploaded_file = st.file_uploader(
        "Choose a CSV containing account-day observations",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        st.stop()

    content = uploaded_file.getvalue()

    if len(content) > MAX_UPLOAD_BYTES:
        st.error("The maximum allowed CSV size is 10 MiB.")
        st.stop()

    # Preventing an older inference report from being displayed for
    # a newly uploaded file.
    fingerprint = hashlib.sha256(content).hexdigest()

    if st.session_state.get("upload_fingerprint") != fingerprint:
        st.session_state["upload_fingerprint"] = fingerprint
        st.session_state.pop("prediction_result", None)

    # Providing a single user action for validation and inference.
    st.subheader("2. Generate predictions")

    if st.button("Generate predictions", type="primary"):
        try:
            with st.spinner("Validating the uploaded CSV..."):
                validation = request_csv(
                    "/uploads/validate",
                    uploaded_file.name,
                    content,
                )

            st.info(
                f"Validated {validation['observations']} observations "
                f"across {validation['accounts']} accounts."
            )

            with st.spinner("Processing financial data and running LSTM inference..."):
                result = request_csv(
                    "/predict/upload",
                    uploaded_file.name,
                    content,
                )

            st.session_state["prediction_result"] = result

        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    # Keeping the report visible during account-selection reruns.
    result = st.session_state.get("prediction_result")

    if result is not None:
        st.divider()
        st.subheader("3. Interactive risk report")
        render_report(result)


if __name__ == "__main__":
    main()
