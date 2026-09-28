from datetime import datetime, timedelta

import pytest
from pyspark.sql import SparkSession

from finance_ml.data.model_dataset_pipeline import (
    prepare_model_dataset,
)


@pytest.fixture(scope="module")
def spark():
    """Creating a local Spark session for model-dataset pipeline tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("model-dataset-pipeline-tests")
        .getOrCreate()
    )

    yield session

    session.stop()


def test_prepare_model_dataset_end_to_end(
    spark,
) -> None:
    """
    Checking the complete model-dataset preparation workflow.

    The test verifies:
    - chronological splitting
    - imputation
    - categorical encoding
    - final feature-schema construction
    - MAD-based scaling
    - sequence construction
    """
    rows = []

    start_date = datetime(2026, 1, 1)

    for account_id, account_type in [
        ("A", "checking"),
        ("B", "credit"),
    ]:
        for offset in range(10):
            date = start_date + timedelta(days=offset)

            rows.append(
                (
                    account_id,
                    date,
                    account_type,
                    1000.0 + offset,
                    100.0,
                    80.0,
                    5,
                    0.4,
                    100.0,
                    90.0,
                    0.9,
                    0,
                    20.0,
                    10.0,
                    999.0 + offset,
                    0.39,
                    0.89,
                    0,
                    1.0,
                    0.01,
                    0.01,
                    1000.0,
                    80.0,
                    0.9,
                    0,
                    offset % 4,
                    [
                        "normal",
                        "elevated_risk",
                        "liquidity_stress",
                        "delinquency_risk",
                    ][offset % 4],
                )
            )

    columns = [
        "account_id",
        "date",
        "account_type",
        "balance",
        "cash_inflow",
        "cash_outflow",
        "transaction_count",
        "credit_utilization",
        "amount_due",
        "amount_paid",
        "payment_ratio",
        "days_past_due",
        "net_cash_flow",
        "payment_gap",
        "previous_balance",
        "previous_credit_utilization",
        "previous_payment_ratio",
        "previous_days_past_due",
        "balance_change",
        "credit_utilization_change",
        "payment_ratio_change",
        "rolling_7d_mean_balance",
        "rolling_7d_mean_cash_outflow",
        "rolling_7d_mean_payment_ratio",
        "rolling_7d_max_days_past_due",
        "risk_state",
        "risk_state_name",
    ]

    gold_df = spark.createDataFrame(
        rows,
        columns,
    )

    result = prepare_model_dataset(
        gold_df,
        train_fraction=0.60,
        validation_fraction=0.20,
        sequence_length=2,
        stride=1,
        scaling_method="mad",
    )

    # Checking sequence shapes.
    assert result["X_train"].shape[1] == 2
    assert result["y_train"].shape[1] == 2

    assert result["X_validation"].shape[1] == 2
    assert result["y_validation"].shape[1] == 2

    assert result["X_test"].shape[1] == 2
    assert result["y_test"].shape[1] == 2

    # Checking that the model-input dimension matches the
    # final feature schema.
    assert result["X_train"].shape[2] == len(result["final_feature_columns"])

    assert result["X_validation"].shape[2] == len(result["final_feature_columns"])

    assert result["X_test"].shape[2] == len(result["final_feature_columns"])

    # Checking training-derived categorical levels.
    assert result["category_levels"] == {
        "account_type": [
            "checking",
            "credit",
        ]
    }

    assert "account_type_checking" in result["final_feature_columns"]

    assert "account_type_credit" in result["final_feature_columns"]

    # Checking the configured scaling strategy.
    assert result["scaling_method"] == "mad"

    assert all(
        parameters["method"] == "mad"
        for parameters in result["scaler_parameters"].values()
    )

    # Checking that fitted scaler parameters contain
    # the expected robust-scaling fields.
    for parameters in result["scaler_parameters"].values():
        assert "center" in parameters
        assert "scale" in parameters
        assert parameters["scale"] > 0

    # Checking temporal split boundaries.
    assert result["train_end"] == datetime(
        2026,
        1,
        6,
    )

    assert result["validation_end"] == datetime(
        2026,
        1,
        8,
    )

    # Checking that the pipeline returns the expected
    # intermediate Spark DataFrames.
    assert result["model_train_df"].count() == 12

    assert result["model_validation_df"].count() == 4

    assert result["model_test_df"].count() == 4
