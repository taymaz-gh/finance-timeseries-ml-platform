from datetime import datetime

import pytest
from pyspark.sql import SparkSession

from finance_ml.lakehouse.build_silver import build_silver_dataframe



def test_build_silver_prefers_complete_targets_when_deduplicating(
    spark,
) -> None:
    """
    Checking that duplicate rows with complete targets are preferred.

    A duplicate with valid targets should survive even when another
    duplicate has a more complete predictor value.
    """
    rows = [
        (
            "ACC_TEST",
            datetime(2026, 1, 1),
            "checking",
            None,
            100.0,
            0.4,
            0.8,
            0,
            0,
            "normal",
        ),
        (
            "ACC_TEST",
            datetime(2026, 1, 1),
            "checking",
            5000.0,
            100.0,
            0.4,
            0.8,
            0,
            None,
            None,
        ),
    ]

    columns = [
        "account_id",
        "date",
        "account_type",
        "balance",
        "cash_outflow",
        "credit_utilization",
        "payment_ratio",
        "days_past_due",
        "risk_state",
        "risk_state_name",
    ]

    bronze_df = spark.createDataFrame(
        rows,
        columns,
    )

    silver_df = build_silver_dataframe(bronze_df)

    result = silver_df.collect()

    assert len(result) == 1
    assert result[0]["risk_state"] == 0
    assert result[0]["risk_state_name"] == "normal"
    assert result[0]["balance"] is None


def test_build_silver_uses_only_past_values_for_imputation(
    spark,
) -> None:
    """
    Checking that missing state values use only earlier observations.

    A future value must never be used to fill an earlier missing value.
    """
    rows = [
        (
            "ACC_TEST",
            datetime(2026, 1, 1),
            "checking",
            1000.0,
            100.0,
            0.4,
            0.8,
            0,
            0,
            "normal",
        ),
        (
            "ACC_TEST",
            datetime(2026, 1, 2),
            "checking",
            None,
            100.0,
            0.4,
            0.8,
            0,
            0,
            "normal",
        ),
        (
            "ACC_TEST",
            datetime(2026, 1, 3),
            "checking",
            3000.0,
            100.0,
            0.4,
            0.8,
            0,
            0,
            "normal",
        ),
    ]

    columns = [
        "account_id",
        "date",
        "account_type",
        "balance",
        "cash_outflow",
        "credit_utilization",
        "payment_ratio",
        "days_past_due",
        "risk_state",
        "risk_state_name",
    ]

    bronze_df = spark.createDataFrame(
        rows,
        columns,
    )

    silver_df = build_silver_dataframe(bronze_df)

    result = silver_df.orderBy("date").collect()

    assert result[0]["balance"] == 1000.0
    assert result[1]["balance"] == 1000.0
    assert result[2]["balance"] == 3000.0


def test_build_silver_converts_invalid_values_to_null(
    spark,
) -> None:
    """
    Checking that structurally invalid numeric values become NULL.
    """
    rows = [
        (
            "ACC_TEST",
            datetime(2026, 1, 1),
            "checking",
            1000.0,
            -50.0,
            1.2,
            0.8,
            0,
            0,
            "normal",
        ),
    ]

    columns = [
        "account_id",
        "date",
        "account_type",
        "balance",
        "cash_outflow",
        "credit_utilization",
        "payment_ratio",
        "days_past_due",
        "risk_state",
        "risk_state_name",
    ]

    bronze_df = spark.createDataFrame(
        rows,
        columns,
    )

    silver_df = build_silver_dataframe(bronze_df)

    result = silver_df.collect()

    assert len(result) == 1
    assert result[0]["cash_outflow"] is None
    assert result[0]["credit_utilization"] is None


def test_build_silver_drops_rows_with_missing_targets(
    spark,
) -> None:
    """
    Checking that rows with incomplete target labels are removed.
    """
    rows = [
        (
            "ACC_KEEP",
            datetime(2026, 1, 1),
            "checking",
            1000.0,
            100.0,
            0.4,
            0.8,
            0,
            0,
            "normal",
        ),
        (
            "ACC_DROP_1",
            datetime(2026, 1, 1),
            "checking",
            1000.0,
            100.0,
            0.4,
            0.8,
            0,
            None,
            "normal",
        ),
        (
            "ACC_DROP_2",
            datetime(2026, 1, 1),
            "checking",
            1000.0,
            100.0,
            0.4,
            0.8,
            0,
            1,
            None,
        ),
    ]

    columns = [
        "account_id",
        "date",
        "account_type",
        "balance",
        "cash_outflow",
        "credit_utilization",
        "payment_ratio",
        "days_past_due",
        "risk_state",
        "risk_state_name",
    ]

    bronze_df = spark.createDataFrame(
        rows,
        columns,
    )

    silver_df = build_silver_dataframe(bronze_df)

    result = silver_df.collect()

    assert len(result) == 1
    assert result[0]["account_id"] == "ACC_KEEP"
