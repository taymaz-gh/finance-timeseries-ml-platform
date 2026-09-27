from datetime import datetime

import pytest
from pyspark.sql import SparkSession

from finance_ml.features.build_gold import build_gold_features


@pytest.fixture(scope="module")
def spark():
    """Creating a local Spark session for Gold behavioral tests."""
    session = (
        SparkSession.builder.master("local[1]").appName("gold-tests").getOrCreate()
    )

    yield session

    session.stop()


def test_build_gold_creates_lag_and_change_features(
    spark,
) -> None:
    """
    Checking that lag and change features use the previous timestep
    from the same account.
    """
    rows = [
        (
            "ACC_TEST",
            datetime(2026, 1, 1),
            1000.0,
            100.0,
            150.0,
            0.4,
            200.0,
            180.0,
            0.9,
            0,
            0,
            "normal",
        ),
        (
            "ACC_TEST",
            datetime(2026, 1, 2),
            1100.0,
            120.0,
            170.0,
            0.5,
            200.0,
            190.0,
            0.95,
            1,
            1,
            "elevated_risk",
        ),
    ]

    columns = [
        "account_id",
        "date",
        "balance",
        "cash_inflow",
        "cash_outflow",
        "credit_utilization",
        "amount_due",
        "amount_paid",
        "payment_ratio",
        "days_past_due",
        "risk_state",
        "risk_state_name",
    ]

    silver_df = spark.createDataFrame(
        rows,
        columns,
    )

    gold_df = build_gold_features(silver_df)

    result = gold_df.orderBy("date").collect()

    assert result[0]["previous_balance"] is None
    assert result[0]["balance_change"] is None

    assert result[1]["previous_balance"] == 1000.0
    assert result[1]["balance_change"] == 100.0

    assert result[1]["previous_credit_utilization"] == 0.4
    assert result[1]["credit_utilization_change"] == pytest.approx(0.1)

    assert result[1]["previous_payment_ratio"] == 0.9
    assert result[1]["payment_ratio_change"] == pytest.approx(0.05)


def test_build_gold_creates_trailing_rolling_features(
    spark,
) -> None:
    """
    Checking that rolling features use only the current and earlier rows.
    """
    rows = [
        (
            "ACC_TEST",
            datetime(2026, 1, day),
            float(day * 100),
            50.0,
            float(day * 10),
            0.4,
            100.0,
            90.0,
            float(day) / 10.0,
            day - 1,
            0,
            "normal",
        )
        for day in range(1, 9)
    ]

    columns = [
        "account_id",
        "date",
        "balance",
        "cash_inflow",
        "cash_outflow",
        "credit_utilization",
        "amount_due",
        "amount_paid",
        "payment_ratio",
        "days_past_due",
        "risk_state",
        "risk_state_name",
    ]

    silver_df = spark.createDataFrame(
        rows,
        columns,
    )

    gold_df = build_gold_features(silver_df)

    result = gold_df.orderBy("date").collect()

    # Day 1: only one observation is available.
    assert result[0]["rolling_7d_mean_balance"] == 100.0
    assert result[0]["rolling_7d_mean_cash_outflow"] == 10.0
    assert result[0]["rolling_7d_max_days_past_due"] == 0

    # Day 7: the window contains days 1 through 7.
    assert result[6]["rolling_7d_mean_balance"] == 400.0
    assert result[6]["rolling_7d_mean_cash_outflow"] == 40.0
    assert result[6]["rolling_7d_mean_payment_ratio"] == pytest.approx(0.4)
    assert result[6]["rolling_7d_max_days_past_due"] == 6

    # Day 8: the window must contain only days 2 through 8.
    assert result[7]["rolling_7d_mean_balance"] == 500.0
    assert result[7]["rolling_7d_mean_cash_outflow"] == 50.0
    assert result[7]["rolling_7d_mean_payment_ratio"] == pytest.approx(0.5)
    assert result[7]["rolling_7d_max_days_past_due"] == 7


def test_build_gold_creates_same_day_derived_features(
    spark,
) -> None:
    """
    Checking same-day derived features and missing-value propagation.
    """
    rows = [
        (
            "ACC_TEST",
            datetime(2026, 1, 1),
            1000.0,
            200.0,
            150.0,
            0.4,
            120.0,
            100.0,
            0.8,
            0,
            0,
            "normal",
        ),
        (
            "ACC_TEST",
            datetime(2026, 1, 2),
            1100.0,
            200.0,
            None,
            0.5,
            120.0,
            90.0,
            0.75,
            0,
            0,
            "normal",
        ),
    ]

    columns = [
        "account_id",
        "date",
        "balance",
        "cash_inflow",
        "cash_outflow",
        "credit_utilization",
        "amount_due",
        "amount_paid",
        "payment_ratio",
        "days_past_due",
        "risk_state",
        "risk_state_name",
    ]

    silver_df = spark.createDataFrame(
        rows,
        columns,
    )

    gold_df = build_gold_features(silver_df)

    result = gold_df.orderBy("date").collect()

    assert result[0]["net_cash_flow"] == 50.0
    assert result[0]["payment_gap"] == 20.0

    assert result[1]["net_cash_flow"] is None
    assert result[1]["payment_gap"] == 30.0


def test_build_gold_uses_date_aware_rolling_window(
    spark,
) -> None:
    """
    Checking that the 7-day rolling window uses calendar time,
    not simply the previous six rows.
    """
    days = [1, 2, 3, 4, 5, 6, 8]

    rows = [
        (
            "ACC_TEST",
            datetime(2026, 1, day),
            float(day * 100),
            50.0,
            float(day * 10),
            0.4,
            100.0,
            90.0,
            float(day) / 10.0,
            day - 1,
            0,
            "normal",
        )
        for day in days
    ]

    columns = [
        "account_id",
        "date",
        "balance",
        "cash_inflow",
        "cash_outflow",
        "credit_utilization",
        "amount_due",
        "amount_paid",
        "payment_ratio",
        "days_past_due",
        "risk_state",
        "risk_state_name",
    ]

    silver_df = spark.createDataFrame(
        rows,
        columns,
    )

    gold_df = build_gold_features(silver_df)

    result = gold_df.orderBy("date").collect()

    jan_8 = result[-1]

    assert jan_8["rolling_7d_mean_balance"] == pytest.approx(
        (200 + 300 + 400 + 500 + 600 + 800) / 6
    )

    assert jan_8["rolling_7d_mean_cash_outflow"] == pytest.approx(
        (20 + 30 + 40 + 50 + 60 + 80) / 6
    )
