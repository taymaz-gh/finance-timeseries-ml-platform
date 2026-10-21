"""Testing Silver/Gold preparation of uploaded financial observations."""

from io import BytesIO

import pandas as pd
import pytest

from finance_ml.api.csv_validation import read_account_csv
from finance_ml.api.upload_pipeline import prepare_uploaded_gold
from finance_ml.data.generate_synthetic import generate_financial_timeseries


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Generating validated, unlabeled financial observations."""

    frame = generate_financial_timeseries(
        n_accounts=3,
        n_days=14,
        start_date="2026-01-01",
        random_seed=123,
    )

    # Removing supervised labels before inference.
    frame = frame.drop(columns=["risk_state", "risk_state_name"])

    # Passing the generated data through our CSV validator.
    return read_account_csv(BytesIO(frame.to_csv(index=False).encode("utf-8")))


def test_multiple_accounts_and_coverage(spark, raw_frame) -> None:
    """Preparing accounts with different historical lengths."""

    # Selecting 14, 7, and 4 observations for three accounts.
    frame = pd.concat(
        [
            raw_frame[raw_frame["account_id"] == "ACC_00001"],
            raw_frame[raw_frame["account_id"] == "ACC_00002"].head(7),
            raw_frame[raw_frame["account_id"] == "ACC_00003"].head(4),
        ],
        ignore_index=True,
    )

    gold, report = prepare_uploaded_gold(
        frame,
        spark=spark,
        sequence_length=7,
    )

    report = report.set_index("account_id")

    assert gold.count() == 25

    assert report.loc["ACC_00001", "eligible_endpoints"] == 8
    assert report.loc["ACC_00002", "eligible_endpoints"] == 1

    assert report.loc["ACC_00003", "status"] == "insufficient_context"

    assert report.loc["ACC_00003", "missing_for_minimum"] == 3

    # Checking that Gold features were engineered.
    assert "previous_balance" in gold.columns
    assert "balance_change" in gold.columns
    assert "rolling_7d_mean_balance" in gold.columns
    assert "net_cash_flow" in gold.columns

    # Preventing risk labels from entering model features.
    assert "risk_state" not in gold.columns
    assert "risk_state_name" not in gold.columns


def test_deduplication_before_coverage(spark, raw_frame) -> None:
    """Ensuring duplicate records do not count as extra timesteps."""

    frame = raw_frame[raw_frame["account_id"] == "ACC_00001"].head(8).copy()

    # Creating a duplicate with a missing balance.
    duplicate = frame.iloc[[0]].copy()
    duplicate.loc[:, "balance"] = None

    frame = pd.concat(
        [frame, duplicate],
        ignore_index=True,
    )

    gold, report = prepare_uploaded_gold(
        frame,
        spark=spark,
        sequence_length=7,
    )

    assert gold.count() == 8
    assert report.loc[0, "available_observations"] == 8
    assert report.loc[0, "eligible_endpoints"] == 2

    # Checking that Silver selected the more complete record.
    first = gold.orderBy("date").first()

    assert first["balance"] == pytest.approx(float(frame.loc[0, "balance"]))


def test_silver_cleaning_and_gold_features(spark, raw_frame) -> None:
    """Checking causal filling and derived features."""

    frame = raw_frame[raw_frame["account_id"] == "ACC_00001"].head(8).copy()

    dates = frame["date"].sort_values().tolist()

    # Introducing a missing balance on the third day.
    frame.loc[frame["date"] == dates[2], "balance"] = float("nan")

    # Introducing invalid financial values on the fifth day.
    frame.loc[frame["date"] == dates[4], "cash_outflow"] = -10.0

    frame.loc[frame["date"] == dates[4], "credit_utilization"] = 1.2

    gold, _ = prepare_uploaded_gold(
        frame,
        spark=spark,
        sequence_length=7,
    )

    rows = gold.orderBy("date").collect()

    # Checking causal forward filling.
    assert rows[2]["balance"] == pytest.approx(rows[1]["balance"])

    # Checking lag feature engineering.
    assert rows[3]["previous_balance"] == pytest.approx(rows[2]["balance"])

    # Checking Silver invalid-value handling.
    assert rows[4]["cash_outflow"] is None
    assert rows[4]["net_cash_flow"] is None

    assert rows[4]["credit_utilization"] == pytest.approx(rows[3]["credit_utilization"])

    # Checking the seven-day trailing rolling average.
    expected_mean = sum(row["balance"] for row in rows[:7]) / 7

    assert rows[6]["rolling_7d_mean_balance"] == pytest.approx(expected_mean)


def test_all_missing_numeric_column(spark, raw_frame) -> None:
    """Preserving Spark types when a feature is entirely missing."""

    frame = raw_frame[raw_frame["account_id"] == "ACC_00001"].head(8).copy()

    frame["cash_inflow"] = None
    frame["account_type"] = None

    gold, report = prepare_uploaded_gold(
        frame,
        spark=spark,
        sequence_length=7,
    )

    first = gold.orderBy("date").first()

    assert first["cash_inflow"] is None
    assert first["net_cash_flow"] is None
    assert first["account_type"] == "unknown"

    assert report.loc[0, "status"] == "eligible"


@pytest.mark.parametrize(
    "sequence_length, expected_endpoints",
    [
        (4, 5),
        (9, 0),
    ],
)
def test_dynamic_sequence_length(
    spark,
    raw_frame,
    sequence_length,
    expected_endpoints,
) -> None:
    """Checking that inference eligibility is model-dependent."""

    frame = raw_frame[raw_frame["account_id"] == "ACC_00001"].head(8).copy()

    _, report = prepare_uploaded_gold(
        frame,
        spark=spark,
        sequence_length=sequence_length,
    )

    assert report.loc[0, "eligible_endpoints"] == expected_endpoints
