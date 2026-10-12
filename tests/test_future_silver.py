"""Verifying label-free cleaning and unchanged labeled defaults."""
from datetime import datetime

from finance_ml.lakehouse.build_silver import build_silver_dataframe


def test_future_silver_accepts_absent_labels(spark):
    rows = [
        ("A", datetime(2026, 3, 2), "checking", 100.0, -5.0, 1.2, 0.8, 0),
        ("A", datetime(2026, 3, 3), "checking", 105.0, 5.0, 0.3, 0.8, 0),
    ]
    cols = ["account_id", "date", "account_type", "balance", "cash_outflow",
            "credit_utilization", "payment_ratio", "days_past_due"]
    df = spark.createDataFrame(rows, cols)
    out = build_silver_dataframe(df, require_labels=False).orderBy("date").collect()
    assert len(out) == 2
    assert out[0]["cash_outflow"] is None
    assert out[0]["credit_utilization"] is None
    assert "risk_state" not in out[0].asDict()


def test_future_silver_default_requires_labels(spark):
    import pytest
    df = spark.createDataFrame(
        [("A", datetime(2026, 3, 2), "checking", 100.0, 5.0, 0.3, 0.8, 0)],
        ["account_id", "date", "account_type", "balance", "cash_outflow",
         "credit_utilization", "payment_ratio", "days_past_due"],
    )
    with pytest.raises(ValueError, match="target columns"):
        build_silver_dataframe(df)
