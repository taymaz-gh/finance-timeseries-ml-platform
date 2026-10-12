"""Testing independent holdout paths without writing Unity Catalog tables."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pyspark.sql import functions as F

from finance_ml.data.generate_synthetic import generate_financial_timeseries
from finance_ml.pipelines.holdout_data_pipeline import (
    DEFAULT_TABLES, LABEL_COLUMNS, _validate_tables, prepare_holdout_data,
)


def _history_silver(spark, *, accounts=3, days=10):
    history = generate_financial_timeseries(n_accounts=accounts, n_days=days)
    history["date"] = history["date"].map(
        lambda timestamp: timestamp.to_pydatetime()
    )
    return spark.createDataFrame(history)


def test_table_guards_reject_originals_and_duplicate_names():
    safe = dict(DEFAULT_TABLES)
    _validate_tables(safe, "finance_ml.silver.account_events_clean")

    dangerous = dict(safe, temporal_gold="finance_ml.gold.account_features")
    with pytest.raises(ValueError, match="Refusing"):
        _validate_tables(dangerous, "finance_ml.silver.account_events_clean")

    duplicate = dict(safe, account_bronze=safe["temporal_bronze"])
    with pytest.raises(ValueError, match="distinct"):
        _validate_tables(duplicate, "finance_ml.silver.account_events_clean")

    non_holdout = dict(safe, account_truth="finance_ml.gold.arbitrary_labels")
    with pytest.raises(ValueError, match="holdout"):
        _validate_tables(non_holdout, "finance_ml.silver.account_events_clean")


@pytest.mark.parametrize("inject", [False, True])
def test_holdout_end_to_end_spark_without_persistence(spark, monkeypatch, inject):
    # A deliberately small, deterministic training history; no remote reads.
    history = _history_silver(spark)
    calls = []
    original_table = spark.table

    def table(name):
        calls.append(name)
        assert name == "finance_ml.silver.account_events_clean"
        return history

    monkeypatch.setattr(spark, "table", table)
    out = prepare_holdout_data(
        spark, n_existing_accounts=3, historical_days=10,
        n_future_days=7, n_new_accounts=2,
        inject_quality_issues=inject, validate=True, persist=False,
    )
    assert calls
    assert out["persisted"] is False
    assert out["history_end"] == datetime(2026, 1, 10)
    assert out["future_start"] == datetime(2026, 1, 11)

    temporal, account = out["temporal"], out["account"]
    assert temporal.gold.count() == 21
    assert temporal.context.count() == 39  # 6 historical + 7 future, per account
    assert account.gold.count() == 34
    assert temporal.truth.count() == 21
    assert account.truth.count() == 34
    assert set(temporal.gold.select("account_id").distinct().toPandas().account_id) == {
        "ACC_00001", "ACC_00002", "ACC_00003"
    }
    assert set(account.gold.select("account_id").distinct().toPandas().account_id) == {
        "ACC_00004", "ACC_00005"
    }
    for group in (temporal, account):
        for layer in ("bronze", "silver", "gold", "context"):
            assert not set(LABEL_COLUMNS).intersection(getattr(group, layer).columns)
        assert set(LABEL_COLUMNS).issubset(group.truth.columns)
        if inject:
            assert group.bronze.count() > group.silver.count()

    # First future previous-balance feature MUST come from Jan 10 history.
    for account_id in ["ACC_00001", "ACC_00002", "ACC_00003"]:
        actual = temporal.gold.filter(
            (F.col("account_id") == account_id) &
            (F.col("date") == F.lit(datetime(2026, 1, 11)))
        ).select("previous_balance").first()[0]
        expected = history.filter(
            (F.col("account_id") == account_id) &
            (F.col("date") == F.lit(datetime(2026, 1, 10)))
        ).select("balance").first()[0]
        assert actual == expected


def test_fails_closed_on_historical_cutoff_mismatch(spark, monkeypatch):
    history = _history_silver(spark, days=9)
    monkeypatch.setattr(spark, "table", lambda name: history)
    with pytest.raises(ValueError, match="cutoff"):
        prepare_holdout_data(
            spark, n_existing_accounts=3, historical_days=10,
            n_future_days=2, n_new_accounts=1, inject_quality_issues=False,
        )
