from datetime import datetime

import pytest
from pyspark.sql import SparkSession

from finance_ml.preprocessing.split import temporal_split


@pytest.fixture(scope="module")
def spark():
    """Creating a local Spark session for temporal split tests."""
    session = (
        SparkSession.builder.master("local[1]").appName("split-tests").getOrCreate()
    )

    yield session

    session.stop()


def test_temporal_split_uses_distinct_timestamps(
    spark,
) -> None:
    """
    Checking that splitting is based on distinct timestamps,
    not row counts.
    """
    rows = []

    for day in range(1, 11):
        for account_id in ["A", "B"]:
            rows.append(
                (
                    account_id,
                    datetime(2026, 1, day),
                    float(day),
                )
            )

    df = spark.createDataFrame(
        rows,
        ["account_id", "date", "value"],
    )

    (
        train_df,
        validation_df,
        test_df,
        train_end,
        validation_end,
    ) = temporal_split(
        df,
        time_col="date",
        train_fraction=0.60,
        validation_fraction=0.20,
    )

    assert train_df.count() == 12
    assert validation_df.count() == 4
    assert test_df.count() == 4

    assert train_end == datetime(2026, 1, 6)
    assert validation_end == datetime(2026, 1, 8)


def test_temporal_split_keeps_same_timestamp_together(
    spark,
) -> None:
    """
    Checking that rows sharing one timestamp cannot be split apart.
    """
    rows = [("A", datetime(2026, 1, day)) for day in range(1, 7)] + [
        ("B", datetime(2026, 1, day)) for day in range(1, 7)
    ]

    df = spark.createDataFrame(
        rows,
        ["account_id", "date"],
    )

    train_df, validation_df, test_df, _, _ = temporal_split(
        df,
        time_col="date",
        train_fraction=0.50,
        validation_fraction=0.25,
    )

    train_dates = {row["date"] for row in train_df.select("date").distinct().collect()}

    validation_dates = {
        row["date"] for row in validation_df.select("date").distinct().collect()
    }

    test_dates = {row["date"] for row in test_df.select("date").distinct().collect()}

    assert train_dates.isdisjoint(validation_dates)
    assert train_dates.isdisjoint(test_dates)
    assert validation_dates.isdisjoint(test_dates)


def test_temporal_split_rejects_invalid_fractions(
    spark,
) -> None:
    """Checking validation of invalid split fractions."""
    df = spark.createDataFrame(
        [
            ("A", datetime(2026, 1, 1)),
            ("A", datetime(2026, 1, 2)),
            ("A", datetime(2026, 1, 3)),
        ],
        ["account_id", "date"],
    )

    with pytest.raises(ValueError):
        temporal_split(
            df,
            time_col="date",
            train_fraction=0.90,
            validation_fraction=0.20,
        )
