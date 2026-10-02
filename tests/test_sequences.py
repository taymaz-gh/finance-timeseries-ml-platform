import numpy as np
import pytest
from pyspark.sql import SparkSession

from finance_ml.preprocessing.sequences import build_sequences



def test_build_sequences_preserves_group_boundaries_and_order(
    spark,
) -> None:
    """
    Checking that sequences never cross groups and preserve time order.
    """
    rows = [
        ("A", 1, 10.0, 0),
        ("A", 2, 20.0, 1),
        ("A", 3, 30.0, 2),
        ("B", 1, 100.0, 3),
        ("B", 2, 200.0, 2),
        ("B", 3, 300.0, 1),
    ]

    df = spark.createDataFrame(
        rows,
        ["account_id", "time", "value", "risk_state"],
    )

    X, y, metadata = build_sequences(
        df,
        feature_columns=["value"],
        target_col="risk_state",
        group_col="account_id",
        time_col="time",
        sequence_length=2,
        stride=1,
    )

    assert X.shape == (4, 2, 1)
    assert y.shape == (4, 2)

    np.testing.assert_array_equal(
        X[:, :, 0],
        np.array(
            [
                [10.0, 20.0],
                [20.0, 30.0],
                [100.0, 200.0],
                [200.0, 300.0],
            ],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        y,
        np.array(
            [
                [0, 1],
                [1, 2],
                [3, 2],
                [2, 1],
            ],
            dtype=np.int64,
        ),
    )

    assert metadata[0]["group"] == "A"
    assert metadata[1]["group"] == "A"
    assert metadata[2]["group"] == "B"
    assert metadata[3]["group"] == "B"


def test_build_sequences_respects_stride(
    spark,
) -> None:
    """Checking that stride controls the window advancement."""
    rows = [("A", time, float(time), time % 4) for time in range(1, 7)]

    df = spark.createDataFrame(
        rows,
        ["account_id", "time", "value", "risk_state"],
    )

    X, y, metadata = build_sequences(
        df,
        feature_columns=["value"],
        target_col="risk_state",
        group_col="account_id",
        time_col="time",
        sequence_length=3,
        stride=2,
    )

    assert X.shape == (2, 3, 1)
    assert y.shape == (2, 3)

    np.testing.assert_array_equal(
        X[:, :, 0],
        np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
            ],
            dtype=np.float32,
        ),
    )

    assert metadata[0]["start_time"] == 1
    assert metadata[0]["end_time"] == 3

    assert metadata[1]["start_time"] == 3
    assert metadata[1]["end_time"] == 5


def test_build_sequences_skips_groups_that_are_too_short(
    spark,
) -> None:
    """
    Checking that groups shorter than sequence_length create no sequence.
    """
    rows = [
        ("A", 1, 10.0, 0),
        ("A", 2, 20.0, 1),
    ]

    df = spark.createDataFrame(
        rows,
        ["account_id", "time", "value", "risk_state"],
    )

    X, y, metadata = build_sequences(
        df,
        feature_columns=["value"],
        target_col="risk_state",
        group_col="account_id",
        time_col="time",
        sequence_length=3,
    )

    assert X.shape[0] == 0
    assert y.shape[0] == 0
    assert metadata == []
