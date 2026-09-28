import pytest
from pyspark.sql import SparkSession

from finance_ml.data.encoding import (
    apply_one_hot_encoding,
    fit_category_levels,
)


@pytest.fixture(scope="module")
def spark():
    """Creating a local Spark session for encoding tests."""
    session = (
        SparkSession.builder.master("local[1]").appName("encoding-tests").getOrCreate()
    )

    yield session

    session.stop()


def test_fit_category_levels_uses_training_data_only(
    spark,
) -> None:
    """Checking that category levels are learned from training data."""
    train_df = spark.createDataFrame(
        [
            ("A", "checking"),
            ("B", "credit"),
            ("C", "business"),
            ("D", "checking"),
        ],
        ["account_id", "account_type"],
    )

    levels = fit_category_levels(
        train_df,
        ["account_type"],
    )

    assert levels == {
        "account_type": [
            "business",
            "checking",
            "credit",
        ]
    }


def test_apply_one_hot_encoding_creates_expected_columns(
    spark,
) -> None:
    """Checking that one-hot columns are created correctly."""
    df = spark.createDataFrame(
        [
            ("A", "checking"),
            ("B", "business"),
        ],
        ["account_id", "account_type"],
    )

    transformed_df = apply_one_hot_encoding(
        df,
        {
            "account_type": [
                "business",
                "checking",
                "credit",
            ]
        },
    )

    result = {row["account_id"]: row for row in transformed_df.collect()}

    assert result["A"]["account_type_business"] == 0
    assert result["A"]["account_type_checking"] == 1
    assert result["A"]["account_type_credit"] == 0

    assert result["B"]["account_type_business"] == 1
    assert result["B"]["account_type_checking"] == 0
    assert result["B"]["account_type_credit"] == 0


def test_unseen_category_produces_zero_vector(
    spark,
) -> None:
    """
    Checking that an unseen validation/test category does not change
    the training-defined feature schema.
    """
    df = spark.createDataFrame(
        [
            ("A", "savings"),
        ],
        ["account_id", "account_type"],
    )

    transformed_df = apply_one_hot_encoding(
        df,
        {
            "account_type": [
                "business",
                "checking",
                "credit",
            ]
        },
    )

    row = transformed_df.first()

    assert row["account_type_business"] == 0
    assert row["account_type_checking"] == 0
    assert row["account_type_credit"] == 0
