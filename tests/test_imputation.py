from datetime import datetime

import pytest
from pyspark.sql import SparkSession

from finance_ml.preprocessing.imputation import (
    apply_numeric_imputation,
    fit_numeric_medians,
)



def test_fit_numeric_medians_uses_training_data_only(
    spark,
) -> None:
    """
    Checking that medians are fitted only from training data.
    """
    train_df = spark.createDataFrame(
        [
            ("A", datetime(2026, 1, 1), 10.0),
            ("A", datetime(2026, 1, 2), 20.0),
            ("A", datetime(2026, 1, 3), None),
            ("A", datetime(2026, 1, 4), 100.0),
        ],
        ["account_id", "date", "value"],
    )

    medians = fit_numeric_medians(
        train_df,
        ["value"],
    )

    assert medians["value"] == 20.0


def test_apply_numeric_imputation_adds_indicator(
    spark,
) -> None:
    """
    Checking median imputation and missing-indicator creation.
    """
    df = spark.createDataFrame(
        [
            ("A", 10.0),
            ("B", None),
        ],
        ["account_id", "value"],
    )

    transformed_df = apply_numeric_imputation(
        df,
        {"value": 20.0},
    )

    result = {row["account_id"]: row for row in transformed_df.collect()}

    assert result["A"]["value"] == 10.0
    assert result["A"]["value_missing"] == 0

    assert result["B"]["value"] == 20.0
    assert result["B"]["value_missing"] == 1


def test_apply_numeric_imputation_preserves_schema_across_splits(
    spark,
) -> None:
    """
    Checking that train-fitted medians create the same columns
    when applied to validation or test data.
    """
    validation_df = spark.createDataFrame(
        [
            ("A", 5.0),
            ("B", None),
        ],
        ["account_id", "value"],
    )

    transformed_df = apply_numeric_imputation(
        validation_df,
        {"value": 20.0},
    )

    assert "value" in transformed_df.columns
    assert "value_missing" in transformed_df.columns
