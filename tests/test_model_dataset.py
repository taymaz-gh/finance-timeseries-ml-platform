from datetime import datetime

import pytest
from pyspark.sql import SparkSession

from finance_ml.data.model_dataset import get_model_columns


@pytest.fixture(scope="module")
def spark():
    """Creating a local Spark session for model-dataset tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("model-dataset-tests")
        .getOrCreate()
    )

    yield session

    session.stop()


def test_get_model_columns_separates_feature_types(
    spark,
) -> None:
    """Checking model, target, identifier, and categorical columns."""
    df = spark.createDataFrame(
        [
            (
                "ACC_1",
                datetime(2026, 1, 1),
                "checking",
                1000.0,
                0.4,
                0,
                "normal",
            )
        ],
        [
            "account_id",
            "date",
            "account_type",
            "balance",
            "credit_utilization",
            "risk_state",
            "risk_state_name",
        ],
    )

    columns = get_model_columns(df)

    assert columns["feature_columns"] == [
        "account_type",
        "balance",
        "credit_utilization",
    ]

    assert columns["numeric_feature_columns"] == [
        "balance",
        "credit_utilization",
    ]

    assert columns["categorical_feature_columns"] == [
        "account_type",
    ]

    assert columns["target_column"] == "risk_state"
    assert columns["target_name_column"] == "risk_state_name"

    assert columns["id_columns"] == [
        "account_id",
        "date",
    ]
