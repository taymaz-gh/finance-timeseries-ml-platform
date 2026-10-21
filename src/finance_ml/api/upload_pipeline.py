"""Preparing uploaded account-day observations for model inference."""

from __future__ import annotations

from typing import Any

import pandas as pd

from finance_ml.api.coverage import assess_account_context
from finance_ml.api.csv_validation import (
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
)
from finance_ml.lakehouse.build_gold import build_gold_features
from finance_ml.lakehouse.build_silver import build_silver_dataframe


def prepare_uploaded_gold(
    frame: pd.DataFrame,
    *,
    spark: Any,
    sequence_length: int,
) -> tuple[Any, pd.DataFrame]:
    """Preparing Silver/Gold features and per-account context reports."""

    from pyspark.sql.types import (
        DoubleType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    # Checking that the validated raw schema is preserved.
    if frame.empty:
        raise ValueError("No account observations provided")

    if set(frame.columns) != set(REQUIRED_COLUMNS):
        raise ValueError("Uploaded data does not match the raw schema")

    # Defining Spark types explicitly, including nullable numeric fields.
    schema = StructType(
        [
            StructField("account_id", StringType(), nullable=False),
            StructField("date", TimestampType(), nullable=False),
            StructField("account_type", StringType(), nullable=True),
            *[
                StructField(column, DoubleType(), nullable=True)
                for column in NUMERIC_COLUMNS
            ],
        ]
    )

    records = []

    # Converting pandas missing values to Spark-compatible NULL values.
    for record in frame.to_dict(orient="records"):

        row = {
            "account_id": str(record["account_id"]),
            "date": pd.Timestamp(record["date"]).to_pydatetime(),
            "account_type": (
                None if pd.isna(record["account_type"]) else str(record["account_type"])
            ),
        }

        for column in NUMERIC_COLUMNS:
            value = record[column]
            row[column] = None if pd.isna(value) else float(value)

        records.append(row)

    # Creating the Bronze representation without persisting uploaded data.
    bronze = spark.createDataFrame(records, schema=schema)

    # Reusing the established cleaning and feature-engineering functions.
    silver = build_silver_dataframe(
        bronze,
        require_labels=False,
    )

    gold = build_gold_features(silver)

    # Checking sequence eligibility after Silver deduplication.
    account_dates = silver.select("account_id", "date").toPandas()

    context_report = assess_account_context(
        account_dates,
        sequence_length=sequence_length,
    )

    return gold, context_report
