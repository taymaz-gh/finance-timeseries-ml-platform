"""Preparing local synthetic Gold training data through Bronze, Silver, and Gold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from finance_ml.data.generate_synthetic import generate_financial_timeseries
from finance_ml.data.inject_quality_issues import inject_data_quality_issues


MAX_SYNTHETIC_TRAINING_ROWS = 20_000


def _to_spark_value(value: object) -> object:
    """Converting pandas/NumPy scalar values to Spark-compatible values."""
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    if isinstance(value, np.generic):
        value = value.item()

    if pd.isna(value):
        return None

    return value


def prepare_synthetic_gold(
    spark,
    *,
    n_accounts: int = 20,
    n_days: int = 60,
    random_seed: int = 42,
):
    """Building local synthetic Gold through reproducible Bronze corruption."""
    from finance_ml.lakehouse.build_silver import build_silver_dataframe
    from finance_ml.lakehouse.build_gold import build_gold_features

    # Validating bounded local simulation parameters.
    for key, value in {
        "n_accounts": n_accounts,
        "n_days": n_days,
    }.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")

    if type(random_seed) is not int or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")

    if n_accounts * n_days > MAX_SYNTHETIC_TRAINING_ROWS:
        raise ValueError(
            f"Synthetic training exceeds the "
            f"{MAX_SYNTHETIC_TRAINING_ROWS:,}-row limit"
        )

    # Generating reproducible clean observations.
    clean = generate_financial_timeseries(
        n_accounts=n_accounts,
        n_days=n_days,
        random_seed=random_seed,
    )

    # Reproducing the original Bronze corruption policy.
    # Keeping its RNG separate from the financial simulator.
    bronze_pd = inject_data_quality_issues(
        clean,
        random_seed=123,
    )

    # Preserving the original pandas column ordering.
    columns = list(bronze_pd.columns)

    # Converting NaN/NaT to actual Spark nulls rather than NaN values.
    records = [
        tuple(_to_spark_value(record[column]) for column in columns)
        for record in bronze_pd.to_dict(orient="records")
    ]

    bronze = spark.createDataFrame(
        records,
        schema=columns,
    )

    print(
        "Synthetic Bronze preparation:",
        f"{len(clean)} clean rows,",
        f"{len(bronze_pd)} Bronze rows,",
        f"{len(bronze_pd) - len(clean)} injected duplicates",
    )

    # Cleaning invalid values and duplicate account/date keys.
    silver = build_silver_dataframe(
        bronze,
        require_labels=True,
    )

    # Engineering lag, change, and rolling-window predictors.
    return build_gold_features(silver)
