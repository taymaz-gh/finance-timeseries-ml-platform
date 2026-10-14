"""Preparing synthetic Gold training data without Databricks access."""

from __future__ import annotations

import pandas as pd
import numpy as np

from finance_ml.data.generate_synthetic import generate_financial_timeseries


MAX_SYNTHETIC_TRAINING_ROWS = 20_000


def prepare_synthetic_gold(
    spark,
    *,
    n_accounts: int = 20,
    n_days: int = 60,
    random_seed: int = 42,
):
    """Build synthetic training data through the existing Silver/Gold layers."""
    from finance_ml.lakehouse.build_silver import build_silver_dataframe
    from finance_ml.lakehouse.build_gold import build_gold_features

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

    historical = generate_financial_timeseries(
        n_accounts=n_accounts,
        n_days=n_days,
        random_seed=random_seed,
    )

    # Converting pandas and NumPy scalar values to native Python types.
    # Preserving simulation values while making records Spark-compatible.
    records = historical.to_dict(orient="records")

    for record in records:
        for column, value in record.items():
            if isinstance(value, np.generic):
                record[column] = value.item()

            elif isinstance(value, pd.Timestamp):
                record[column] = value.to_pydatetime()

    bronze = spark.createDataFrame(records)

    silver = build_silver_dataframe(bronze, require_labels=True)
    return build_gold_features(silver)
