"""Integration testing synthetic training preparation using real local Spark."""

from __future__ import annotations

import numpy as np

from finance_ml.data.synthetic_training import prepare_synthetic_gold
from finance_ml.pipelines.model_dataset_pipeline import prepare_model_dataset


def test_synthetic_gold_and_model_dataset(spark) -> None:
    """Preparing real Spark Gold features and model-ready LSTM sequences."""
    n_accounts = 8
    n_days = 60
    sequence_length = 7

    gold = prepare_synthetic_gold(
        spark,
        n_accounts=n_accounts,
        n_days=n_days,
        random_seed=42,
    )

    # Verifying the generated historical data and Gold transformations.
    assert gold.count() == n_accounts * n_days

    required_columns = {
        "account_id",
        "date",
        "risk_state",
        "risk_state_name",
        "previous_balance",
        "net_cash_flow",
        "rolling_7d_mean_balance",
    }
    assert required_columns.issubset(set(gold.columns))

    dataset = prepare_model_dataset(
        gold,
        sequence_length=sequence_length,
        stride=1,
        scaling_method="mad",
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    feature_count = len(dataset["final_feature_columns"])
    assert feature_count > 0

    # Verifying all three chronological partitions.
    for x_key, y_key in (
        ("X_train", "y_train"),
        ("X_validation", "y_validation"),
        ("X_test", "y_test"),
    ):
        x = dataset[x_key]
        y = dataset[y_key]

        assert x.ndim == 3
        assert y.ndim == 2

        assert x.shape[0] > 0
        assert x.shape[1:] == (sequence_length, feature_count)
        assert y.shape == (x.shape[0], sequence_length)

        assert np.isfinite(x).all()
        assert np.isin(y, [0, 1, 2, 3]).all()

    # Verifying strictly ordered split boundaries.
    assert dataset["train_end"] < dataset["validation_end"]

    # Verifying all necessary preprocessing metadata exists.
    assert dataset["medians"]
    assert dataset["category_levels"]
    assert dataset["scaler_parameters"]
