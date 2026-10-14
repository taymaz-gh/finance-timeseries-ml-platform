"""Testing synthetic training integration without fitting a neural network."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from finance_ml.config.training_config import TRAINING_CONFIG
from finance_ml.pipelines import training_runner


def synthetic_config() -> dict:
    """Prepare a local-only training configuration."""
    return {
        **TRAINING_CONFIG,
        "log_to_mlflow": False,
        "register_in_uc": False,
    }


def test_synthetic_dispatch_uses_prepared_gold(monkeypatch) -> None:
    """Pass synthetic Gold data directly into the existing model pipeline."""
    from finance_ml.data import synthetic_training

    spark = Mock()
    gold = Mock(name="gold_dataframe")
    pipeline = Mock(return_value={"model_uri": None})

    monkeypatch.setattr(
        synthetic_training,
        "prepare_synthetic_gold",
        Mock(return_value=gold),
    )

    result = training_runner.run_training(
        synthetic_config(),
        spark=spark,
        pipeline_fn=pipeline,
        data_source="synthetic",
        synthetic_accounts=12,
        synthetic_days=60,
        synthetic_seed=7,
    )

    assert result == {"model_uri": None}

    synthetic_training.prepare_synthetic_gold.assert_called_once_with(
        spark,
        n_accounts=12,
        n_days=60,
        random_seed=7,
    )

    pipeline.assert_called_once_with(
        config=synthetic_config(),
        spark=spark,
        gold_df=gold,
    )

    spark.stop.assert_not_called()


def test_synthetic_dispatch_rejects_mlflow() -> None:
    """Prevent synthetic demonstration runs from remote model logging."""
    with pytest.raises(ValueError, match="MLflow logging"):
        training_runner.run_training(
            TRAINING_CONFIG,
            spark=Mock(),
            pipeline_fn=Mock(),
            data_source="synthetic",
        )


def test_invalid_data_source_is_rejected() -> None:
    """Reject unknown data providers before training."""
    with pytest.raises(ValueError, match="data_source"):
        training_runner.run_training(
            synthetic_config(),
            spark=Mock(),
            pipeline_fn=Mock(),
            data_source="invalid",
        )


def test_synthetic_gold_rejects_excessive_rows() -> None:
    """Avoid allocating oversized synthetic demo datasets."""
    from finance_ml.data.synthetic_training import prepare_synthetic_gold

    with pytest.raises(ValueError, match="row limit"):
        prepare_synthetic_gold(
            Mock(),
            n_accounts=1000,
            n_days=60,
        )


def test_synthetic_gold_converts_numpy_scalars(monkeypatch) -> None:
    """Passing Spark-compatible Python scalars to createDataFrame."""
    import numpy as np
    import pandas as pd

    from finance_ml.data import synthetic_training
    from finance_ml.lakehouse import build_gold, build_silver

    generated = pd.DataFrame(
        [
            {
                "account_id": "ACC_00001",
                "date": pd.Timestamp("2026-01-01"),
                "account_type": np.str_("checking"),
                "balance": np.float64(100.0),
                "transaction_count": np.int64(5),
                "risk_state": np.int64(0),
            }
        ]
    )

    monkeypatch.setattr(
        synthetic_training,
        "generate_financial_timeseries",
        lambda **kwargs: generated,
    )

    monkeypatch.setattr(
        build_silver,
        "build_silver_dataframe",
        lambda df, **kwargs: df,
    )
    monkeypatch.setattr(
        build_gold,
        "build_gold_features",
        lambda df: df,
    )

    spark = Mock()
    spark.createDataFrame.return_value = Mock()

    synthetic_training.prepare_synthetic_gold(
        spark,
        n_accounts=1,
        n_days=1,
    )

    records = spark.createDataFrame.call_args.args[0]
    row = records[0]

    assert type(row["account_type"]) is str
    assert type(row["transaction_count"]) is int
    assert type(row["balance"]) is float
    assert type(row["date"]) is not pd.Timestamp
