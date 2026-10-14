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

    # Bypassing corruption for this isolated scalar-conversion test.
    monkeypatch.setattr(
        synthetic_training,
        "inject_data_quality_issues",
        lambda df, **kwargs: df,
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

    call = spark.createDataFrame.call_args
    records = call.args[0]
    columns = call.kwargs["schema"]

    # Reconstructing column-name access from positional Spark records.
    row = dict(zip(columns, records[0]))

    assert type(row["account_type"]) is str
    assert type(row["transaction_count"]) is int
    assert type(row["balance"]) is float
    assert type(row["date"]) is not pd.Timestamp


def test_synthetic_training_injects_bronze_quality_issues(
    monkeypatch,
) -> None:
    """Verifying raw corruption and Spark-null conversion before cleaning."""
    from finance_ml.data import synthetic_training
    from finance_ml.data.generate_synthetic import generate_financial_timeseries
    from finance_ml.lakehouse import build_gold, build_silver

    # Intercepting the Bronze frame without launching Spark.
    spark = Mock()
    bronze_marker = Mock(name="bronze")
    spark.createDataFrame.return_value = bronze_marker

    monkeypatch.setattr(
        build_silver,
        "build_silver_dataframe",
        lambda frame, **kwargs: frame,
    )
    monkeypatch.setattr(
        build_gold,
        "build_gold_features",
        lambda frame: frame,
    )

    synthetic_training.prepare_synthetic_gold(
        spark,
        n_accounts=8,
        n_days=60,
        random_seed=42,
    )

    call = spark.createDataFrame.call_args
    records = call.args[0]
    columns = call.kwargs["schema"]

    clean = generate_financial_timeseries(
        n_accounts=8,
        n_days=60,
        random_seed=42,
    )

    # Preserving the original clean-data field order.
    assert columns == list(clean.columns)

    # Confirming deliberate Bronze duplication.
    assert len(records) == 482

    rows = [dict(zip(columns, values)) for values in records]

    # Confirming that pandas missing values became Spark nulls.
    assert any(row["cash_inflow"] is None for row in rows)
    assert any(row["account_type"] is None for row in rows)

    # Confirming that invalid Bronze observations exist.
    assert any(
        row["credit_utilization"] is not None and row["credit_utilization"] > 1.0
        for row in rows
    )

    # Confirming label preservation.
    assert all(row["risk_state"] is not None for row in rows)
    assert all(row["risk_state_name"] is not None for row in rows)
