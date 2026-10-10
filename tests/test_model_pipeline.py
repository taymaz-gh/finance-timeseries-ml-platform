"""Testing the training coordinator without starting expensive model training."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from finance_ml.pipelines import model_pipeline as pipeline


def _dataset():
    return {
        "X_train": np.zeros((3, 7, 2), dtype=np.float32),
        "y_train": np.zeros((3, 7), dtype=int),
        "X_validation": np.zeros((2, 7, 2), dtype=np.float32),
        "y_validation": np.zeros((2, 7), dtype=int),
        "X_test": np.zeros((2, 7, 2), dtype=np.float32),
        "y_test": np.zeros((2, 7), dtype=int),
        "train_metadata": [],
        "validation_metadata": [],
        "test_metadata": [],
        "final_feature_columns": ["balance", "cash_inflow"],
        "model_columns": {"numeric_feature_columns": ["balance", "cash_inflow"]},
        "medians": {"balance": 5.0},
        "category_levels": {"account_type": ["A"]},
        "scaler_parameters": {"balance": {"center": 5.0, "scale": 1.0}},
        "scaling_method": "mad",
        "train_end": "2026-01-01",
        "validation_end": "2026-02-01",
    }


def _result():
    return SimpleNamespace(
        validation_evaluation={"macro_f1": 0.8, "accuracy": 0.9},
        test_evaluation=None,
        model=SimpleNamespace(),
    )


@pytest.mark.parametrize("method", ["spark", "connector", "sqlalchemy"])
def test_acquisition_methods_normalize_to_spark(monkeypatch, method):
    """Selecting each query mechanism without duplicating preprocessing."""
    spark_df = object()
    fake_spark = SimpleNamespace(
        createDataFrame=lambda frame: spark_df,
    )
    monkeypatch.setattr(pipeline, "query_with_spark", lambda session, sql: spark_df)
    monkeypatch.setattr(
        pipeline, "query_with_sql_connector", lambda sql: pd.DataFrame({"x": [1]})
    )
    monkeypatch.setattr(
        pipeline, "query_with_sqlalchemy", lambda sql: pd.DataFrame({"x": [1]})
    )
    obtained = pipeline._acquire_gold_data(
        config={"query_method": method, "source_table": "finance_ml.gold.account_features"},
        spark=fake_spark,
    )
    assert obtained is spark_df


@pytest.mark.parametrize("bad_options", [
    {"query_method": "unknown"},
    {"run_mode": "bad"},
    {"register_in_uc": True, "log_to_mlflow": False},
    {"register_in_uc": True, "run_mode": "development"},
    {"run_mode": "final", "log_to_mlflow": False},
])
def test_configuration_rejects_invalid_combinations(bad_options):
    """Failing fast before remote queries or expensive training."""
    with pytest.raises(ValueError):
        pipeline._validate_config({**pipeline.CONFIG, **bad_options})


def test_development_run_skips_test_and_mlflow(monkeypatch, tmp_path):
    """Reusing dataset preparation and training without test-set evaluation."""
    calls = {}
    monkeypatch.setattr(pipeline, "_acquire_gold_data", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "prepare_model_dataset", lambda *args, **kwargs: _dataset())

    def fake_train(**kwargs):
        calls.update(kwargs)
        return _result()

    monkeypatch.setattr(pipeline, "run_lstm_experiment", fake_train)
    output = pipeline.run_model_pipeline(
        config={
            "run_mode": "development",
            "log_to_mlflow": False,
            "save_keras_checkpoint": False,
            "artifact_dir": str(tmp_path),
        },
        spark=object(),
    )

    assert calls["evaluate_test"] is False
    assert calls["n_features"] == 2
    assert output["model_uri"] is None
    assert output["registered_version"] is None
    metadata = json.loads(output["metadata_path"].read_text())
    assert metadata["final_feature_columns"] == ["balance", "cash_inflow"]
    assert metadata["medians"] == {"balance": 5.0}


def test_final_run_logs_model_and_registers_when_requested(monkeypatch, tmp_path):
    """Registering only the model URI returned from this final training run."""
    calls = {}
    monkeypatch.setattr(pipeline, "_acquire_gold_data", lambda **kwargs: object())
    monkeypatch.setattr(pipeline, "prepare_model_dataset", lambda *args, **kwargs: _dataset())

    def fake_train(**kwargs):
        calls["train"] = kwargs
        result = _result()
        result.test_evaluation = {"macro_f1": 0.79, "accuracy": 0.83}
        return result

    monkeypatch.setattr(pipeline, "run_lstm_experiment", fake_train)
    monkeypatch.setattr(
        pipeline, "log_sequence_experiment",
        lambda **kwargs: calls.setdefault("log", kwargs) and "models:/m-new",
    )
    monkeypatch.setattr(
        pipeline, "register_logged_model",
        lambda **kwargs: calls.setdefault("register", kwargs) and SimpleNamespace(version="2"),
    )

    class FakeRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    import mlflow
    monkeypatch.setattr(mlflow, "active_run", lambda: None)
    monkeypatch.setattr(mlflow, "start_run", lambda **kwargs: FakeRun())
    monkeypatch.setattr(mlflow, "set_tag", lambda *args: None)
    monkeypatch.setattr(mlflow, "log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *args: None)

    output = pipeline.run_model_pipeline(
        config={
            "run_mode": "final",
            "log_to_mlflow": True,
            "register_in_uc": True,
            "artifact_dir": str(tmp_path),
            "save_keras_checkpoint": False,
        },
        spark=object(),
    )

    assert calls["train"]["evaluate_test"] is True
    assert calls["log"]["log_model"] is True
    assert calls["log"]["model_input_example"].shape == (1, 7, 2)
    assert calls["register"]["model_uri"] == "models:/m-new"
    assert output["registered_version"].version == "2"
