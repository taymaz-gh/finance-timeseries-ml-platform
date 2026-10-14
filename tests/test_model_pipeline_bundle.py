"""Testing local artifact publication without Spark or LSTM fitting."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from finance_ml.models.local_bundle import inspect_local_bundle
from finance_ml.pipelines import model_pipeline as pipeline


def _dataset() -> dict:
    return {
        "X_train": np.zeros((2, 7, 3), dtype=np.float32),
        "y_train": np.zeros((2, 7), dtype=int),
        "X_validation": np.zeros((1, 7, 3), dtype=np.float32),
        "y_validation": np.zeros((1, 7), dtype=int),
        "X_test": np.zeros((1, 7, 3), dtype=np.float32),
        "y_test": np.zeros((1, 7), dtype=int),
        "train_metadata": [], "validation_metadata": [], "test_metadata": [],
        "final_feature_columns": [
            "balance", "balance_missing", "account_type_checking"
        ],
        "model_columns": {
            "numeric_feature_columns": ["balance"],
            "categorical_feature_columns": ["account_type"],
        },
        "medians": {"balance": 5.0},
        "category_levels": {"account_type": ["checking"]},
        "scaler_parameters": {
            "balance": {"center": 5.0, "scale": 3.0, "method": "mad"}
        },
        "scaling_method": "mad",
        "train_end": "2026-02-11", "validation_end": "2026-02-20",
    }


def _mock_pipeline(monkeypatch, *, create_checkpoint: bool = True):
    monkeypatch.setattr(
        pipeline, "prepare_model_dataset", lambda *args, **kwargs: _dataset()
    )
    def fake_train(**kwargs):
        if create_checkpoint and kwargs["save_best_model"]:
            (kwargs["artifact_dir"] / "best_model.keras").write_bytes(b"keras test placeholder")
        return SimpleNamespace(
            model=SimpleNamespace(input_shape=(None, 7, 3), output_shape=(None, 7, 4)),
            validation_evaluation={"macro_f1": 0.5, "accuracy": 0.6},
            test_evaluation=None,
        )
    monkeypatch.setattr(pipeline, "run_lstm_experiment", fake_train)


def test_isolated_runs_publish_reusable_contracts(monkeypatch, tmp_path) -> None:
    """Keeping model and metadata together across repeated training runs."""
    _mock_pipeline(monkeypatch)
    config = {
        "log_to_mlflow": False,
        "register_in_uc": False,
        "save_keras_checkpoint": True,
        "artifact_dir": str(tmp_path),
    }
    gold = object()
    first = pipeline.run_model_pipeline(config, gold_df=gold)
    second = pipeline.run_model_pipeline(config, gold_df=gold)

    assert first["bundle_dir"] != second["bundle_dir"]
    assert first["bundle_dir"].parent == tmp_path
    assert second["bundle_dir"].parent == tmp_path
    assert first["metadata_path"].parent == first["bundle_dir"]
    assert first["bundle_manifest_path"].is_file()
    assert second["bundle_manifest_path"].is_file()
    assert first["model_uri"] is None
    assert first["registered_version"] is None
    assert first["feature_columns"] == _dataset()["final_feature_columns"]
    metadata, manifest = inspect_local_bundle(first["bundle_dir"])
    assert metadata["training_input"] == "supplied_gold_dataframe"
    assert manifest["signature"]["n_features"] == 3


def test_checkpoint_disabled_creates_no_inference_bundle(monkeypatch, tmp_path) -> None:
    """Distinguishing training metadata from a deployable saved model."""
    _mock_pipeline(monkeypatch, create_checkpoint=False)
    result = pipeline.run_model_pipeline(
        {
            "log_to_mlflow": False,
            "save_keras_checkpoint": False,
            "artifact_dir": str(tmp_path),
        },
        gold_df=object(),
    )
    assert result["bundle_dir"] is None
    assert result["bundle_manifest_path"] is None
    assert result["metadata_path"].is_file()


def test_missing_checkpoint_never_publishes_manifest(monkeypatch, tmp_path) -> None:
    """Keeping unsuccessful/incomplete runs unavailable to inference."""
    _mock_pipeline(monkeypatch, create_checkpoint=False)
    with pytest.raises(FileNotFoundError, match="model.keras"):
        pipeline.run_model_pipeline(
            {
                "log_to_mlflow": False,
                "save_keras_checkpoint": True,
                "artifact_dir": str(tmp_path),
            },
            gold_df=object(),
        )
    assert list(tmp_path.glob("run_*/bundle_manifest.json")) == []
