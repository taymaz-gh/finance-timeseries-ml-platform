"""Testing the local Keras artifact contract without loading TensorFlow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from finance_ml.models.local_bundle import (
    MANIFEST_FILENAME,
    METADATA_FILENAME,
    MODEL_FILENAME,
    inspect_local_bundle,
    load_local_bundle,
    publish_local_bundle,
    validate_preprocessing_metadata,
)


def _metadata() -> dict:
    return {
        "training_input": "supplied_gold_dataframe",
        "sequence_length": 7,
        "stride": 1,
        "boundary_width": 2,
        "n_classes": 4,
        "final_feature_columns": [
            "balance", "balance_missing", "account_type_checking",
        ],
        "model_columns": {
            "numeric_feature_columns": ["balance"],
            "categorical_feature_columns": ["account_type"],
        },
        "medians": {"balance": 5.0},
        "category_levels": {"account_type": ["checking"]},
        "scaler_parameters": {
            "balance": {"center": 5.0, "scale": 3.0, "method": "mad"},
        },
        "scaling_method": "mad",
        "train_end": "2026-02-11",
        "validation_end": "2026-02-20",
    }


def _fake_model(*, n_features: int = 3, n_classes: int = 4):
    return SimpleNamespace(
        input_shape=(None, 7, n_features),
        output_shape=(None, 7, n_classes),
    )


def _publish(tmp_path: Path) -> Path:
    (tmp_path / MODEL_FILENAME).write_bytes(b"dummy keras bytes for a mocked loader")
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(_metadata()), encoding="utf-8")
    publish_local_bundle(tmp_path, model=_fake_model())
    return tmp_path


def _refresh_metadata_hash(tmp_path: Path) -> None:
    path = tmp_path / METADATA_FILENAME
    manifest_path = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["preprocessing"]["bytes"] = path.stat().st_size
    manifest["preprocessing"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_publishes_and_loads_bundle_without_tensorflow(tmp_path) -> None:
    """Publishing both files and loading them only through a verified manifest."""
    _publish(tmp_path)
    loader = Mock(return_value=_fake_model())
    bundle = load_local_bundle(tmp_path, model_loader=loader)

    assert bundle.directory == tmp_path.resolve()
    assert bundle.metadata["sequence_length"] == 7
    assert bundle.manifest["format_version"] == 1
    assert bundle.manifest["signature"]["n_features"] == 3
    loader.assert_called_once_with(tmp_path / MODEL_FILENAME)


def test_inspection_does_not_invoke_model_loader(tmp_path) -> None:
    """Verifying bundle integrity without importing TensorFlow or Keras."""
    _publish(tmp_path)
    metadata, manifest = inspect_local_bundle(tmp_path)
    assert metadata["n_classes"] == 4
    assert manifest["selection_metric"] == "val_macro_f1"


def test_missing_manifest_rejects_incomplete_training(tmp_path) -> None:
    """Refusing partial runs even if their model and metadata files exist."""
    (tmp_path / MODEL_FILENAME).write_bytes(b"checkpoint")
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(_metadata()))
    with pytest.raises(FileNotFoundError, match="Incomplete local model bundle"):
        load_local_bundle(tmp_path, model_loader=Mock())


def test_missing_checkpoint_rejected_before_manifest(tmp_path) -> None:
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(_metadata()))
    with pytest.raises(FileNotFoundError, match="model.keras"):
        publish_local_bundle(tmp_path, model=_fake_model())


def test_rejects_republishing_manifest(tmp_path) -> None:
    _publish(tmp_path)
    with pytest.raises(FileExistsError, match="already published"):
        publish_local_bundle(tmp_path, model=_fake_model())


@pytest.mark.parametrize("target", [MODEL_FILENAME, METADATA_FILENAME])
def test_detects_file_tampering_before_load(tmp_path, target) -> None:
    _publish(tmp_path)
    (tmp_path / target).write_bytes(b"modified file")
    loader = Mock(return_value=_fake_model())
    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        load_local_bundle(tmp_path, model_loader=loader)
    loader.assert_not_called()


def test_rejects_unsupported_manifest_version(tmp_path) -> None:
    _publish(tmp_path)
    manifest_file = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(manifest_file.read_text())
    manifest["format_version"] = 999
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="format_version"):
        inspect_local_bundle(tmp_path)


def test_rejects_manifest_path_substitution(tmp_path) -> None:
    _publish(tmp_path)
    path = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    manifest["model"]["file"] = "../another.keras"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fixed name"):
        inspect_local_bundle(tmp_path)


def test_detects_manifest_signature_mismatch(tmp_path) -> None:
    _publish(tmp_path)
    path = tmp_path / MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    manifest["signature"]["n_features"] = 99
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="signature disagrees"):
        inspect_local_bundle(tmp_path)


def test_model_input_mismatch_detected_when_loading(tmp_path) -> None:
    _publish(tmp_path)
    with pytest.raises(ValueError, match="Model input shape"):
        load_local_bundle(tmp_path, model_loader=lambda path: _fake_model(n_features=4))


def test_model_output_mismatch_detected_when_loading(tmp_path) -> None:
    _publish(tmp_path)
    with pytest.raises(ValueError, match="Model output shape"):
        load_local_bundle(tmp_path, model_loader=lambda path: _fake_model(n_classes=3))


def test_rejects_incorrect_feature_order_even_with_matching_checksums(tmp_path) -> None:
    _publish(tmp_path)
    path = tmp_path / METADATA_FILENAME
    metadata = json.loads(path.read_text())
    metadata["final_feature_columns"].reverse()
    path.write_text(json.dumps(metadata))
    _refresh_metadata_hash(tmp_path)
    with pytest.raises(ValueError, match="final_feature_columns"):
        inspect_local_bundle(tmp_path)


def test_rejects_invalid_scaling_metadata(tmp_path) -> None:
    metadata = _metadata()
    metadata["scaler_parameters"]["balance"]["scale"] = 0
    with pytest.raises(ValueError, match="scale must be positive"):
        validate_preprocessing_metadata(metadata)


def test_rejects_unsafe_boundary_trim(tmp_path) -> None:
    metadata = _metadata()
    metadata["boundary_width"] = 4
    with pytest.raises(ValueError, match="boundary_width"):
        validate_preprocessing_metadata(metadata)


def test_rejects_invalid_metadata_even_before_checkpoint_published(tmp_path) -> None:
    (tmp_path / MODEL_FILENAME).write_bytes(b"checkpoint")
    metadata = _metadata()
    metadata.pop("category_levels")
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="missing fields"):
        publish_local_bundle(tmp_path, model=_fake_model())
    assert not (tmp_path / MANIFEST_FILENAME).exists()


def test_rejects_wrong_model_at_publish_time(tmp_path) -> None:
    (tmp_path / MODEL_FILENAME).write_bytes(b"checkpoint")
    (tmp_path / METADATA_FILENAME).write_text(json.dumps(_metadata()))
    with pytest.raises(ValueError, match="Model input shape"):
        publish_local_bundle(tmp_path, model=_fake_model(n_features=2))
    assert not (tmp_path / MANIFEST_FILENAME).exists()
