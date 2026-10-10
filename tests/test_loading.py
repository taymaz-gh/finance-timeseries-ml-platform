"""Checking model-version metadata provenance and integrity."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from finance_ml.models.loading import load_model_bundle


def test_load_bundle_checks_hash_and_version(tmp_path):
    metadata = {
        "final_feature_columns": ["balance", "balance_missing"],
        "medians": {"balance": 1.0}, "category_levels": {},
        "scaler_parameters": {"balance": {"center": 1.0, "scale": 2.0}},
        "sequence_length": 7, "stride": 1, "boundary_width": 2,
    }
    file = tmp_path / "preprocessing_metadata.json"
    file.write_text(json.dumps(metadata), encoding="utf-8")
    checksum = hashlib.sha256(file.read_bytes()).hexdigest()
    tags = {
        "preprocessing_metadata_run_id": "run1",
        "preprocessing_metadata_path": "reconstructed_preprocessing/preprocessing_metadata.json",
        "preprocessing_metadata_sha256": checksum,
    }
    registry = SimpleNamespace(get_model_version=lambda **kw: SimpleNamespace(tags=tags, run_id="run1"))
    tracking = SimpleNamespace(download_artifacts=lambda *args: str(file))
    model = SimpleNamespace(input_shape=(None, 7, 2))
    bundle = load_model_bundle("catalog.schema.model", 1, registry_client=registry, tracking_client=tracking, model_loader=lambda uri: model)
    assert bundle.metadata_sha256 == checksum
    assert bundle.model is model
    assert bundle.model_version == "1"
    tags["preprocessing_metadata_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_model_bundle("catalog.schema.model", 1, registry_client=registry, tracking_client=tracking, model_loader=lambda uri: model)


def test_reject_cross_run_metadata(tmp_path):
    tags = {"preprocessing_metadata_run_id": "wrong", "preprocessing_metadata_path": "x.json", "preprocessing_metadata_sha256": "a" * 64}
    registry = SimpleNamespace(get_model_version=lambda **kw: SimpleNamespace(tags=tags, run_id="original"))
    with pytest.raises(ValueError, match="source run"):
        load_model_bundle("c.s.m", 1, registry_client=registry)
