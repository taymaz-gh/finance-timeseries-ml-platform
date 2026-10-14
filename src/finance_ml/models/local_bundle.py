"""Publish and load versioned local Keras model bundles for inference.

Keeping model files and fitted preprocessing metadata together. Refusing
incomplete, internally inconsistent, or modified artifact bundles before
attempting to deserialize a Keras model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4


BUNDLE_FORMAT_VERSION = 1
MODEL_FILENAME = "best_model.keras"
METADATA_FILENAME = "preprocessing_metadata.json"
MANIFEST_FILENAME = "bundle_manifest.json"


@dataclass(frozen=True)
class LocalModelBundle:
    """Hold a loaded Keras model and its verified preprocessing contract."""

    directory: Path
    model: Any
    metadata: dict[str, Any]
    manifest: dict[str, Any]


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _strings(value: Any, name: str, *, nonempty: bool = False) -> list[str]:
    if (
        not isinstance(value, list)
        or (nonempty and not value)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name} must be a list of distinct nonempty strings")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def validate_preprocessing_metadata(metadata: Any) -> dict[str, Any]:
    """Validate the fitted feature order and numerical preprocessing contract.

    Supporting the existing preprocessing_metadata.json schema, without
    requiring labels or data access. This validation does not fit or modify
    any preprocessing statistics.
    """
    metadata = _mapping(metadata, "preprocessing metadata")
    required = {
        "sequence_length", "stride", "boundary_width", "n_classes",
        "final_feature_columns", "model_columns", "medians",
        "category_levels", "scaler_parameters", "scaling_method",
        "train_end", "validation_end",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Preprocessing metadata missing fields: {', '.join(missing)}")

    seq_len = _positive_int(metadata["sequence_length"], "sequence_length")
    _positive_int(metadata["stride"], "stride")
    boundary = _nonnegative_int(metadata["boundary_width"], "boundary_width")
    if boundary * 2 >= seq_len:
        raise ValueError("boundary_width must preserve interior sequence positions")
    if _positive_int(metadata["n_classes"], "n_classes") < 2:
        raise ValueError("n_classes must be at least 2")

    feature_columns = _strings(
        metadata["final_feature_columns"], "final_feature_columns", nonempty=True
    )
    if any(name in {"risk_state", "risk_state_name", "account_id", "date"}
           for name in feature_columns):
        raise ValueError("Model feature columns include an identifier or target")

    columns = _mapping(metadata["model_columns"], "model_columns")
    numeric = _strings(
        columns.get("numeric_feature_columns"), "numeric_feature_columns", nonempty=True
    )
    categorical = _strings(
        columns.get("categorical_feature_columns"), "categorical_feature_columns"
    )
    if set(numeric) & set(categorical):
        raise ValueError("Numeric and categorical feature columns overlap")

    medians = _mapping(metadata["medians"], "medians")
    levels = _mapping(metadata["category_levels"], "category_levels")
    scalers = _mapping(metadata["scaler_parameters"], "scaler_parameters")
    scaling_method = metadata["scaling_method"]
    if scaling_method not in {"mad", "standard"}:
        raise ValueError("Unsupported scaling_method in preprocessing metadata")

    if not set(medians) <= set(numeric):
        raise ValueError("Imputation medians refer to unknown numeric features")
    if set(levels) != set(categorical):
        raise ValueError("Categorical levels do not match the fitted feature schema")
    if set(scalers) != set(numeric):
        raise ValueError("Scaler keys do not match the numeric feature schema")

    for name, median in medians.items():
        _finite_number(median, f"medians.{name}")
    for name, values in levels.items():
        _strings(values, f"category_levels.{name}", nonempty=True)
    for name, params in scalers.items():
        params = _mapping(params, f"scaler_parameters.{name}")
        _finite_number(params.get("center"), f"scaler_parameters.{name}.center")
        _finite_number(params.get("scale"), f"scaler_parameters.{name}.scale", positive=True)
        if params.get("method") != scaling_method:
            raise ValueError(f"Scaler method mismatch for {name}")

    # Reusing the same ordering logic used to build training tensors.
    from finance_ml.preprocessing.feature_schema import build_final_feature_columns

    expected_columns = build_final_feature_columns(numeric, medians, levels)
    if feature_columns != expected_columns:
        raise ValueError("Saved final_feature_columns do not match the fitted feature schema")
    for name in ("train_end", "validation_end"):
        if not isinstance(metadata[name], str) or not metadata[name]:
            raise ValueError(f"{name} must be a nonempty serialized timestamp")
    return dict(metadata)


def _shape_tuple(shape: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(shape, (tuple, list)) or len(shape) != 3:
        raise ValueError(f"{name} must describe one rank-3 seq2seq tensor")
    return tuple(shape)


def validate_model_signature(model: Any, metadata: Mapping[str, Any]) -> None:
    """Require Keras inputs and outputs to match saved preprocessing."""
    inputs = _shape_tuple(getattr(model, "input_shape", None), "model.input_shape")
    outputs = _shape_tuple(getattr(model, "output_shape", None), "model.output_shape")
    expected_in = (metadata["sequence_length"], len(metadata["final_feature_columns"]))
    expected_out = (metadata["sequence_length"], metadata["n_classes"])
    if inputs[1:] != expected_in:
        raise ValueError(f"Model input shape {inputs[1:]} != expected {expected_in}")
    if outputs[1:] != expected_out:
        raise ValueError(f"Model output shape {outputs[1:]} != expected {expected_out}")


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON artifact: {path.name}") from exc


def publish_local_bundle(directory: str | Path, *, model: Any) -> Path:
    """Publish a manifest last, only after model and metadata are complete.

    The caller must use a fresh per-run directory. No model is loaded here;
    actual Keras checkpointing remains owned by ModelCheckpoint.
    """
    directory = Path(directory)
    model_file = directory / MODEL_FILENAME
    metadata_file = directory / METADATA_FILENAME
    manifest_file = directory / MANIFEST_FILENAME
    if manifest_file.exists():
        raise FileExistsError(f"Bundle already published: {manifest_file}")
    for path in (model_file, metadata_file):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required nonempty bundle artifact is missing: {path}")

    metadata = validate_preprocessing_metadata(_read_json(metadata_file))
    validate_model_signature(model, metadata)
    manifest: dict[str, Any] = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "val_macro_f1",
        "model_format": "keras_v3",
        "model": {
            "file": MODEL_FILENAME,
            "sha256": _digest(model_file),
            "bytes": model_file.stat().st_size,
        },
        "preprocessing": {
            "file": METADATA_FILENAME,
            "sha256": _digest(metadata_file),
            "bytes": metadata_file.stat().st_size,
        },
        "signature": {
            "sequence_length": metadata["sequence_length"],
            "n_features": len(metadata["final_feature_columns"]),
            "n_classes": metadata["n_classes"],
            "stride": metadata["stride"],
            "boundary_width": metadata["boundary_width"],
        },
    }
    # Atomically making the bundle discoverable only after both files exist.
    temporary = directory / f".{MANIFEST_FILENAME}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_file)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_file


def inspect_local_bundle(directory: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a bundle and return (metadata, manifest) without importing Keras."""
    directory = Path(directory)
    manifest_file = directory / MANIFEST_FILENAME
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Incomplete local model bundle: {manifest_file} is missing")
    manifest = _mapping(_read_json(manifest_file), "bundle manifest")
    if type(manifest.get("format_version")) is not int or manifest["format_version"] != BUNDLE_FORMAT_VERSION:
        raise ValueError("Unsupported local model bundle format_version")
    if manifest.get("model_format") != "keras_v3" or manifest.get("selection_metric") != "val_macro_f1":
        raise ValueError("Unsupported bundle model format or selection metric")

    for key, filename in (("model", MODEL_FILENAME), ("preprocessing", METADATA_FILENAME)):
        entry = _mapping(manifest.get(key), f"manifest.{key}")
        if entry.get("file") != filename:
            raise ValueError(f"Bundle {key} filename is not the expected fixed name")
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Bundle artifact is missing: {path}")
        expected_size = _positive_int(entry.get("bytes"), f"manifest.{key}.bytes")
        if path.stat().st_size != expected_size:
            raise ValueError(f"Bundle {key} file size mismatch")
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"Bundle {key} SHA-256 digest is invalid")
        if _digest(path) != expected_hash:
            raise ValueError(f"Bundle {key} SHA-256 mismatch")

    metadata = validate_preprocessing_metadata(_read_json(directory / METADATA_FILENAME))
    signature = _mapping(manifest.get("signature"), "manifest.signature")
    expected = {
        "sequence_length": metadata["sequence_length"],
        "n_features": len(metadata["final_feature_columns"]),
        "n_classes": metadata["n_classes"],
        "stride": metadata["stride"],
        "boundary_width": metadata["boundary_width"],
    }
    if signature != expected:
        raise ValueError("Bundle manifest signature disagrees with preprocessing metadata")
    return metadata, dict(manifest)


def load_local_bundle(
    directory: str | Path,
    *,
    model_loader: Callable[[Path], Any] | None = None,
) -> LocalModelBundle:
    """Load a verified local Keras bundle without network or registry calls.

    The injectable loader enables fast tests without importing TensorFlow.
    Only load models from a source you trust: a checksum manifest is not
    a cryptographic authenticity signature.
    """
    directory = Path(directory)
    metadata, manifest = inspect_local_bundle(directory)
    if model_loader is None:
        import tensorflow as tf

        def model_loader(path: Path) -> Any:
            return tf.keras.models.load_model(path, compile=False, safe_mode=True)

    model = model_loader(directory / MODEL_FILENAME)
    validate_model_signature(model, metadata)
    return LocalModelBundle(
        directory=directory.resolve(), model=model, metadata=metadata, manifest=manifest
    )
