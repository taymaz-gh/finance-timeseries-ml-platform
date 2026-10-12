"""Loading a UC model together with its version-pinned preprocessing contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelBundle:
    """Holding a loaded model and verified, version-pinned preprocessing metadata."""

    model: Any
    metadata: dict[str, Any]
    model_name: str
    model_version: str
    metadata_sha256: str


def load_model_bundle(
    model_name: str,
    version: str | int,
    *,
    registry_client: Any = None,
    tracking_client: Any = None,
    model_loader: Any = None,
) -> ModelBundle:
    """Load UC model and metadata; fail closed on provenance or integrity mismatch.

    Clients/loaders can be injected for offline unit tests. Authentication and
    MLflow tracking/registry URIs must be configured by the caller.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    if registry_client is None:
        registry_client = MlflowClient(registry_uri="databricks-uc")
    if tracking_client is None:
        tracking_client = MlflowClient()
    if model_loader is None:
        model_loader = mlflow.keras.load_model

    name, version = str(model_name), str(version)
    info = registry_client.get_model_version(name=name, version=version)
    tags = info.tags or {}
    required = (
        "preprocessing_metadata_run_id",
        "preprocessing_metadata_path",
        "preprocessing_metadata_sha256",
    )
    missing = [key for key in required if not tags.get(key)]
    if missing:
        raise ValueError(f"Model version is missing preprocessing tags: {missing}")
    run_id = tags["preprocessing_metadata_run_id"]
    if info.run_id and run_id != info.run_id:
        raise ValueError("Metadata run ID does not match registered model source run")
    relative_path = tags["preprocessing_metadata_path"]
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise ValueError("Unsafe preprocessing artifact path")
    expected_hash = tags["preprocessing_metadata_sha256"].lower()
    if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
        raise ValueError("Invalid SHA-256 tag")

    downloaded = Path(tracking_client.download_artifacts(run_id, relative_path))
    if not downloaded.is_file():
        raise FileNotFoundError(f"Metadata artifact not found: {relative_path}")
    payload = downloaded.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("Preprocessing metadata SHA-256 mismatch")
    metadata = json.loads(payload.decode("utf-8"))
    for key in ("final_feature_columns", "medians", "category_levels", "scaler_parameters", "sequence_length", "boundary_width", "stride"):
        if key not in metadata:
            raise ValueError(f"Incomplete preprocessing metadata: {key}")
    if not metadata["final_feature_columns"]:
        raise ValueError("Empty training feature schema")
    if int(metadata["sequence_length"]) <= 2 * int(metadata["boundary_width"]):
        raise ValueError("Invalid sequence and boundary settings")

    model = model_loader(f"models:/{name}/{version}")
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, tuple) and len(input_shape) == 3:
        if input_shape[1] is not None and input_shape[1] != int(metadata["sequence_length"]):
            raise ValueError("Model input timesteps differ from metadata")
        if input_shape[2] is not None and input_shape[2] != len(metadata["final_feature_columns"]):
            raise ValueError("Model input features differ from metadata")
    return ModelBundle(model, metadata, name, version, expected_hash)
