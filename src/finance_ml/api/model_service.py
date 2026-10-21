"""Managing the verified Keras model used by the API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from finance_ml.models.local_bundle import (
    LocalModelBundle,
    load_local_bundle,
)


@dataclass(frozen=True)
class ModelService:
    """Holding a verified model and its saved preprocessing contract."""

    bundle: LocalModelBundle

    @classmethod
    def from_directory(cls, directory: str | Path) -> ModelService:
        """Loading the model bundle without retraining or refitting."""
        return cls(bundle=load_local_bundle(directory))

    def describe(self) -> dict[str, str | int]:
        """Reporting non-sensitive information about the loaded model."""
        metadata = self.bundle.metadata

        return {
            "model_source": "local",
            "sequence_length": metadata["sequence_length"],
            "n_features": len(metadata["final_feature_columns"]),
            "n_classes": metadata["n_classes"],
            "model_sha256": self.bundle.manifest["model"]["sha256"],
        }
