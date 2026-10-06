"""MLflow tracking utilities for sequence-model experiments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from finance_ml.models.experiment import (
    SequenceExperimentResult,
)


def log_sequence_experiment(
    *,
    result: SequenceExperimentResult,
    parameters: Mapping[str, Any],
) -> None:
    """
    Log a completed sequence-model experiment to an active MLflow run.

    This function records experiment parameters, validation metrics,
    optional test metrics, and saved training artifacts. The MLflow
    run lifecycle is intentionally managed by the caller.

    Args:
        result:
            Completed sequence-model experiment result.

        parameters:
            Model and training parameters to log.

    Raises:
        RuntimeError:
            If no MLflow run is active.

        FileNotFoundError:
            If a reported artifact path does not exist.
    """
    # Importing MLflow lazily keeps the core project testable in
    # environments where MLflow is not installed.
    import mlflow

    if mlflow.active_run() is None:
        raise RuntimeError(
            "An active MLflow run is required before logging "
            "a sequence-model experiment."
        )

    mlflow.log_params(dict(parameters))

    validation_metrics = {
        "validation_accuracy": float(
            result.validation_evaluation["accuracy"]
        ),
        "validation_macro_f1": float(
            result.validation_evaluation["macro_f1"]
        ),
    }

    history = result.history.history

    validation_macro_f1_history = np.asarray(
        history["val_macro_f1"],
        dtype=float,
    )

    best_epoch = int(
        np.argmax(validation_macro_f1_history) + 1
    )

    validation_metrics["best_val_macro_f1"] = float(
        validation_macro_f1_history[best_epoch - 1]
    )

    mlflow.log_metrics(validation_metrics)

    mlflow.set_tag(
        "best_epoch",
        str(best_epoch),
    )

    if result.test_evaluation is not None:
        test_metrics = {
            "test_accuracy": float(
                result.test_evaluation["accuracy"]
            ),
            "test_macro_f1": float(
                result.test_evaluation["macro_f1"]
            ),
        }

        mlflow.log_metrics(test_metrics)

    if result.artifact_paths is not None:
        for artifact_path in result.artifact_paths.values():
            artifact_path = Path(artifact_path)

            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"MLflow artifact does not exist: "
                    f"{artifact_path}"
                )

            mlflow.log_artifact(
                str(artifact_path),
                artifact_path="training_artifacts",
            )