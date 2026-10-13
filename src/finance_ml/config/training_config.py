"""Defining and validating training configuration without ML dependencies."""

from __future__ import annotations

from typing import Any, Mapping


TRAINING_CONFIG: dict[str, Any] = {
    "query_method": "spark",
    "source_table": "finance_ml.gold.account_features",
    "sequence_length": 7,
    "stride": 1,
    "scaling_method": "mad",
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "n_classes": 4,
    "boundary_width": 2,
    "lstm_units": (64,),
    "dropout_rate": 0.30,
    "learning_rate": 1e-3,
    "epochs": 50,
    "batch_size": 256,
    "patience": 5,
    "seed": 42,
    "verbose": 1,
    "reduce_lr_on_plateau": False,
    "lr_factor": 0.5,
    "lr_patience": 2,
    "min_lr": 1e-6,
    "run_mode": "development",
    "save_keras_checkpoint": True,
    "log_to_mlflow": True,
    "register_in_uc": False,
    "mlflow_experiment_name": None,
    "mlflow_model_name": "lstm_model",
    "registered_model_name": "finance_ml.models.lstm_risk_classifier",
    "artifact_dir": "artifacts/model_pipeline",
}


def validate_training_config(config: Mapping[str, Any]) -> None:
    """Reject unsupported combinations before querying or training."""
    if config["query_method"] not in {"spark", "connector", "sqlalchemy"}:
        raise ValueError("query_method must be spark, connector, or sqlalchemy")

    if config["run_mode"] not in {"development", "final"}:
        raise ValueError("run_mode must be development or final")

    if (
        not config["source_table"]
        or not all(
            part.replace("_", "").isalnum()
            for part in config["source_table"].split(".")
        )
        or len(config["source_table"].split(".")) != 3
    ):
        raise ValueError("source_table must be catalog.schema.table")

    if config["register_in_uc"] and not config["log_to_mlflow"]:
        raise ValueError("UC registration requires MLflow model logging")

    if config["register_in_uc"] and config["run_mode"] != "final":
        raise ValueError("UC registration is only supported for final runs")

    if config["run_mode"] == "final" and not config["log_to_mlflow"]:
        raise ValueError("Final runs must be logged to MLflow")

    if config["sequence_length"] <= 2 * config["boundary_width"]:
        raise ValueError("sequence_length must exceed twice boundary_width")
