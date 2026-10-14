""" Defining and validating default training configuration without ML dependencies.

The defaults provide a single source of truth for training parameters.
Run-specific overrides may be supplied through CLI arguments or external
configuration files (YAML or JSON), without modifying these defaults.
"""

from __future__ import annotations

import math

from numbers import Integral, Real

from typing import Any, Mapping


TRAINING_CONFIG: dict[str, Any] = {
    "query_method": "spark",
    "source_table": "finance_ml.gold.account_features",
    "sequence_length": 7,
    "stride": 1,
    "scaling_method": "mad",
    "feature_schema_profile": "observed",
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
    """Validate training parameters and reject unsupported configurations."""

    # Validating the data-query method.
    if config["query_method"] not in {"spark", "connector", "sqlalchemy"}:
        raise ValueError("query_method must be spark, connector, or sqlalchemy")

    # Validating the execution mode.
    if config["run_mode"] not in {"development", "final"}:
        raise ValueError("run_mode must be development or final")

    # Validating the model input-schema policy.
    if config["feature_schema_profile"] not in {
        "observed",
        "reference_v1",
    }:
        raise ValueError("feature_schema_profile must be observed or reference_v1")

    # Validating the fully qualified Unity Catalog source table.
    source_table = config["source_table"]

    if (
        not isinstance(source_table, str)
        or len(source_table.split(".")) != 3
        or not all(part.replace("_", "").isalnum() for part in source_table.split("."))
    ):
        raise ValueError("source_table must be catalog.schema.table")

    boolean_keys = (
        "save_keras_checkpoint",
        "log_to_mlflow",
        "register_in_uc",
        "reduce_lr_on_plateau",
    )

    for key in boolean_keys:
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")

    # Validating MLflow logging and model-registration requirements.
    if config["register_in_uc"] and not config["log_to_mlflow"]:
        raise ValueError("UC registration requires MLflow model logging")

    if config["register_in_uc"] and config["run_mode"] != "final":
        raise ValueError("UC registration is only supported for final runs")

    if config["run_mode"] == "final" and not config["log_to_mlflow"]:
        raise ValueError("Final runs must be logged to MLflow")

    # Validating positive integer training and model parameters.
    positive_integer_keys = (
        "sequence_length",
        "stride",
        "epochs",
        "batch_size",
        "n_classes",
    )

    for key in positive_integer_keys:
        value = config[key]

        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")

    if config["n_classes"] < 2:
        raise ValueError("n_classes must be at least 2")

    # Validating non-negative integer parameters.
    nonnegative_integer_keys = (
        "boundary_width",
        "patience",
        "seed",
        "verbose",
    )

    for key in nonnegative_integer_keys:
        value = config[key]

        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")

    # Validating the relationship between sequence length and boundaries.
    if config["sequence_length"] <= 2 * config["boundary_width"]:
        raise ValueError("sequence_length must exceed twice boundary_width")

    # Validating finite numerical hyperparameters.
    def require_finite_number(key: str) -> float:
        """Return a validated finite numerical configuration value."""
        value = config[key]

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{key} must be a finite number")

        return float(value)

    # Validating the learning rate.
    learning_rate = require_finite_number("learning_rate")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")

    # Validating the dropout probability.
    dropout_rate = require_finite_number("dropout_rate")

    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be in [0, 1)")

    # Validating the chronological train/validation/test split.
    train_fraction = require_finite_number("train_fraction")
    validation_fraction = require_finite_number("validation_fraction")

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")

    # Validating LSTM architecture.
    # Accepting tuples from Python and lists from YAML.
    lstm_units = config["lstm_units"]

    if (
        not isinstance(lstm_units, (list, tuple))
        or not lstm_units
        or any(
            isinstance(units, bool) or not isinstance(units, Integral) or units <= 0
            for units in lstm_units
        )
    ):
        raise ValueError(
            "lstm_units must be a nonempty list or tuple " "of positive integers"
        )

    if config["reduce_lr_on_plateau"]:
        lr_factor = require_finite_number("lr_factor")
        min_lr = require_finite_number("min_lr")
        lr_patience = config["lr_patience"]

        if not 0 < lr_factor < 1:
            raise ValueError("lr_factor must be between 0 and 1")

        if (
            isinstance(lr_patience, bool)
            or not isinstance(lr_patience, Integral)
            or lr_patience < 0
        ):
            raise ValueError("lr_patience must be a non-negative integer")

        if min_lr < 0:
            raise ValueError("min_lr must be non-negative")
