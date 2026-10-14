"""Testing lightweight training configuration and backward compatibility."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from finance_ml.config.training_config import (
    TRAINING_CONFIG,
    validate_training_config,
)


def test_default_training_config_is_valid() -> None:
    """Checking that the default configuration remains valid."""
    validate_training_config(TRAINING_CONFIG)

    assert TRAINING_CONFIG["sequence_length"] == 7
    assert TRAINING_CONFIG["boundary_width"] == 2
    assert TRAINING_CONFIG["epochs"] == 50
    assert TRAINING_CONFIG["run_mode"] == "development"
    assert TRAINING_CONFIG["register_in_uc"] is False


@pytest.mark.parametrize(
    ("updates", "expected_error"),
    [
        ({"query_method": "invalid"}, "query_method"),
        ({"run_mode": "invalid"}, "run_mode"),
        ({"source_table": "invalid"}, "source_table"),
        (
            {"register_in_uc": True, "log_to_mlflow": False},
            "UC registration",
        ),
        (
            {"register_in_uc": True, "run_mode": "development"},
            "final runs",
        ),
        (
            {"run_mode": "final", "log_to_mlflow": False},
            "Final runs",
        ),
        (
            {"sequence_length": 4, "boundary_width": 2},
            "sequence_length",
        ),
    ],
)
def test_invalid_training_config(
    updates: dict,
    expected_error: str,
) -> None:
    """Rejecting unsupported training configurations."""
    config = {**TRAINING_CONFIG, **updates}

    with pytest.raises(ValueError, match=expected_error):
        validate_training_config(config)


def test_backward_compatible_pipeline_config_imports() -> None:
    """Preserving configuration imports from the existing pipeline."""
    from finance_ml.pipelines.model_pipeline import (
        CONFIG,
        _validate_config,
    )

    assert CONFIG is TRAINING_CONFIG
    assert _validate_config is validate_training_config


def test_training_dry_run_does_not_import_tensorflow() -> None:
    """Checking dependency isolation in a fresh Python interpreter."""
    script = """
import json
import sys
from typer.testing import CliRunner
from finance_ml.cli import app

result = CliRunner().invoke(app, ["train", "--dry-run"])

print(json.dumps({
    "exit_code": result.exit_code,
    "tensorflow_imported": "tensorflow" in sys.modules,
    "pipeline_imported": "finance_ml.pipelines.model_pipeline" in sys.modules,
    "dry_run_completed": "Dry run complete." in result.output,
}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(result.stdout)

    assert report["exit_code"] == 0
    assert report["tensorflow_imported"] is False
    assert report["pipeline_imported"] is False
    assert report["dry_run_completed"] is True


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("epochs", 0),
        ("epochs", -5),
        ("epochs", 2.5),
        ("epochs", True),
        ("epochs", "20"),
        ("batch_size", 0),
        ("batch_size", -1),
        ("batch_size", 128.5),
        ("batch_size", False),
        ("batch_size", "128"),
    ],
)
def test_reject_invalid_training_integer(
    key: str,
    value: object,
) -> None:
    """Rejecting nonpositive and noninteger training parameters."""
    config = {**TRAINING_CONFIG, key: value}

    with pytest.raises(ValueError, match=f"{key} must be a positive integer"):
        validate_training_config(config)


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"learning_rate": -0.01}, "learning_rate"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"learning_rate": float("inf")}, "learning_rate"),
        ({"dropout_rate": 1.5}, "dropout_rate"),
        ({"dropout_rate": -0.1}, "dropout_rate"),
        ({"train_fraction": 1.5}, "train_fraction"),
        ({"validation_fraction": 0.0}, "validation_fraction"),
        (
            {"train_fraction": 0.8, "validation_fraction": 0.3},
            "train_fraction.*validation_fraction",
        ),
        ({"sequence_length": 0}, "sequence_length"),
        ({"stride": 0}, "stride"),
        ({"n_classes": 1}, "n_classes"),
        ({"patience": -1}, "patience"),
        ({"boundary_width": -1}, "boundary_width"),
        ({"lstm_units": []}, "lstm_units"),
        ({"lstm_units": [64, -32]}, "lstm_units"),
        ({"lstm_units": [True]}, "lstm_units"),
        ({"reduce_lr_on_plateau": "yes"}, "reduce_lr_on_plateau"),
        (
            {"reduce_lr_on_plateau": True, "lr_factor": 1.5},
            "lr_factor",
        ),
        (
            {"reduce_lr_on_plateau": True, "lr_patience": -1},
            "lr_patience",
        ),
        (
            {"reduce_lr_on_plateau": True, "min_lr": -0.1},
            "min_lr",
        ),
    ],
)
def test_reject_invalid_training_hyperparameters(
    updates: dict,
    error: str,
) -> None:
    """Rejecting invalid ML hyperparameters before execution."""
    config = {**TRAINING_CONFIG, **updates}

    with pytest.raises(ValueError, match=error):
        validate_training_config(config)


def test_yaml_style_lstm_units_are_supported() -> None:
    """Accepting list-valued LSTM units from YAML."""
    config = {**TRAINING_CONFIG, "lstm_units": [64, 32]}

    validate_training_config(config)


def test_valid_learning_rate_scheduler_configuration() -> None:
    """Accepting valid optional learning-rate scheduler settings."""
    config = {
        **TRAINING_CONFIG,
        "reduce_lr_on_plateau": True,
        "lr_factor": 0.5,
        "lr_patience": 2,
        "min_lr": 1e-6,
    }

    validate_training_config(config)


@pytest.mark.parametrize(
    "key",
    [
        "save_keras_checkpoint",
        "log_to_mlflow",
        "register_in_uc",
        "reduce_lr_on_plateau",
    ],
)
def test_reject_string_boolean_configuration(key: str) -> None:
    """Rejecting strings where actual boolean values are required."""
    config = {**TRAINING_CONFIG, key: "false"}

    with pytest.raises(ValueError, match=f"{key} must be a boolean"):
        validate_training_config(config)
