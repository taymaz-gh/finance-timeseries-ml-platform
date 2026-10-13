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
