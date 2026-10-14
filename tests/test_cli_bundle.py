"""Testing the CLI's local-model output without starting training."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from typer.testing import CliRunner

from finance_ml.cli import app


def test_training_cli_reports_local_model_bundle(monkeypatch, tmp_path) -> None:
    """Exposing the exact saved bundle directory for later inference commands."""
    from finance_ml.pipelines import training_runner

    run_training = Mock(return_value={
        "model_uri": None,
        "bundle_dir": tmp_path / "run_example",
    })
    monkeypatch.setattr(training_runner, "run_training", run_training)
    config = tmp_path / "config.yaml"
    config.write_text("log_to_mlflow: false\nregister_in_uc: false\n")

    result = CliRunner().invoke(app, [
        "train", "--data-source", "synthetic", "--config", str(config),
    ])
    assert result.exit_code == 0
    assert f"Local model bundle: {tmp_path / 'run_example'}" in result.output
    run_training.assert_called_once()


def test_dry_run_does_not_create_bundle(monkeypatch, tmp_path) -> None:
    """Preserving side-effect-free inspection even with checkpoints enabled."""
    from finance_ml.pipelines import training_runner

    run_training = Mock()
    monkeypatch.setattr(training_runner, "run_training", run_training)
    config = tmp_path / "config.yaml"
    config.write_text(
        "log_to_mlflow: false\nregister_in_uc: false\n"
        "save_keras_checkpoint: true\n"
    )
    result = CliRunner().invoke(app, [
        "train", "--data-source", "synthetic", "--config", str(config),
        "--dry-run",
    ])
    assert result.exit_code == 0
    assert "Dry run complete." in result.output
    assert list(tmp_path.glob("run_*")) == []
    run_training.assert_not_called()
