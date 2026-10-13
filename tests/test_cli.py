"""Testing the CLI without running external ML pipelines."""

import json

from typer.testing import CliRunner

from finance_ml.cli import app


runner = CliRunner()


def test_cli_help() -> None:
    """Checking the main help output."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "check" in result.output
    assert "Financial Time-Series ML Platform" in result.output


def test_cli_version() -> None:
    """Checking the version option."""
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_check() -> None:
    """Checking the basic installation command."""
    result = runner.invoke(app, ["check"])

    assert result.exit_code == 0
    assert "CLI installation: OK" in result.output


def _extract_training_config(output: str) -> dict:
    """Extracting the displayed JSON training configuration."""
    prefix = "Training configuration:\n"
    suffix = "\nDry run complete."

    config_text = output.split(prefix, 1)[1].split(suffix, 1)[0]
    return json.loads(config_text)


def test_training_cli_overrides() -> None:
    """Checking that explicit CLI options override defaults."""
    result = runner.invoke(
        app,
        [
            "train",
            "--dry-run",
            "--epochs",
            "20",
            "--batch-size",
            "128",
            "--run-mode",
            "development",
        ],
    )

    assert result.exit_code == 0

    config = _extract_training_config(result.output)

    assert config["epochs"] == 20
    assert config["batch_size"] == 128
    assert config["run_mode"] == "development"


def test_training_cli_defaults_remain_unchanged() -> None:
    """Checking that CLI overrides do not mutate shared defaults."""
    from finance_ml.config.training_config import TRAINING_CONFIG

    result = runner.invoke(
        app,
        ["train", "--dry-run", "--epochs", "20"],
    )

    assert result.exit_code == 0
    assert TRAINING_CONFIG["epochs"] == 50

    second_result = runner.invoke(app, ["train", "--dry-run"])
    config = _extract_training_config(second_result.output)

    assert config["epochs"] == 50


def test_training_cli_rejects_nonpositive_epochs() -> None:
    """Rejecting an invalid numeric option."""
    result = runner.invoke(
        app,
        ["train", "--dry-run", "--epochs", "0"],
    )

    assert result.exit_code != 0


def test_training_cli_rejects_invalid_mode() -> None:
    """Rejecting unsupported training modes."""
    result = runner.invoke(
        app,
        ["train", "--dry-run", "--run-mode", "invalid"],
    )

    assert result.exit_code != 0
    assert "Invalid training configuration" in result.output


def test_training_cli_requires_dry_run() -> None:
    """Preventing accidental training execution."""
    result = runner.invoke(app, ["train"])

    assert result.exit_code == 2
    assert "Training execution is not yet connected" in result.output


def test_training_cli_loads_yaml(tmp_path) -> None:
    """Loading experiment configuration through the CLI."""
    path = tmp_path / "training.yaml"
    path.write_text(
        "epochs: 20\nbatch_size: 128\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["train", "--config", str(path), "--dry-run"],
    )

    assert result.exit_code == 0

    config = _extract_training_config(result.output)
    assert config["epochs"] == 20
    assert config["batch_size"] == 128


def test_training_cli_override_precedence(tmp_path) -> None:
    """Giving explicit CLI arguments priority over YAML values."""
    path = tmp_path / "training.yaml"
    path.write_text(
        "epochs: 20\nbatch_size: 128\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(path),
            "--epochs",
            "35",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0

    config = _extract_training_config(result.output)

    assert config["epochs"] == 35
    assert config["batch_size"] == 128


def test_training_cli_rejects_invalid_yaml_value(tmp_path) -> None:
    """Rejecting invalid training parameters supplied through YAML."""
    path = tmp_path / "training.yaml"
    path.write_text("epochs: -5\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["train", "--config", str(path), "--dry-run"],
    )

    assert result.exit_code == 2
    assert "epochs must be a positive integer" in result.output


def test_training_cli_rejects_unknown_yaml_key(tmp_path) -> None:
    """Rejecting misspelled configuration parameters."""
    path = tmp_path / "training.yaml"
    path.write_text("epohcs: 20\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["train", "--config", str(path), "--dry-run"],
    )

    assert result.exit_code == 2
    assert "Unknown configuration keys" in result.output


def test_training_cli_rejects_missing_yaml_file(tmp_path) -> None:
    """Reporting missing configuration files clearly."""
    path = tmp_path / "missing.yaml"

    result = runner.invoke(
        app,
        ["train", "--config", str(path), "--dry-run"],
    )

    assert result.exit_code == 2
    assert "Configuration file not found" in result.output
