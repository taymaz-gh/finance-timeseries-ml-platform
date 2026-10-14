"""Testing the CLI without running external ML pipelines."""

import json

import pytest

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


def test_training_cli_executes_runner_when_not_dry_run(monkeypatch) -> None:
    """Dispatching validated training configuration to the runner."""
    from unittest.mock import Mock

    from finance_ml.config.training_config import TRAINING_CONFIG
    from finance_ml.pipelines import training_runner

    run_training = Mock(return_value={"model_uri": None})

    monkeypatch.setattr(
        training_runner,
        "run_training",
        run_training,
    )

    result = runner.invoke(
        app,
        ["train", "--epochs", "20", "--query-method", "connector"],
    )

    assert result.exit_code == 0
    assert "Training completed successfully." in result.output

    run_training.assert_called_once_with(
        {
            **TRAINING_CONFIG,
            "epochs": 20,
            "query_method": "connector",
        }
    )


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


def test_training_dry_run_never_calls_runner(monkeypatch) -> None:
    """Ensuring dry-run execution never starts the training runner."""
    from unittest.mock import Mock

    from finance_ml.pipelines import training_runner

    run_training = Mock()

    monkeypatch.setattr(
        training_runner,
        "run_training",
        run_training,
    )

    result = runner.invoke(
        app,
        ["train", "--config", "configs/training.yaml", "--dry-run"],
    )

    assert result.exit_code == 0
    assert "Dry run complete." in result.output
    run_training.assert_not_called()


def test_training_cli_reports_execution_failure(monkeypatch) -> None:
    """Reporting pipeline execution failures with a nonzero exit code."""
    from unittest.mock import Mock

    from finance_ml.pipelines import training_runner

    run_training = Mock(side_effect=RuntimeError("Simulated training failure"))

    monkeypatch.setattr(
        training_runner,
        "run_training",
        run_training,
    )

    result = runner.invoke(
        app,
        ["train", "--epochs", "20", "--query-method", "connector"],
    )

    assert result.exit_code == 1
    assert "Training failed: Simulated training failure" in result.output
    run_training.assert_called_once()


def test_invalid_training_config_does_not_start_runner(monkeypatch) -> None:
    """Rejecting invalid configuration before training dispatch."""
    from unittest.mock import Mock

    from finance_ml.pipelines import training_runner

    run_training = Mock()

    monkeypatch.setattr(
        training_runner,
        "run_training",
        run_training,
    )

    result = runner.invoke(
        app,
        ["train", "--run-mode", "unsupported"],
    )

    assert result.exit_code == 2
    assert "Invalid training configuration:" in result.output
    run_training.assert_not_called()


@pytest.mark.parametrize(
    "method",
    ["connector", "sqlalchemy"],
)
def test_training_cli_accepts_local_query_methods(
    monkeypatch,
    method: str,
) -> None:
    """Dispatching local training through a supported remote query method."""
    from unittest.mock import Mock

    from finance_ml.pipelines import training_runner

    run_training = Mock(return_value={"model_uri": None})
    monkeypatch.setattr(training_runner, "run_training", run_training)

    result = runner.invoke(
        app,
        ["train", "--query-method", method],
    )

    assert result.exit_code == 0
    run_training.assert_called_once()
    assert run_training.call_args.args[0]["query_method"] == method


def test_training_cli_rejects_local_spark_query(monkeypatch) -> None:
    """Preventing local Spark from querying Databricks UC directly."""
    from unittest.mock import Mock

    from finance_ml.pipelines import training_runner

    run_training = Mock()
    monkeypatch.setattr(training_runner, "run_training", run_training)

    result = runner.invoke(
        app,
        ["train", "--query-method", "spark"],
    )

    assert result.exit_code == 2
    assert "Standalone CLI training cannot access" in result.output
    run_training.assert_not_called()


def test_training_cli_requires_registration_approval(
    monkeypatch,
    tmp_path,
) -> None:
    """Preventing unintended Unity Catalog model registration."""
    from unittest.mock import Mock

    from finance_ml.pipelines import training_runner

    run_training = Mock()
    monkeypatch.setattr(training_runner, "run_training", run_training)

    path = tmp_path / "registration.yaml"
    path.write_text(
        "query_method: connector\n"
        "run_mode: final\n"
        "log_to_mlflow: true\n"
        "register_in_uc: true\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["train", "--config", str(path)],
    )

    assert result.exit_code == 2
    assert "--allow-uc-registration" in result.output
    run_training.assert_not_called()


def test_training_cli_allows_explicit_registration_approval(
    monkeypatch,
    tmp_path,
) -> None:
    """Dispatching registration only after explicit CLI approval."""
    from unittest.mock import Mock

    from finance_ml.pipelines import training_runner

    run_training = Mock(return_value={"model_uri": "runs:/example/model"})
    monkeypatch.setattr(training_runner, "run_training", run_training)

    path = tmp_path / "registration.yaml"
    path.write_text(
        "query_method: connector\n"
        "run_mode: final\n"
        "log_to_mlflow: true\n"
        "register_in_uc: true\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(path),
            "--allow-uc-registration",
        ],
    )

    assert result.exit_code == 0
    run_training.assert_called_once()
    assert run_training.call_args.args[0]["register_in_uc"] is True


def test_training_cli_spark_dry_run_remains_available() -> None:
    """Allowing Spark configuration inspection without execution."""
    result = runner.invoke(
        app,
        ["train", "--query-method", "spark", "--dry-run"],
    )

    assert result.exit_code == 0
    assert "Dry run complete." in result.output
