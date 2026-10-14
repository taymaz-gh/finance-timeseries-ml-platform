"""Testing the offline synthetic-data CLI command."""

from __future__ import annotations

from unittest.mock import Mock

from typer.testing import CliRunner

from finance_ml.cli import app


runner = CliRunner()


def test_simulate_command() -> None:
    """Running a small credential-independent demonstration."""
    result = runner.invoke(
        app,
        [
            "simulate",
            "--existing-accounts",
            "3",
            "--historical-days",
            "10",
            "--future-days",
            "4",
            "--new-accounts",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "historical: 30 rows" in result.output
    assert "temporal_observations: 12 rows" in result.output
    assert "account_observations: 28 rows" in result.output
    assert "No data was persisted" in result.output


def test_simulate_rejects_excessive_generation() -> None:
    """Preventing unexpectedly large local memory allocations."""
    result = runner.invoke(
        app,
        [
            "simulate",
            "--existing-accounts",
            "20000",
            "--historical-days",
            "60",
        ],
    )

    assert result.exit_code == 2
    assert "row limit" in result.output


def test_simulate_does_not_start_training(monkeypatch) -> None:
    """Keeping synthetic generation independent of training execution."""
    from finance_ml.pipelines import training_runner

    run_training = Mock()
    monkeypatch.setattr(
        training_runner,
        "run_training",
        run_training,
    )

    result = runner.invoke(app, ["simulate"])

    assert result.exit_code == 0
    run_training.assert_not_called()
