"""Testing the CLI without running external ML pipelines."""

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
