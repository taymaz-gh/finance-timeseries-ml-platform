"""Providing command-line entry points for the financial ML platform."""

from __future__ import annotations

import typer

from pathlib import Path

from finance_ml import __version__


app = typer.Typer(
    name="finance-ml",
    help="Command-line interface for the Financial Time-Series ML Platform.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Displaying the installed project version when requested."""
    if value:
        typer.echo(f"finance-ml {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the installed package version.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Providing the main CLI command group."""


@app.command()
def check() -> None:
    """Checking that the CLI is installed and accessible."""
    typer.echo("Financial Time-Series ML Platform")
    typer.echo(f"Version: {__version__}")
    typer.echo("CLI installation: OK")


@app.command()
def train(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and display settings without training.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="YAML file containing training configuration overrides.",
    ),
    epochs: int | None = typer.Option(
        None,
        "--epochs",
        min=1,
        help="Override the maximum number of training epochs.",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help="Override the training batch size.",
    ),
    run_mode: str | None = typer.Option(
        None,
        "--run-mode",
        help="Training mode: development or final.",
    ),
) -> None:
    """Validate training configuration without starting model training."""
    import json

    from finance_ml.config.config_loader import load_yaml_config
    from finance_ml.config.training_config import (
        TRAINING_CONFIG,
        validate_training_config,
    )

    # Copying the default training configuration.
    options = dict(TRAINING_CONFIG)

    # Loading experiment-specific YAML overrides.
    if config is not None:
        try:
            options.update(
                load_yaml_config(
                    config,
                    allowed_keys=TRAINING_CONFIG,
                )
            )
        except ValueError as exc:
            typer.echo(f"Configuration error: {exc}", err=True)
            raise typer.Exit(code=2) from exc

    # Applying explicit CLI overrides with highest precedence.
    cli_overrides = {
        "epochs": epochs,
        "batch_size": batch_size,
        "run_mode": run_mode,
    }

    options.update(
        {key: value for key, value in cli_overrides.items() if value is not None}
    )

    # Validating the resolved training configuration.
    try:
        validate_training_config(options)

    except (ValueError, TypeError, KeyError) as exc:
        typer.echo(
            f"Invalid training configuration: {exc}",
            err=True,
        )
        raise typer.Exit(code=2) from exc

    # Keeping actual training disabled until the execution adapter exists.
    if not dry_run:
        typer.echo(
            "Training execution is not yet connected. "
            "Use --dry-run to inspect the configuration.",
            err=True,
        )
        raise typer.Exit(code=2)

    typer.echo("Training configuration:")
    typer.echo(json.dumps(options, indent=2))
    typer.echo("Dry run complete. No training was started.")
