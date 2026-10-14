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
    query_method: str | None = typer.Option(
        None,
        "--query-method",
        help="Data query method: spark, connector, or sqlalchemy.",
    ),
    allow_uc_registration: bool = typer.Option(
        False,
        "--allow-uc-registration",
        help="Explicitly permit Unity Catalog model registration.",
    ),
) -> None:
    """Validate training configuration and optionally execute training."""
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
        "query_method": query_method,
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

    # Validating execution-specific safeguards without modifying ML defaults.
    if not dry_run:
        if options["register_in_uc"] and not allow_uc_registration:
            typer.echo(
                "Unity Catalog registration requires " "--allow-uc-registration.",
                err=True,
            )
            raise typer.Exit(code=2)

        if options["query_method"] == "spark":
            typer.echo(
                "Standalone CLI training cannot access the configured "
                "Databricks Unity Catalog table through local Spark. "
                "Use --query-method connector or sqlalchemy. "
                "For managed Databricks Spark, call the training runner "
                "with the existing SparkSession.",
                err=True,
            )
            raise typer.Exit(code=2)

    # Displaying the resolved configuration before any execution.
    typer.echo("Training configuration:")
    typer.echo(json.dumps(options, indent=2))

    if dry_run:
        typer.echo("Dry run complete. No training was started.")
        return

    # Loading training infrastructure only for actual execution.
    from finance_ml.pipelines.training_runner import run_training

    typer.echo("Starting model training...")

    try:
        result = run_training(options)
    except Exception as exc:
        typer.echo(
            f"Training failed: {exc}",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    typer.echo("Training completed successfully.")
    typer.echo(f"MLflow model URI: {result.get('model_uri')}")
