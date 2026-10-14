"""Providing command-line entry points for the financial ML platform."""

from __future__ import annotations

import json
from pathlib import Path

import typer

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
    data_source: str = typer.Option(
        "databricks",
        "--data-source",
        help="Training data source: databricks or synthetic.",
    ),
    synthetic_accounts: int = typer.Option(
        20,
        "--synthetic-accounts",
        min=1,
        help="Number of synthetic training accounts.",
    ),
    synthetic_days: int = typer.Option(
        60,
        "--synthetic-days",
        min=1,
        help="Number of historical days for synthetic training.",
    ),
    synthetic_seed: int = typer.Option(
        42,
        "--synthetic-seed",
        min=0,
        help="Synthetic training random seed.",
    ),
) -> None:
    """Validate training configuration and optionally execute training."""
    from finance_ml.config.config_loader import load_yaml_config
    from finance_ml.config.training_config import (
        TRAINING_CONFIG,
        validate_training_config,
    )

    # Copying defaults, applying YAML overrides, then explicit CLI arguments.
    options = dict(TRAINING_CONFIG)
    if config is not None:
        try:
            options.update(load_yaml_config(config, allowed_keys=TRAINING_CONFIG))
        except ValueError as exc:
            typer.echo(f"Configuration error: {exc}", err=True)
            raise typer.Exit(code=2) from exc

    cli_overrides = {
        "epochs": epochs,
        "batch_size": batch_size,
        "run_mode": run_mode,
        "query_method": query_method,
    }
    options.update({key: value for key, value in cli_overrides.items() if value is not None})

    try:
        validate_training_config(options)
    except (ValueError, TypeError, KeyError) as exc:
        typer.echo(f"Invalid training configuration: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if data_source not in {"databricks", "synthetic"}:
        typer.echo("Invalid data source: choose databricks or synthetic.", err=True)
        raise typer.Exit(code=2)

    if data_source == "synthetic":
        if options["log_to_mlflow"] or options["register_in_uc"]:
            typer.echo(
                "Synthetic training requires log_to_mlflow=false "
                "and register_in_uc=false.",
                err=True,
            )
            raise typer.Exit(code=2)

    # Keeping execution safeguards separate from configuration inspection.
    if not dry_run:
        if options["register_in_uc"] and not allow_uc_registration:
            typer.echo(
                "Unity Catalog registration requires --allow-uc-registration.",
                err=True,
            )
            raise typer.Exit(code=2)

        if data_source == "databricks" and options["query_method"] == "spark":
            typer.echo(
                "Standalone CLI training cannot access the configured "
                "Databricks Unity Catalog table through local Spark. "
                "Use --query-method connector or sqlalchemy. "
                "For managed Databricks Spark, call the training runner "
                "with the existing SparkSession.",
                err=True,
            )
            raise typer.Exit(code=2)

    typer.echo(f"Data source: {data_source}")
    if data_source == "synthetic":
        typer.echo("Synthetic data configuration:")
        typer.echo(f"  Accounts: {synthetic_accounts}")
        typer.echo(f"  Historical days: {synthetic_days}")
        typer.echo(f"  Random seed: {synthetic_seed}")
        typer.echo(
            "  Data acquisition: in-memory generation "
            "followed by local Spark Silver/Gold processing"
        )

    typer.echo("Training configuration:")
    typer.echo(json.dumps(options, indent=2))
    if dry_run:
        typer.echo("Dry run complete. No training was started.")
        return

    from finance_ml.pipelines.training_runner import run_training

    typer.echo("Starting model training...")
    try:
        if data_source == "synthetic":
            result = run_training(
                options,
                data_source="synthetic",
                synthetic_accounts=synthetic_accounts,
                synthetic_days=synthetic_days,
                synthetic_seed=synthetic_seed,
            )
        else:
            result = run_training(options)
    except Exception as exc:
        typer.echo(f"Training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("Training completed successfully.")
    typer.echo(f"MLflow model URI: {result.get('model_uri')}")
    if result.get("bundle_dir") is not None:
        typer.echo(f"Local model bundle: {result['bundle_dir']}")


@app.command()
def simulate(
    existing_accounts: int = typer.Option(
        3,
        "--existing-accounts",
        min=1,
        help="Number of existing synthetic accounts.",
    ),
    historical_days: int = typer.Option(
        10,
        "--historical-days",
        min=1,
        help="Number of historical observations per account.",
    ),
    future_days: int = typer.Option(
        4,
        "--future-days",
        min=1,
        help="Number of future observations per account.",
    ),
    new_accounts: int = typer.Option(
        2,
        "--new-accounts",
        min=0,
        help="Number of previously unseen synthetic accounts.",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        min=0,
        help="Random seed for reproducible generation.",
    ),
) -> None:
    """Preview reproducible synthetic financial data in memory."""
    from finance_ml.data.synthetic_sources import generate_in_memory_synthetic_data

    try:
        data = generate_in_memory_synthetic_data(
            n_existing_accounts=existing_accounts,
            historical_days=historical_days,
            future_days=future_days,
            n_new_accounts=new_accounts,
            random_seed=seed,
        )
    except (ValueError, TypeError) as exc:
        typer.echo(f"Simulation configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo("In-memory synthetic data:")
    for name, frame in data.items():
        typer.echo(f"  {name}: {len(frame)} rows")
    typer.echo("Future observations exclude ground-truth labels.")
    typer.echo("Simulation complete. No data was persisted.")
