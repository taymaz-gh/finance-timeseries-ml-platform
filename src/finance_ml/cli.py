"""Providing command-line entry points for the financial ML platform."""

from __future__ import annotations

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
        help="Validate and display training settings without training.",
    ),
) -> None:
    """Preparing a training run without starting it."""

    if not dry_run:
        typer.echo(
            "Training execution is not yet connected. "
            "Use --dry-run to inspect the configuration.",
            err=True,
        )
        raise typer.Exit(code=2)

    # Loading the existing training configuration only when invoked.
    from finance_ml.pipelines.model_pipeline import (
        CONFIG,
        _validate_config,
    )

    import json

    options = dict(CONFIG)
    _validate_config(options)

    typer.echo("Training configuration:")
    typer.echo(json.dumps(options, indent=2))
    typer.echo("Dry run complete. No training was started.")

