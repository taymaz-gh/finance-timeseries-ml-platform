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
