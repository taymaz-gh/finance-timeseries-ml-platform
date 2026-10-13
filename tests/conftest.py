"""Shared pytest fixtures."""

from __future__ import annotations

import os

import sys

# Matching PySpark workers to the active pytest Python interpreter.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from collections.abc import Generator

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    """Creating a Spark session appropriate for the current environment."""
    running_in_databricks = (
        "DATABRICKS_RUNTIME_VERSION" in os.environ
    )

    if running_in_databricks:
        # Reusing the active Spark Connect session provided by
        # the Databricks notebook runtime.
        session = SparkSession.getActiveSession()

        if session is None:
            raise RuntimeError(
                "No active Spark session is available "
                "in the Databricks notebook process."
            )

        yield session
        return

    # Creating a local Spark session for local development and CI.
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("finance-timeseries-ml-platform-tests")
        .getOrCreate()
    )

    yield session

    session.stop()
