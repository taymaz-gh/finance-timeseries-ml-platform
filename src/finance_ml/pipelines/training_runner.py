"""Running model training in local or externally managed Spark environments.

Keeping Spark session management separate from the training pipeline
and command-line interface.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Mapping

from finance_ml.config.training_config import validate_training_config


def _create_local_spark_session() -> tuple[Any, bool]:
    """Acquire a Spark session without assuming ownership of reused sessions."""
    from pyspark.sql import SparkSession

    # Reusing a Spark session active in the calling thread.
    existing = SparkSession.getActiveSession()

    if existing is not None:
        return existing, False

    # Configuring local workers to use the driver's Python interpreter.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # getOrCreate() may reuse a default session even when none is active.
    session = (
        SparkSession.builder.master("local[1]")
        .appName("finance-ml-training")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )

    # Conservatively treating the returned session as externally managed.
    return session, False


def run_training(
    config: Mapping[str, Any],
    *,
    spark: Any = None,
    pipeline_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute the model pipeline using an available Spark session.

    Validating the configuration before allocating execution resources.
    Reusing caller-supplied sessions without managing their lifecycle.

    Acquiring a local session when none is supplied. Sessions returned
    by Spark's getOrCreate() are not assumed to be exclusively owned,
    because Spark may reuse an existing default session.

    Accepting an injectable pipeline callable for lightweight tests.
    """
    validate_training_config(config)

    # Loading the production pipeline only when actually needed.
    if pipeline_fn is None:
        from finance_ml.pipelines.model_pipeline import run_model_pipeline

        pipeline_fn = run_model_pipeline

    # Tracking whether the runner owns the Spark session.
    owns_spark = False

    if spark is None:
        spark, owns_spark = _create_local_spark_session()

    try:
        return pipeline_fn(config=config, spark=spark)
    finally:
        if owns_spark:
            spark.stop()
