"""Orchestrating model training with explicit Spark session ownership.

Reusing caller-provided or previously existing Spark sessions without
stopping them. Managing newly created standalone local Spark sessions
and preserving training failures when cleanup also fails.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Mapping

from finance_ml.config.training_config import validate_training_config


def _existing_spark_context() -> bool:
    """Check whether the Python process already owns a SparkContext.

    Using PySpark's internal context reference because its public session
    API does not reliably distinguish newly created and reused sessions.
    Keeping this version-sensitive inspection isolated in one helper.
    """
    from pyspark import SparkContext

    return SparkContext._active_spark_context is not None


def _create_local_spark_session() -> tuple[Any, bool]:
    """Acquire local Spark and report whether this runner created it."""
    from pyspark.sql import SparkSession

    # Reusing an active session without assuming ownership.
    active = SparkSession.getActiveSession()
    if active is not None:
        return active, False

    # Detecting existing Spark resources before calling getOrCreate().
    had_context = _existing_spark_context()

    if not had_context:
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    session = (
        SparkSession.builder.master("local[1]")
        .appName("finance-ml-training")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )

    # Owning only a Spark session created when no SparkContext existed.
    return session, not had_context


def run_training(
    config: Mapping[str, Any],
    *,
    spark: Any = None,
    pipeline_fn: Callable[..., dict[str, Any]] | None = None,
    data_source: str = "databricks",
    synthetic_accounts: int = 20,
    synthetic_days: int = 60,
    synthetic_seed: int = 42,
) -> dict[str, Any]:
    """Execute training with explicit resource management.

    Caller-provided Spark sessions are never stopped by this runner.
    Existing Spark sessions are reused without taking ownership.
    Newly created standalone local sessions are stopped after execution.
    Cleanup failures do not conceal an original training exception.
    """
    validate_training_config(config)

    if data_source not in {"databricks", "synthetic"}:
        raise ValueError("data_source must be databricks or synthetic")

    if data_source == "synthetic":
        if config["log_to_mlflow"] or config["register_in_uc"]:
            raise ValueError(
                "Synthetic training requires MLflow logging and "
                "UC registration to be disabled"
            )

        from finance_ml.data.synthetic_training import (
            MAX_SYNTHETIC_TRAINING_ROWS,
        )

        parameters = {
            "synthetic_accounts": synthetic_accounts,
            "synthetic_days": synthetic_days,
        }
        for name, value in parameters.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

        if type(synthetic_seed) is not int or synthetic_seed < 0:
            raise ValueError("synthetic_seed must be a non-negative integer")

        if synthetic_accounts * synthetic_days > MAX_SYNTHETIC_TRAINING_ROWS:
            raise ValueError(
                f"Synthetic training exceeds the "
                f"{MAX_SYNTHETIC_TRAINING_ROWS:,}-row limit"
            )

        n_days = synthetic_days
        n_train = int(n_days * config["train_fraction"])
        n_validation = int(n_days * config["validation_fraction"])
        n_test = n_days - n_train - n_validation

        if min(n_train, n_validation, n_test) < config["sequence_length"]:
            raise ValueError(
                "Each chronological partition must contain at least "
                "sequence_length days for synthetic training"
            )

    # Loading the full model pipeline only when execution is requested.
    if pipeline_fn is None:
        from finance_ml.pipelines.model_pipeline import run_model_pipeline

        pipeline_fn = run_model_pipeline

    owns_spark = False
    if spark is None:
        spark, owns_spark = _create_local_spark_session()

    training_error: BaseException | None = None

    try:
        if data_source == "synthetic":
            from finance_ml.data.synthetic_training import (
                prepare_synthetic_gold,
            )

            gold_df = prepare_synthetic_gold(
                spark,
                n_accounts=synthetic_accounts,
                n_days=synthetic_days,
                random_seed=synthetic_seed,
            )

            result = pipeline_fn(
                config=config,
                spark=spark,
                gold_df=gold_df,
            )
        else:
            result = pipeline_fn(config=config, spark=spark)

    except BaseException as exc:
        training_error = exc
        raise

    finally:
        if owns_spark:
            try:
                spark.stop()
            except Exception as cleanup_error:
                if training_error is not None:
                    training_error.add_note(
                        f"Spark cleanup also failed: {cleanup_error}"
                    )
                else:
                    raise

    return result
