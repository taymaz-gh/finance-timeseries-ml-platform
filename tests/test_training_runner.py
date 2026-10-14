"""Testing training orchestration without executing model training."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from finance_ml.config.training_config import TRAINING_CONFIG
from finance_ml.pipelines import training_runner


def test_reuses_supplied_spark_session() -> None:
    """Reusing an external Spark session without stopping it."""
    spark = Mock()
    pipeline = Mock(return_value={"model_uri": None})

    result = training_runner.run_training(
        TRAINING_CONFIG,
        spark=spark,
        pipeline_fn=pipeline,
    )

    assert result == {"model_uri": None}
    pipeline.assert_called_once_with(
        config=TRAINING_CONFIG,
        spark=spark,
    )
    spark.stop.assert_not_called()


def test_creates_and_stops_local_spark(monkeypatch) -> None:
    """Releasing a locally created Spark session after execution."""
    spark = Mock()
    pipeline = Mock(return_value={"model_uri": None})

    monkeypatch.setattr(
        training_runner,
        "_create_local_spark_session",
        lambda: (spark, True),
    )

    result = training_runner.run_training(
        TRAINING_CONFIG,
        pipeline_fn=pipeline,
    )

    assert result == {"model_uri": None}
    pipeline.assert_called_once_with(
        config=TRAINING_CONFIG,
        spark=spark,
    )
    spark.stop.assert_called_once()


def test_stops_local_spark_when_training_fails(monkeypatch) -> None:
    """Releasing local Spark resources even after pipeline failure."""
    spark = Mock()
    pipeline = Mock(side_effect=RuntimeError("Training failed"))

    monkeypatch.setattr(
        training_runner,
        "_create_local_spark_session",
        lambda: (spark, True),
    )

    with pytest.raises(RuntimeError, match="Training failed"):
        training_runner.run_training(
            TRAINING_CONFIG,
            pipeline_fn=pipeline,
        )

    spark.stop.assert_called_once()


def test_rejects_invalid_config_before_creating_spark(monkeypatch) -> None:
    """Rejecting invalid options before allocating Spark resources."""
    create_spark = Mock()

    monkeypatch.setattr(
        training_runner,
        "_create_local_spark_session",
        create_spark,
    )

    config = {**TRAINING_CONFIG, "epochs": -5}

    with pytest.raises(ValueError, match="epochs"):
        training_runner.run_training(
            config,
            pipeline_fn=Mock(),
        )

    create_spark.assert_not_called()


def test_reuses_active_spark_without_stopping(monkeypatch) -> None:
    """Preserving an already active Spark session."""
    spark = Mock()
    pipeline = Mock(return_value={"model_uri": None})

    monkeypatch.setattr(
        training_runner,
        "_create_local_spark_session",
        lambda: (spark, False),
    )

    result = training_runner.run_training(
        TRAINING_CONFIG,
        pipeline_fn=pipeline,
    )

    assert result == {"model_uri": None}
    pipeline.assert_called_once_with(config=TRAINING_CONFIG, spark=spark)
    spark.stop.assert_not_called()


def test_reuses_active_spark_session(monkeypatch) -> None:
    """Reusing an active session without creating another."""
    from pyspark.sql import SparkSession

    spark = Mock()
    get_or_create = Mock()

    monkeypatch.setattr(
        SparkSession,
        "getActiveSession",
        lambda: spark,
    )

    session, owns_spark = training_runner._create_local_spark_session()

    assert session is spark
    assert owns_spark is False
    get_or_create.assert_not_called()


def test_new_local_session_is_owned(monkeypatch) -> None:
    """Owning Spark only when no context existed before acquisition."""
    from pyspark.sql import SparkSession

    spark = Mock()
    builder = Mock()
    builder.master.return_value = builder
    builder.appName.return_value = builder
    builder.config.return_value = builder
    builder.getOrCreate.return_value = spark

    monkeypatch.setattr(SparkSession, "getActiveSession", lambda: None)
    monkeypatch.setattr(SparkSession, "builder", builder)
    monkeypatch.setattr(training_runner, "_existing_spark_context", lambda: False)

    session, owns_spark = training_runner._create_local_spark_session()

    assert session is spark
    assert owns_spark is True


def test_existing_context_is_not_owned(monkeypatch) -> None:
    """Preserving Spark resources created by another component."""
    from pyspark.sql import SparkSession

    spark = Mock()
    builder = Mock()
    builder.master.return_value = builder
    builder.appName.return_value = builder
    builder.config.return_value = builder
    builder.getOrCreate.return_value = spark

    monkeypatch.setattr(SparkSession, "getActiveSession", lambda: None)
    monkeypatch.setattr(SparkSession, "builder", builder)
    monkeypatch.setattr(training_runner, "_existing_spark_context", lambda: True)

    session, owns_spark = training_runner._create_local_spark_session()

    assert session is spark
    assert owns_spark is False


def test_cleanup_failure_preserves_training_error(monkeypatch) -> None:
    """Keeping the training error primary when Spark cleanup also fails."""
    spark = Mock()
    spark.stop.side_effect = RuntimeError("Spark stop failed")

    monkeypatch.setattr(
        training_runner,
        "_create_local_spark_session",
        lambda: (spark, True),
    )

    pipeline = Mock(side_effect=RuntimeError("Model training failed"))

    with pytest.raises(RuntimeError, match="Model training failed") as info:
        training_runner.run_training(
            TRAINING_CONFIG,
            pipeline_fn=pipeline,
        )

    assert "Spark cleanup also failed" in "\n".join(info.value.__notes__)
    spark.stop.assert_called_once()


def test_cleanup_failure_after_success_is_reported(monkeypatch) -> None:
    """Reporting cleanup failures even when model training succeeds."""
    spark = Mock()
    spark.stop.side_effect = RuntimeError("Spark stop failed")

    monkeypatch.setattr(
        training_runner,
        "_create_local_spark_session",
        lambda: (spark, True),
    )

    pipeline = Mock(return_value={"model_uri": None})

    with pytest.raises(RuntimeError, match="Spark stop failed"):
        training_runner.run_training(
            TRAINING_CONFIG,
            pipeline_fn=pipeline,
        )

    spark.stop.assert_called_once()


def test_external_spark_not_stopped_on_failure() -> None:
    """Preserving a caller-provided session after a training failure."""
    spark = Mock()
    pipeline = Mock(side_effect=RuntimeError("Training failed"))

    with pytest.raises(RuntimeError, match="Training failed"):
        training_runner.run_training(
            TRAINING_CONFIG,
            spark=spark,
            pipeline_fn=pipeline,
        )

    spark.stop.assert_not_called()


def test_invalid_synthetic_size_rejected_before_spark(monkeypatch) -> None:
    """Checking synthetic resource limits before creating Spark."""
    from finance_ml.config.training_config import TRAINING_CONFIG

    acquire = Mock()
    monkeypatch.setattr(training_runner, "_create_local_spark_session", acquire)

    config = {
        **TRAINING_CONFIG,
        "log_to_mlflow": False,
        "register_in_uc": False,
    }

    with pytest.raises(ValueError, match="row limit"):
        training_runner.run_training(
            config,
            data_source="synthetic",
            synthetic_accounts=1000,
            synthetic_days=60,
            pipeline_fn=Mock(),
        )

    acquire.assert_not_called()


def test_insufficient_synthetic_history_rejected_before_spark(
    monkeypatch,
) -> None:
    """Rejecting configurations unable to form validation sequences."""
    acquire = Mock()
    monkeypatch.setattr(training_runner, "_create_local_spark_session", acquire)

    config = {
        **TRAINING_CONFIG,
        "log_to_mlflow": False,
        "register_in_uc": False,
    }

    with pytest.raises(ValueError, match="chronological partition"):
        training_runner.run_training(
            config,
            data_source="synthetic",
            synthetic_accounts=8,
            synthetic_days=20,
            pipeline_fn=Mock(),
        )

    acquire.assert_not_called()
