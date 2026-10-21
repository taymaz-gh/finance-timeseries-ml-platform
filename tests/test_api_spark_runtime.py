"""Testing reusable Spark resources for API inference."""

from unittest.mock import Mock

import pandas as pd
import pytest

from finance_ml.api import spark_runtime


def test_lazy_initialization_and_reuse(monkeypatch) -> None:
    """Checking that one owned Spark session serves multiple requests."""

    session = Mock()
    acquire = Mock(return_value=(session, True))
    prediction = Mock(return_value={"predictions": []})

    monkeypatch.setattr(
        "finance_ml.pipelines.training_runner._create_local_spark_session",
        acquire,
    )
    monkeypatch.setattr(
        spark_runtime,
        "predict_uploaded_accounts",
        prediction,
    )

    runtime = spark_runtime.SparkRuntime()
    frame = pd.DataFrame()
    service = Mock()

    # Verifying that construction does not start Spark.
    acquire.assert_not_called()

    # Executing two inference requests.
    runtime.predict(frame, service)
    runtime.predict(frame, service)

    # Confirming one-time initialization and session reuse.
    acquire.assert_called_once()
    assert prediction.call_count == 2

    for call in prediction.call_args_list:
        assert call.kwargs["spark"] is session
        assert call.kwargs["service"] is service

    # Releasing an owned Spark session exactly once.
    runtime.close()
    runtime.close()

    session.stop.assert_called_once()


def test_borrowed_session_is_not_stopped(monkeypatch) -> None:
    """Preserving Spark sessions owned by another component."""

    session = Mock()
    acquire = Mock(return_value=(session, False))
    prediction = Mock(return_value={"predictions": []})

    monkeypatch.setattr(
        "finance_ml.pipelines.training_runner._create_local_spark_session",
        acquire,
    )
    monkeypatch.setattr(
        spark_runtime,
        "predict_uploaded_accounts",
        prediction,
    )

    runtime = spark_runtime.SparkRuntime()

    runtime.predict(pd.DataFrame(), Mock())
    runtime.close()

    acquire.assert_called_once()
    session.stop.assert_not_called()


def test_cleanup_after_prediction_failure(monkeypatch) -> None:
    """Allowing owned Spark resources to be released after inference errors."""

    session = Mock()

    monkeypatch.setattr(
        "finance_ml.pipelines.training_runner._create_local_spark_session",
        Mock(return_value=(session, True)),
    )
    monkeypatch.setattr(
        spark_runtime,
        "predict_uploaded_accounts",
        Mock(side_effect=ValueError("Inference failed")),
    )

    runtime = spark_runtime.SparkRuntime()

    with pytest.raises(ValueError, match="Inference failed"):
        runtime.predict(pd.DataFrame(), Mock())

    runtime.close()

    session.stop.assert_called_once()
