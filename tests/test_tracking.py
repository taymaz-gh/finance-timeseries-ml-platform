"""Tests for MLflow experiment-tracking utilities."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from finance_ml.models.tracking import (
    log_sequence_experiment,
)


def test_log_sequence_experiment_requires_active_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require an active MLflow run before logging experiment results."""
    fake_mlflow = SimpleNamespace(
        active_run=lambda: None,
    )

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        fake_mlflow,
    )

    result = SimpleNamespace(
        history=SimpleNamespace(
            history={
                "val_macro_f1": [0.80, 0.85],
            }
        ),
        validation_evaluation={
            "accuracy": 0.90,
            "macro_f1": 0.85,
        },
        test_evaluation=None,
        artifact_paths=None,
    )

    with pytest.raises(
        RuntimeError,
        match="active MLflow run is required",
    ):
        log_sequence_experiment(
            result=result,
            parameters={
                "learning_rate": 1e-3,
            },
        )


def test_log_sequence_experiment_logs_parameters_metrics_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Log experiment parameters, validation metrics, and artifacts."""
    logged_params = {}
    logged_metrics = {}
    logged_tags = {}
    logged_artifacts = []

    class FakeMLflow:
        def active_run(self):
            return object()

        def log_params(self, parameters):
            logged_params.update(parameters)

        def log_metrics(self, metrics):
            logged_metrics.update(metrics)

        def set_tag(self, key, value):
            logged_tags[key] = value

        def log_artifact(self, path, artifact_path=None):
            logged_artifacts.append(
                (
                    path,
                    artifact_path,
                )
            )

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        FakeMLflow(),
    )

    history = SimpleNamespace(
        history={
            "val_macro_f1": [
                0.80,
                0.85,
                0.83,
            ],
        }
    )

    artifact_1 = tmp_path / "training_history.csv"
    artifact_2 = tmp_path / "learning_curve_macro_f1.png"

    artifact_1.write_text("epoch,val_macro_f1\n1,0.80\n")
    artifact_2.write_text("fake image")

    result = SimpleNamespace(
        history=history,
        validation_evaluation={
            "accuracy": 0.90,
            "macro_f1": 0.85,
        },
        test_evaluation=None,
        artifact_paths={
            "history": artifact_1,
            "macro_f1": artifact_2,
        },
    )

    parameters = {
        "lstm_units": "(64,)",
        "dropout_rate": 0.3,
        "learning_rate": 1e-3,
    }

    log_sequence_experiment(
        result=result,
        parameters=parameters,
    )

    assert logged_params == parameters

    assert logged_metrics == {
        "validation_accuracy": 0.90,
        "validation_macro_f1": 0.85,
        "best_val_macro_f1": 0.85,
    }

    assert logged_tags == {
    "best_epoch": "2",
    }

    assert logged_artifacts == [
        (
            str(artifact_1),
            "training_artifacts",
        ),
        (
            str(artifact_2),
            "training_artifacts",
        ),
    ]


def test_log_sequence_experiment_logs_test_metrics_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Log test metrics when test evaluation is available."""
    logged_metrics = {}

    class FakeMLflow:
        def active_run(self):
            return object()

        def log_params(self, parameters):
            pass

        def log_metrics(self, metrics):
            logged_metrics.update(metrics)

        def log_artifact(self, path, artifact_path=None):
            pass

        def set_tag(self, key, value):
            pass

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        FakeMLflow(),
    )

    result = SimpleNamespace(
        history=SimpleNamespace(
            history={
                "val_macro_f1": [
                    0.80,
                    0.85,
                    0.83,
                ],
            }
        ),
        validation_evaluation={
            "accuracy": 0.90,
            "macro_f1": 0.85,
        },
        test_evaluation={
            "accuracy": 0.89,
            "macro_f1": 0.84,
        },
        artifact_paths=None,
    )

    log_sequence_experiment(
        result=result,
        parameters={
            "lstm_units": "(64,)",
            "dropout_rate": 0.3,
            "learning_rate": 1e-3,
        },
    )

    assert logged_metrics == {
        "validation_accuracy": 0.90,
        "validation_macro_f1": 0.85,
        "best_val_macro_f1": 0.85,
        "test_accuracy": 0.89,
        "test_macro_f1": 0.84,
    }


def test_log_sequence_experiment_returns_none_when_model_logging_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return None without logging a model when model logging is disabled."""

    class FakeMLflow:
        def active_run(self):
            return object()

        def log_params(self, parameters):
            pass

        def log_metrics(self, metrics):
            pass

        def set_tag(self, key, value):
            pass

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        FakeMLflow(),
    )

    result = SimpleNamespace(
        history=SimpleNamespace(
            history={
                "val_macro_f1": [0.80, 0.85],
            }
        ),
        validation_evaluation={
            "accuracy": 0.90,
            "macro_f1": 0.85,
        },
        test_evaluation=None,
        artifact_paths=None,
    )

    model_uri = log_sequence_experiment(
        result=result,
        parameters={
            "learning_rate": 1e-3,
        },
        log_model=False,
    )

    assert model_uri is None


def test_log_sequence_experiment_logs_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Log the trained Keras model when model logging is enabled."""
    logged_model = {}

    class FakeKeras:
        def log_model(
            self,
            model,
            name,
            signature,
        ):
            logged_model["model"] = model
            logged_model["name"] = name
            logged_model["signature"] = signature

            return SimpleNamespace(
                model_uri="runs:/test-run/model",
            )

    class FakeModels:
        @staticmethod
        def infer_signature(
            model_input,
            model_output,
        ):
            return "fake-signature"

    class FakeMLflow:
        keras = FakeKeras()
        models = FakeModels()

        def active_run(self):
            return object()

        def log_params(self, parameters):
            pass

        def log_metrics(self, metrics):
            pass

        def set_tag(self, key, value):
            pass

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        FakeMLflow(),
    )

    class FakeModel:
        def predict(
            self,
            model_input,
            verbose=0,
        ):
            return [
                [
                    [0.7, 0.2, 0.1],
                    [0.1, 0.8, 0.1],
                ]
            ]


    fake_model = FakeModel()

    input_example = [
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    ]

    result = SimpleNamespace(
        model=fake_model,
        history=SimpleNamespace(
            history={
                "val_macro_f1": [0.80, 0.85],
            }
        ),
        validation_evaluation={
            "accuracy": 0.90,
            "macro_f1": 0.85,
        },
        test_evaluation=None,
        artifact_paths=None,
    )

    model_uri = log_sequence_experiment(
        result=result,
        parameters={
            "lstm_units": "(64,)",
        },
        log_model=True,
        model_name="model",
        model_input_example=input_example,
    )

    assert model_uri == "runs:/test-run/model"

    assert logged_model["model"] is fake_model
    assert logged_model["name"] == "model"
    assert logged_model["signature"] == "fake-signature"


def test_log_sequence_experiment_requires_input_example_for_model_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require an input example when MLflow model logging is enabled."""

    class FakeKeras:
        def log_model(self, *args, **kwargs):
            raise AssertionError(
                "log_model should not be called."
            )

    class FakeMLflow:
        keras = FakeKeras()

        def active_run(self):
            return object()

        def log_params(self, parameters):
            pass

        def log_metrics(self, metrics):
            pass

        def set_tag(self, key, value):
            pass

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        FakeMLflow(),
    )

    result = SimpleNamespace(
        model=object(),
        history=SimpleNamespace(
            history={
                "val_macro_f1": [0.80, 0.85],
            }
        ),
        validation_evaluation={
            "accuracy": 0.90,
            "macro_f1": 0.85,
        },
        test_evaluation=None,
        artifact_paths=None,
    )

    with pytest.raises(
        ValueError,
        match="model_input_example is required",
    ):
        log_sequence_experiment(
            result=result,
            parameters={
                "lstm_units": "(64,)",
            },
            log_model=True,
        )