"""Fast inference validation, causal serving, and leakage regression tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finance_ml.pipelines import inference_runner as infer


class EndPointModel:
    def __init__(self):
        self.inputs = []

    def predict(self, x, **kwargs):
        self.inputs.append(x.copy())
        p = np.zeros((len(x), x.shape[1], 4), dtype=float)
        p[..., 0] = 1.0
        return p


def data():
    dates = pd.date_range("2026-01-01", periods=9)
    frame = pd.DataFrame({"account_id": ["A"] * 9, "date": dates,
                          "value": np.arange(9, dtype=float)})
    return frame


def test_causal_only_uses_past_and_present():
    frame = data()
    model = EndPointModel()
    expected = frame.iloc[[6, 7, 8]][["account_id", "date"]]
    predictions, coverage = infer.causal_endpoint_predictions(
        model, frame, expected, feature_columns=["value"],
        sequence_length=7, n_classes=4)
    assert len(predictions) == 3
    assert (coverage.status == "predicted").all()
    assert model.inputs[0].shape == (3, 7, 1)
    np.testing.assert_array_equal(model.inputs[0][0, :, 0], np.arange(7))
    np.testing.assert_array_equal(model.inputs[0][2, :, 0], np.arange(2, 9))


def test_cold_start_reports_missing_coverage():
    frame = data()
    model = EndPointModel()
    requested = frame[["account_id", "date"]]
    predictions, coverage = infer.causal_endpoint_predictions(
        model, frame, requested, feature_columns=["value"],
        sequence_length=7, n_classes=4)
    assert len(predictions) == 3
    assert (coverage.status == "insufficient_context").sum() == 6


def test_no_model_called_without_full_sequence():
    frame = data().iloc[:4]
    model = EndPointModel()
    predictions, coverage = infer.causal_endpoint_predictions(
        model, frame, frame[["account_id", "date"]],
        feature_columns=["value"], sequence_length=7, n_classes=4)
    assert predictions.empty and model.inputs == []
    assert len(coverage) == 4


def test_duplicate_keys_rejected():
    frame = pd.concat([data(), data().iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate"):
        infer.causal_endpoint_predictions(EndPointModel(), frame,
            data()[["account_id", "date"]], feature_columns=["value"],
            sequence_length=7, n_classes=4)


def test_invalid_probabilities_rejected():
    class Broken:
        def predict(self, x, **kwargs):
            return np.full((len(x), 7, 4), 0.7)
    frame = data()
    with pytest.raises(ValueError, match="invalid"):
        infer.causal_endpoint_predictions(Broken(), frame,
            frame.iloc[[6]][["account_id", "date"]],
            feature_columns=["value"], sequence_length=7, n_classes=4)


def test_evaluation_only_joins_at_end():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
    predicted = pd.DataFrame({"account_id": ["A", "A"], "date": dates,
                              "predicted_class": [0, 1]})
    truth = pd.DataFrame({"account_id": ["A", "A"], "date": dates,
                          "risk_state": [0, 2]})
    scores = infer._evaluate(predicted, truth, 4)
    assert scores["accuracy"] == 0.5
    assert scores["confusion_matrix"][2][1] == 1
    assert scores["n_evaluated"] == 2


def test_spark_rejects_labels_before_creation():
    with pytest.raises(ValueError, match="Label leakage"):
        infer._build_gold(object(), pd.DataFrame({"risk_state": [1]}))


def test_invalid_requests_prevent_heavy_import(monkeypatch):
    metadata = {"sequence_length": 7, "final_feature_columns": ["value"]}
    monkeypatch.setattr(infer, "inspect_local_bundle", lambda path: (metadata, {}))
    with pytest.raises(ValueError, match="row limit"):
        infer.validate_request(bundle_dir="bundle", existing_accounts=1000,
                               historical_days=60)
    with pytest.raises(ValueError, match="holdout"):
        infer.validate_request(bundle_dir="bundle", holdout="invalid")
    with pytest.raises(ValueError, match="historical_days"):
        infer.validate_request(bundle_dir="bundle", historical_days=2)


def test_smoke_is_bounded(monkeypatch):
    metadata = {"sequence_length": 7, "final_feature_columns": ["value"]}
    monkeypatch.setattr(infer, "inspect_local_bundle", lambda path: (metadata, {}))
    plan = infer.validate_request(bundle_dir="bundle", existing_accounts=25,
        future_days=90, new_accounts=10, smoke=True)
    assert plan["existing_accounts"] == 2
    assert plan["new_accounts"] == 1
    assert plan["future_days"] == 4


def test_no_future_timestep_used_for_early_date():
    frame = data()
    model = EndPointModel()
    infer.causal_endpoint_predictions(model, frame,
       frame.iloc[[6]][["account_id", "date"]],
       feature_columns=["value"], sequence_length=7, n_classes=4)
    assert model.inputs[0].max() == 6
