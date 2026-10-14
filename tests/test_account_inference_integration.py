"""Test the full account/date orchestration with real simulator, mock JVM/ML model."""
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from finance_ml.data.synthetic_provenance import SyntheticProvenance
from finance_ml.pipelines import account_inference_runner as runner
from finance_ml.pipelines import inference_runner


class FakeGold:
    def __init__(self, rows):
        self.rows = rows
        self.columns = list(rows.columns)

    def toPandas(self):
        return self.rows.copy()


class DeterministicModel:
    def predict(self, inputs, **kwargs):
        probs = np.zeros((*inputs.shape[:2], 4), dtype=float)
        probs[:, :, 0] = 1.0
        return probs


def _setup(monkeypatch):
    p = SyntheticProvenance("1" * 64, 2, 60, 42, date(2026, 1, 1),
                            date(2026, 4, 30), 1000)
    metadata = {"final_feature_columns": ["balance"], "sequence_length": 7, "n_classes": 4}
    monkeypatch.setattr(runner, "inspect_local_bundle", lambda path: (metadata, {"model": {"sha256": p.model_sha256}}))
    monkeypatch.setattr(runner, "load_profile", lambda *a, **kw: p)
    monkeypatch.setattr(runner, "load_local_bundle", lambda path: SimpleNamespace(
        model=DeterministicModel(), metadata=metadata))
    captured = []
    def build_gold(_spark, predictors):
        captured.append(predictors.copy())
        return FakeGold(predictors)
    monkeypatch.setattr(inference_runner, "_build_gold", build_gold)
    monkeypatch.setattr(inference_runner, "apply_saved_preprocessing", lambda gold, metadata: gold)
    return captured


def test_real_replay_to_predictions_and_cohort_report(tmp_path, monkeypatch):
    captured = _setup(monkeypatch)
    result = runner.run_account_inference(
        bundle_dir="ignored", spark=object(), account_ids=["1", "3"],
        start_date="2026-03-02", end_date="2026-03-05", output_dir=tmp_path)
    summary = result["summary"]
    assert summary["requested"] == summary["predictions"] == 8
    assert set(summary["evaluations"]) == {"date_holdout", "joint_holdout"}
    assert len(captured) == 1
    assert {"risk_state", "risk_state_name"}.isdisjoint(captured[0].columns)
    pred = pd.read_csv(result["run_dir"] / "predictions.csv")
    assert set(pred.cohort) == {"date_holdout", "joint_holdout"}
    assert pred.groupby("account_id").size().to_dict() == {"ACC_00001": 4, "ACC_00003": 4}
    assert len(pd.read_csv(result["run_dir"] / "coverage.csv")) == 8


def test_development_excluded_from_evaluation(tmp_path, monkeypatch):
    _setup(monkeypatch)
    result = runner.run_account_inference(
        bundle_dir="ignored", spark=object(), account_ids=["1"],
        start_date="2026-02-01", end_date="2026-02-02", output_dir=tmp_path)
    assert result["summary"]["predictions"] == 2
    assert result["summary"]["evaluations"] == {}


def test_new_account_first_six_days_have_no_prediction(tmp_path, monkeypatch):
    _setup(monkeypatch)
    result = runner.run_account_inference(
        bundle_dir="ignored", spark=object(), account_ids=["3"],
        start_date="2026-01-01", end_date="2026-01-08", output_dir=tmp_path)
    assert result["summary"]["predictions"] == 2
    assert result["summary"]["insufficient_context"] == 6


def test_future_extension_is_flagged_and_no_persistence_to_spark(tmp_path, monkeypatch):
    _setup(monkeypatch)
    result = runner.run_account_inference(
        bundle_dir="ignored", spark=object(), account_ids=["3"],
        start_date="2026-05-01", end_date="2026-05-03", output_dir=tmp_path,
        simulate_future=True)
    predictions = pd.read_csv(result["run_dir"] / "predictions.csv")
    assert (predictions["cohort"] == "simulated_extension").all()
    assert (predictions["observation_origin"] == "simulated_extension").all()
    assert result["summary"]["metric_policy"].startswith("causal_endpoint_v1")
