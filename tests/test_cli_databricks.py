"""Testing Typer's UC route and rejecting unsupported source combinations."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from finance_ml.cli import app

RUNNER = CliRunner()
BASE = ["infer", "--data-source", "databricks", "--model-source", "uc",
        "--account-id", "521", "--start-date", "2026-03-10",
        "--end-date", "2026-03-12"]


def test_databricks_dry_run_has_no_side_effects(monkeypatch):
    import finance_ml.pipelines.databricks_inference_runner as m
    monkeypatch.setattr(m, "run_databricks_inference", lambda **kw: (_ for _ in ()).throw(
        AssertionError("dry-run must not execute remote inference")))
    response = RUNNER.invoke(app, [*BASE, "--dry-run"])
    assert response.exit_code == 0, response.output
    assert '"model_source": "unity_catalog"' in response.output
    assert "No remote query" in response.output
    assert "finance_ml.gold.account_features_temporal_holdout" in response.output


def test_databricks_smoke_dispatch(monkeypatch, tmp_path):
    import finance_ml.pipelines.databricks_inference_runner as m
    calls = []
    def run(**options):
        calls.append(options)
        return {"run_dir": tmp_path, "summary": {
            "predictions": 3, "requested": 3, "insufficient_context": 0}}
    monkeypatch.setattr(m, "run_databricks_inference", run)
    response = RUNNER.invoke(app, [*BASE, "--smoke"])
    assert response.exit_code == 0, response.output
    assert calls and calls[0]["model_version"] == "1"
    assert calls[0]["smoke"] is True
    assert "Predictions: 3/3" in response.output


def test_databricks_displays_evaluation_metrics(
    monkeypatch,
    tmp_path,
) -> None:
    """Checking that CLI output includes post-scoring metrics."""
    import finance_ml.pipelines.databricks_inference_runner as module

    def fake_inference(**options):
        # Returning representative completed inference results.
        assert options["evaluate"] is True

        return {
            "run_dir": tmp_path,
            "summary": {
                "predictions": 3,
                "requested": 3,
                "insufficient_context": 0,
                "evaluations": {
                    "date_holdout": {
                        "n_evaluated": 3,
                        "accuracy": 1.0,
                        "macro_f1": 0.25,
                    }
                },
            },
        }

    monkeypatch.setattr(
        module,
        "run_databricks_inference",
        fake_inference,
    )

    result = RUNNER.invoke(
        app,
        [*BASE, "--smoke", "--evaluate"],
    )

    assert result.exit_code == 0, result.output
    assert "Cohort: date_holdout" in result.output
    assert "Evaluated observations: 3" in result.output
    assert "Accuracy: 1.0000" in result.output
    assert "Macro-F1: 0.2500" in result.output


def test_databricks_rejects_synthetic_flags():
    for extra in (["--simulate-future"], ["--bundle", "artifacts/test"], ["--profile", "other.json"]):
        response = RUNNER.invoke(app, [*BASE, *extra, "--dry-run"])
        assert response.exit_code == 2
        assert "rejects" in response.output


def test_databricks_rejects_local_model_without_running():
    response = RUNNER.invoke(app, ["infer", "--data-source", "databricks",
        "--account-id", "521", "--start-date", "2026-03-10", "--end-date", "2026-03-12",
        "--dry-run"])
    assert response.exit_code == 2
    assert "requires --model-source uc" in response.output


def test_databricks_rejects_out_of_range_and_unapproved_model_version():
    for extra in (["--model-version", "3"], ["--account-id", "3001"]):
        response = RUNNER.invoke(app, [*BASE, *extra, "--dry-run"])
        assert response.exit_code == 2


def test_databricks_rejects_both_execution_flags():
    response = RUNNER.invoke(app, [*BASE, "--smoke", "--dry-run"])
    assert response.exit_code == 2
