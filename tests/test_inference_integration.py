"""Optional real Spark/Keras integration test using an existing Step 24 bundle.

Set FINANCE_ML_TEST_BUNDLE to a verified local-bundle directory. The test
never trains a model and is skipped unless the directory is explicitly set.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from finance_ml.pipelines.inference_runner import run_local_synthetic_inference


@pytest.mark.skipif(not os.getenv("FINANCE_ML_TEST_BUNDLE"),
                    reason="Set FINANCE_ML_TEST_BUNDLE to an existing Step 24 bundle")
def test_real_offline_inference_with_saved_bundle(spark, tmp_path):
    """Checking label separation, causal coverage, and true model predictions."""
    outcome = run_local_synthetic_inference(
        bundle_dir=Path(os.environ["FINANCE_ML_TEST_BUNDLE"]),
        spark=spark,
        holdout="both", existing_accounts=2, historical_days=14,
        future_days=4, new_accounts=1, seed=42, output_dir=tmp_path,
    )
    reports = outcome["summary"]["groups"]
    assert reports["temporal"]["requested"] == 8
    assert reports["temporal"]["predicted"] == 8
    assert reports["account"]["requested"] == 18
    assert reports["account"]["insufficient_context"] == 6
    for group in ("temporal", "account"):
        predictions = pd.read_csv(outcome["run_dir"] / f"{group}_predictions.csv")
        assert "risk_state" not in predictions.columns
        assert "risk_state_name" not in predictions.columns
        if group == "account":
            assert predictions["account_id"].str.startswith("NEW_").all()
        probability_columns = [c for c in predictions if c.startswith("prob_")]
        assert ((predictions[probability_columns].sum(axis=1) - 1).abs() < 1e-3).all()
