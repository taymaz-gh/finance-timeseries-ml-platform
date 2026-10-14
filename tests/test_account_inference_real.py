"""Optional genuine Spark + persisted-Keras integration (Windows or Databricks).

Set FINANCE_ML_TEST_BUNDLE to the real local checkpoint directory, after
creating its synthetic_provenance.json sidecar with finance-ml profile-init.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance_ml.pipelines.account_inference_runner import run_account_inference


def test_real_account_date_inference_with_saved_model(tmp_path):
    bundle = os.environ.get("FINANCE_ML_TEST_BUNDLE")
    if not bundle:
        pytest.skip("Set FINANCE_ML_TEST_BUNDLE to execute real Spark/Keras inference")
    bundle_path = Path(bundle)
    if not (bundle_path / "synthetic_provenance.json").is_file():
        pytest.fail("Create bundle-bound synthetic provenance with finance-ml profile-init")
    result = run_account_inference(
        bundle_dir=bundle_path, account_ids=["1", "9"],
        start_date="2026-03-02", end_date="2026-03-03", smoke=True,
        output_dir=tmp_path,
    )
    predictions = pd.read_csv(result["run_dir"] / "predictions.csv")
    coverage = pd.read_csv(result["run_dir"] / "coverage.csv")
    assert len(coverage) == 4
    assert len(predictions) == 4
    assert not predictions.duplicated(["account_id", "date"]).any()
    assert set(predictions.cohort) == {"date_holdout", "joint_holdout"}
    prob_cols = [f"prob_{name}" for name in
                 ("normal", "elevated_risk", "liquidity_stress", "delinquency_risk")]
    assert np.isfinite(predictions[prob_cols].to_numpy()).all()
    assert np.allclose(predictions[prob_cols].sum(axis=1), 1.0, atol=1e-4)
    assert (coverage["status"] == "predicted").all()
