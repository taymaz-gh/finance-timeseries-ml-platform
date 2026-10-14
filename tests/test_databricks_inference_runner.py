"""Testing the end-to-end UC adapter via injected offline dependencies."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from finance_ml.pipelines.databricks_inference_runner import (
    _require_uc_contract, run_databricks_inference, uc_public_plan,
)
from finance_ml.data.databricks_holdout import validate_uc_request


def fake_bundle(nfeatures=41):
    numeric = [f"x{i}" for i in range(nfeatures)]
    metadata = {
        "model_columns": {"numeric_feature_columns": numeric, "categorical_feature_columns": []},
        "medians": {}, "category_levels": {},
        "scaler_parameters": {x: {"center": 0., "scale": 1., "method": "mad"} for x in numeric},
        "final_feature_columns": numeric, "sequence_length": 7, "stride": 1,
        "boundary_width": 2, "n_classes": 4,
    }
    model = SimpleNamespace(input_shape=(None, 7, nfeatures))
    return SimpleNamespace(model=model, metadata=metadata, model_name="finance_ml.models.lstm_risk_classifier",
                           model_version="1", metadata_sha256="a"*64)


def test_dynamic_metadata_validation_blocks_wrong_local_schema():
    with pytest.raises(ValueError, match="41"):
        _require_uc_contract(fake_bundle(31))
    assert len(_require_uc_contract(fake_bundle())["final_feature_columns"]) == 41


def test_remote_dry_run_plans_gold_without_remote_execution():
    req = validate_uc_request(account_ids=["1", "2001"], start_date="2026-03-02",
                              end_date="2026-03-03", evaluate=True)
    plan = uc_public_plan(req)
    assert plan["read_only_gold_tables"] == [
        "finance_ml.gold.account_features", "finance_ml.gold.account_features_temporal_holdout",
        "finance_ml.gold.account_features_account_holdout"
    ]
    assert plan["model_source"] == "unity_catalog"
    assert plan["cohorts"] == ["date_holdout", "joint_holdout"]
    assert len(plan["read_only_truth_tables"]) == 2


def test_inference_never_exposes_gold_labels_to_preprocessing(tmp_path):
    seen = {"queries": [], "gold": None, "stopped": False}
    def query(sql):
        seen["queries"].append(sql)
        assert sql.startswith("SELECT ") and "DROP" not in sql
        if "account_features" in sql:
            # Query only a requested known account on March 10-12 with 6-day prefix.
            return pd.DataFrame({
                "account_id": ["ACC_00521"]*9,
                "date": pd.date_range("2026-03-04", periods=9, tz="Etc/UTC"),
                "risk_state": [3]*9,  # Must be stripped before preprocess.
                "x0": [1.]*9,
            })
        raise AssertionError("Truth must not be queried without --evaluate")
    class Prepared:
        def toPandas(self):
            return pd.DataFrame({"account_id": ["ACC_00521"]*9,
                                 "date": pd.date_range("2026-03-04", periods=9, tz="UTC"),
                                 **{f"x{i}": [0.]*9 for i in range(41)}})
    def spark_frame(spark, frame, metadata):
        assert "risk_state" not in frame.columns
        assert str(frame["date"].dtype) == "datetime64[ns]"
        seen["gold"] = frame.copy()
        return frame
    def predict(model, frame, requested, *, feature_columns, sequence_length, n_classes):
        assert len(feature_columns) == 41 and sequence_length == 7
        rows = requested.copy()
        rows["predicted_class"] = 3
        rows["predicted_state_name"] = "delinquency_risk"
        rows.update(pd.DataFrame({"prob_normal": [0.]*3,
                                  "prob_elevated_risk": [0.]*3,
                                  "prob_liquidity_stress": [0.]*3,
                                  "prob_delinquency_risk": [1.]*3}))
        coverage = requested.copy()
        coverage["status"] = "predicted"
        return rows, coverage
    result = run_databricks_inference(
        account_ids=["521"], start_date="2026-03-10", end_date="2026-03-12",
        output_dir=tmp_path, spark=object(), query_fn=query,
        bundle_loader=lambda name, version: fake_bundle(),
        spark_frame_fn=spark_frame, preprocess_fn=lambda frame, metadata: Prepared(),
        prediction_fn=predict,
    )
    assert len(seen["queries"]) == 1
    assert all("SELECT" in sql for sql in seen["queries"])
    assert result["summary"]["predictions"] == 3
    assert result["summary"]["model_features"] == 41
    assert set(pd.read_csv(result["run_dir"] / "predictions.csv")["cohort"]) == {"date_holdout"}
    assert result["summary"]["evaluations"] == {}
    assert set(p.name for p in result["run_dir"].iterdir()) == {
        "predictions.csv", "coverage.csv", "inference_report.json"}


def test_missing_persisted_row_fails_instead_of_silent_coverage(tmp_path):
    def query(sql):
        return pd.DataFrame({"account_id": ["ACC_00521"], "date": ["2026-03-10"]})
    with pytest.raises(ValueError, match="Missing persisted Gold"):
        run_databricks_inference(account_ids=["521"], start_date="2026-03-10",
                                end_date="2026-03-11", output_dir=tmp_path,
                                bundle_loader=lambda *_: fake_bundle(), query_fn=query,
                                spark=object())
    assert list(tmp_path.iterdir()) == []


def test_optional_evaluation_queries_truth_after_predictions(tmp_path):
    events = []
    def query(sql):
        if "account_labels_temporal_holdout" in sql:
            events.append("truth")
            return pd.DataFrame({"account_id": ["ACC_00521"]*2,
                                 "date": pd.date_range("2026-03-10", periods=2, tz="Etc/UTC"),
                                 "risk_state": [1, 1]})
        events.append("gold")
        return pd.DataFrame({"account_id": ["ACC_00521"]*8,
                             "date": pd.date_range("2026-03-04", periods=8),
                             "x0": [2.]*8})
    class Prepared:
        def toPandas(self):
            return pd.DataFrame({"account_id": ["ACC_00521"]*8,
                "date": pd.date_range("2026-03-04", periods=8),
                **{f"x{i}": [0.]*8 for i in range(41)}})
    def predict(model, frame, requested, **kwargs):
        events.append("predict")
        rows = requested.copy()
        rows["predicted_class"] = 1
        rows["predicted_state_name"] = "elevated_risk"
        coverage = requested.copy()
        coverage["status"] = "predicted"
        return rows, coverage
    outcome = run_databricks_inference(
        account_ids=["521"], start_date="2026-03-10", end_date="2026-03-11",
        output_dir=tmp_path, spark=object(), query_fn=query,
        bundle_loader=lambda *_: fake_bundle(),
        spark_frame_fn=lambda spark, rows, metadata: rows,
        preprocess_fn=lambda frame, metadata: Prepared(),
        prediction_fn=predict, evaluate=True,
    )
    assert events == ["gold", "predict", "truth"]
    assert outcome["summary"]["evaluations"]["date_holdout"]["accuracy"] == 1.0
