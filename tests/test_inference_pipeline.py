"""Checking unlabeled windows and leakage-free transform orchestration."""
import numpy as np
from types import SimpleNamespace

from finance_ml.pipelines import inference_pipeline as pipeline


class Frame:
    def __init__(self, rows):
        self.rows = rows
        self.columns = list(rows[0])
    def select(self, *columns):
        return Frame([{k: r[k] for k in columns} for r in self.rows])
    def orderBy(self, *columns):
        return Frame(sorted(self.rows, key=lambda r: tuple(r[k] for k in columns)))
    def collect(self):
        return self.rows


def test_unlabeled_sequence_windows_preserve_order():
    rows = [{"account_id": "a", "date": i, "balance": float(i)} for i in range(9)]
    X, metadata = pipeline.build_unlabeled_sequences(Frame(rows), feature_columns=["balance"], group_col="account_id", time_col="date", sequence_length=7)
    assert X.shape == (3, 7, 1)
    assert metadata[1]["times"] == list(range(1, 8))
    assert np.array_equal(X[0, :, 0], np.arange(7))


def test_transform_uses_saved_parameters_only(monkeypatch):
    calls = []
    frame = Frame([{"account_id": "a", "date": 1, "balance": 3.0}])
    for name in ("apply_numeric_imputation", "apply_one_hot_encoding", "apply_scaling"):
        monkeypatch.setattr(pipeline, name, lambda df, metadata, n=name: (calls.append(n), df)[1])
    metadata = {"model_columns": {"numeric_feature_columns": ["balance"]}, "category_levels": {}, "medians": {}, "scaler_parameters": {}, "final_feature_columns": ["balance"]}
    transformed = pipeline.transform_for_inference(frame, metadata)
    assert transformed.columns == ["account_id", "date", "balance"]
    assert calls == ["apply_numeric_imputation", "apply_one_hot_encoding", "apply_scaling"]
