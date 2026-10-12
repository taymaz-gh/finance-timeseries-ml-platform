"""Checking that bounded inference reuses one model and preserves accounts."""
from types import SimpleNamespace
import pytest

from finance_ml.pipelines import batched_inference_pipeline as batcher


class FakeFrame:
    columns = ["account_id", "date"]

    def __init__(self, ids):
        self.ids = ids

    def select(self, *cols):
        return self

    def distinct(self):
        return self

    def orderBy(self, *cols):
        return self

    def collect(self):
        return [(x,) for x in self.ids]

    def filter(self, predicate):
        return self


@pytest.fixture
def fake_components(monkeypatch):
    calls = {"load": 0, "transform": 0, "predict": 0}
    metadata = {
        "final_feature_columns": ["x"],
        "sequence_length": 7,
        "stride": 1,
        "boundary_width": 2,
    }
    def load(*args, **kwargs):
        calls["load"] += 1
        return SimpleNamespace(model=object(), metadata=metadata, model_name="m", model_version="1", metadata_sha256="abc")
    def transform(df, *args, **kwargs):
        calls["transform"] += 1
        return df
    def sequences(df, **kwargs):
        return [0, 1], [{"times": [1] * 7}] * 2
    def predict(model, X, info, **kwargs):
        calls["predict"] += 1
        return [{"predicted_class": 0, "group": "a", "time": 1}]
    monkeypatch.setattr(batcher, "load_model_bundle", load)
    monkeypatch.setattr(batcher, "transform_for_inference", transform)
    monkeypatch.setattr(batcher, "build_unlabeled_sequences", sequences)
    monkeypatch.setattr(batcher, "predict_and_aggregate", predict)
    return calls


def test_batched_loads_once_and_returns_predictions(spark, fake_components):
    df = spark.createDataFrame(
        [(f"ACC_{i:02d}", "2026-04-01") for i in range(5)],
        ["account_id", "date"],
    )
    result = batcher.run_batched_inference_pipeline(
        df, accounts_per_batch=2, config={"boundary_width": 0}
    )
    assert result["n_accounts"] == 5
    assert result["n_batches"] == 3
    assert result["n_sequences"] == 6
    assert result["n_predictions"] == 3
    assert len(result["predictions"]) == 3
    assert fake_components == {"load": 1, "transform": 3, "predict": 3}


def test_callback_does_not_accumulate(spark, fake_components):
    df = spark.createDataFrame(
        [(f"ACC_{i:02d}", "2026-04-01") for i in range(3)],
        ["account_id", "date"],
    )
    received = []
    result = batcher.run_batched_inference_pipeline(
        df, accounts_per_batch=2, on_batch=lambda i, p: received.append((i, len(p)))
    )
    assert received == [("1", 1), ("2", 1)]
    assert result["predictions"] is None
    assert result["n_predictions"] == 2
    assert fake_components["load"] == 1


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_batch_counts_rejected(value):
    with pytest.raises(ValueError, match="accounts_per_batch"):
        batcher.run_batched_inference_pipeline(FakeFrame(["a"]), accounts_per_batch=value)


def test_empty_dataset_rejected(spark, fake_components):
    df = spark.createDataFrame([("a", "2026-01-01")], ["account_id", "date"]).limit(0)
    with pytest.raises(ValueError, match="No accounts"):
        batcher.run_batched_inference_pipeline(df)
