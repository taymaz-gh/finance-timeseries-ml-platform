"""Testing in-memory DataFrame inference without changing table-based behavior."""
from types import SimpleNamespace

import numpy as np

from finance_ml.pipelines import inference_pipeline as pipeline


def test_in_memory_gold_dataframe_skips_table_queries(monkeypatch):
    """Verifying that a supplied Spark frame takes precedence over table access."""
    frame = object()
    marker = object()
    bundle = SimpleNamespace(
        metadata={"final_feature_columns": ["a"], "sequence_length": 7, "stride": 1, "boundary_width": 2},
        model=marker,
        model_name="finance_ml.models.test",
        model_version="1",
        metadata_sha256="abc",
    )
    monkeypatch.setattr(pipeline, "load_model_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(pipeline, "query_with_spark", lambda *args: (_ for _ in ()).throw(AssertionError("unexpected table query")))
    monkeypatch.setattr(pipeline, "transform_for_inference", lambda df, metadata, **kw: frame if df is frame else None)
    monkeypatch.setattr(pipeline, "build_unlabeled_sequences", lambda df, **kw: (np.zeros((1, 7, 1)), [{"group": "a"}]))
    monkeypatch.setattr(pipeline, "predict_and_aggregate", lambda *args, **kwargs: [{"predicted_class": 0}])
    result = pipeline.run_inference_pipeline(
        {"source_table": "not_a_catalog_table"}, spark=marker, gold_df=frame,
    )
    assert result["n_sequences"] == 1
    assert result["n_predictions"] == 1
    assert result["model_version"] == "1"
