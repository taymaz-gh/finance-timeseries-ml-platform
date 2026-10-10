"""Running version-pinned, label-free inference on engineered Gold-layer records."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from finance_ml.data.query import query_with_spark, query_with_sql_connector, query_with_sqlalchemy
from finance_ml.models.inference import predict_and_aggregate
from finance_ml.models.loading import load_model_bundle
from finance_ml.preprocessing.encoding import apply_one_hot_encoding
from finance_ml.preprocessing.imputation import apply_numeric_imputation
from finance_ml.preprocessing.scaling import apply_scaling


INFERENCE_CONFIG: dict[str, Any] = {
    "model_name": "finance_ml.models.lstm_risk_classifier",
    "model_version": "1",
    "query_method": "spark",
    "source_table": "finance_ml.gold.account_features",
    "group_col": "account_id",
    "time_col": "date",
    "batch_size": 256,
}


def build_unlabeled_sequences(df, *, feature_columns, group_col, time_col, sequence_length, stride=1):
    """Construct training-compatible windows without requiring target labels.

    Materializes records on the driver, matching the existing training builder.
    Use a bounded inference batch; this is not a streaming implementation.
    """
    required = [group_col, time_col, *feature_columns]
    absent = sorted(set(required) - set(df.columns))
    if absent:
        raise ValueError(f"Missing sequence input columns: {absent}")
    if sequence_length < 1 or stride < 1:
        raise ValueError("sequence_length and stride must be positive")
    rows = df.select(*required).orderBy(group_col, time_col).collect()
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_col]].append(row)
    windows, details = [], []
    for group, items in grouped.items():
        for start in range(0, len(items) - sequence_length + 1, stride):
            part = items[start:start + sequence_length]
            windows.append([[float(row[name]) for name in feature_columns] for row in part])
            details.append({
                "group": group,
                "start_time": part[0][time_col],
                "end_time": part[-1][time_col],
                "times": [row[time_col] for row in part],
            })
    if not windows:
        raise ValueError("No complete sequences; provide at least sequence_length records per group")
    X = np.asarray(windows, dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Preprocessed inference features contain non-finite values")
    return X, details


def transform_for_inference(gold_df, metadata, *, group_col="account_id", time_col="date"):
    """Apply fitted training transforms, without fitting any new statistics."""
    model_columns = metadata["model_columns"]
    numeric = model_columns["numeric_feature_columns"]
    categorical = list(metadata["category_levels"])
    required = {group_col, time_col, *numeric, *categorical}
    missing = sorted(required - set(gold_df.columns))
    if missing:
        raise ValueError(f"Gold input missing required columns: {missing}")
    df = apply_numeric_imputation(gold_df, metadata["medians"])
    df = apply_one_hot_encoding(df, metadata["category_levels"])
    df = apply_scaling(df, metadata["scaler_parameters"])
    features = metadata["final_feature_columns"]
    remaining = sorted(set(features) - set(df.columns))
    if remaining:
        raise ValueError(f"Training features absent after transformations: {remaining}")
    return df.select(group_col, time_col, *features)


def run_inference_pipeline(config=None, *, spark=None, registry_client=None, tracking_client=None, model_loader=None):
    """Query Gold data, recover the UC model contract, and predict per entity/time.

    Returns a dictionary containing predictions and model provenance. It does
    not write results to a serving table or perform streaming inference.
    """
    options = {**INFERENCE_CONFIG, **(dict(config) if config else {})}
    if spark is None:
        raise ValueError("A SparkSession is required for the current inference pipeline")
    table = options["source_table"]
    if len(table.split(".")) != 3 or not all(p.replace("_", "").isalnum() for p in table.split(".")):
        raise ValueError("source_table must be catalog.schema.table")
    bundle = load_model_bundle(
        options["model_name"], options["model_version"],
        registry_client=registry_client, tracking_client=tracking_client, model_loader=model_loader,
    )
    sql = f"SELECT * FROM {table}"
    method = options["query_method"]
    if method == "spark":
        gold_df = query_with_spark(spark, sql)
    elif method in ("connector", "sqlalchemy"):
        pandas_df = query_with_sql_connector(sql) if method == "connector" else query_with_sqlalchemy(sql)
        if pandas_df.empty:
            raise ValueError("Query returned no records")
        gold_df = spark.createDataFrame(pandas_df)
    else:
        raise ValueError("query_method must be spark, connector, or sqlalchemy")
    m = bundle.metadata
    group_col, time_col = options["group_col"], options["time_col"]
    prepared = transform_for_inference(gold_df, m, group_col=group_col, time_col=time_col)
    X, sequence_metadata = build_unlabeled_sequences(
        prepared, feature_columns=m["final_feature_columns"],
        group_col=group_col, time_col=time_col,
        sequence_length=int(m["sequence_length"]), stride=int(m["stride"]),
    )
    predictions = predict_and_aggregate(
        bundle.model, X, sequence_metadata,
        sequence_length=int(m["sequence_length"]),
        boundary_width=int(m["boundary_width"]),
        batch_size=int(options["batch_size"]),
    )
    return {
        "predictions": predictions,
        "n_sequences": len(X),
        "n_predictions": len(predictions),
        "model_name": bundle.model_name,
        "model_version": bundle.model_version,
        "metadata_sha256": bundle.metadata_sha256,
        "feature_columns": m["final_feature_columns"],
    }
