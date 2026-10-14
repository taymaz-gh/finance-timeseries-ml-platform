"""Version-pinned, read-only UC account/date inference using saved transformations.

Queries a deliberately restricted set of already-engineered Gold/truth tables;
never creates/modifies source tables, registers models, or simulates data.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from finance_ml.data.databricks_holdout import (
    AVAILABLE_END, DEV_END, START, MODEL_NAME, MODEL_VERSION,
    UCRequest, cohort, execute_queries, gold_query_plan, truth_query_plan,
    validate_uc_request, normalize_utc_daily_dates,
)
from finance_ml.pipelines.inference_runner import (
    RISK_NAMES, _evaluate, apply_saved_preprocessing, causal_endpoint_predictions,
)


def uc_public_plan(request: UCRequest, *, model_name: str = MODEL_NAME,
                   model_version: str = MODEL_VERSION) -> dict[str, Any]:
    """Show remote tables/selection without reaching Databricks or importing MLflow."""
    cohorts = sorted({cohort(account, day.date()) for account in request.accounts
                      for day in pd.date_range(request.start, request.end, freq="D")})
    warnings = []
    if "development_period" in cohorts:
        warnings.append("Development-period predictions are NOT independent holdout evaluation.")
    return {
        "account_ids": list(request.accounts),
        "start_date": request.start.isoformat(), "end_date": request.end.isoformat(),
        "requested_rows": request.requested_rows,
        "account_population": "known 1-2000; new 2001-3000",
        "development_end": DEV_END.isoformat(),
        "available_dates": [START.isoformat(), AVAILABLE_END.isoformat()],
        "model_source": "unity_catalog", "model_name": model_name,
        "model_version": str(model_version), "data_source": "databricks",
        "feature_schema": "load verified UC metadata at execution (41 features expected for v1)",
        "prediction_policy": "causal_endpoint_v1", "query_method": request.query_method,
        "smoke": request.smoke, "evaluate": request.evaluate,
        "read_only_gold_tables": [table for table, _ in gold_query_plan(request)],
        "read_only_truth_tables": [table for table, _ in truth_query_plan(request)],
        "cohorts": cohorts, "warnings": warnings,
    }


def _query_function(method: str) -> Callable[[str], pd.DataFrame]:
    """Resolve an established Databricks query method only for a real run."""
    from finance_ml.data.query import query_with_sql_connector, query_with_sqlalchemy

    if method == "connector":
        return query_with_sql_connector
    if method == "sqlalchemy":
        return query_with_sqlalchemy
    raise ValueError("Unsupported Databricks query method")


def _load_uc_bundle(model_name: str, version: str):
    """Configure Databricks authentication and load a pinned, checksum-verified model.

    `DATABRICKS_SERVER_HOSTNAME` and `DATABRICKS_TOKEN` come from the project's
    existing .env loader. Respect explicitly configured DATABRICKS_HOST.
    """
    from finance_ml.utils.config import load_databricks_config
    config = load_databricks_config()
    os.environ.setdefault("DATABRICKS_HOST", "https://" + config["server_hostname"].removeprefix("https://"))
    os.environ.setdefault("DATABRICKS_TOKEN", config["access_token"])

    import mlflow
    from finance_ml.models.loading import load_model_bundle

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    return load_model_bundle(model_name, version)


def _require_uc_contract(bundle: Any) -> Mapping[str, Any]:
    """Refuse silent 31/41 feature mismatch and unsupported model shapes."""
    m = bundle.metadata
    expected = {"model_columns", "medians", "category_levels", "scaler_parameters",
                "final_feature_columns", "sequence_length", "stride", "boundary_width"}
    absent = sorted(expected - set(m))
    if absent:
        raise ValueError(f"UC preprocessing contract missing {absent}")
    if int(m["sequence_length"]) != 7 or len(m["final_feature_columns"]) != 41:
        raise ValueError("The original UC v1 model requires seven timesteps and 41 ordered features")
    if int(m.get("n_classes", 4)) != 4:
        raise ValueError("Expected four risk classes for the registered UC model")
    if len(m["final_feature_columns"]) != len(set(m["final_feature_columns"])):
        raise ValueError("Duplicate features in UC preprocessing contract")
    numeric = m["model_columns"].get("numeric_feature_columns")
    categoric = m["model_columns"].get("categorical_feature_columns")
    if not isinstance(numeric, list) or not isinstance(categoric, list):
        raise ValueError("Invalid UC model_columns schema")
    if set(m["scaler_parameters"]) != set(numeric):
        raise ValueError("UC scaler keys disagree with numeric features")
    if set(m["category_levels"]) != set(categoric):
        raise ValueError("UC category-level keys disagree with categorical features")
    # Checking the loaded model, not just the metadata description.
    shape = getattr(bundle.model, "input_shape", None)
    if isinstance(shape, (tuple, list)) and len(shape) == 3:
        if shape[1:] != (7, 41):
            raise ValueError("Loaded UC model has incompatible model input shape")
    return m


def gold_to_spark(spark: Any, gold: pd.DataFrame, metadata: Mapping[str, Any]):
    """Build a typed Spark Gold frame; convert SQL NULL/NaN to genuine nulls.

    Using the *saved* numeric/categorical source schema avoids Arrow or local
    Spark inference failures when a bounded sample contains all-null columns.
    """
    from pyspark.sql.types import DateType, DoubleType, StringType, StructField, StructType

    numeric = list(metadata["model_columns"]["numeric_feature_columns"])
    categorical = list(metadata["model_columns"]["categorical_feature_columns"])
    columns = ["account_id", "date", *numeric, *categorical]
    if len(columns) != len(set(columns)):
        raise ValueError("Overlapping source feature names in UC metadata")
    missing = sorted(set(columns) - set(gold.columns))
    if missing:
        raise ValueError(f"Persisted Gold is missing expected columns: {missing}")
    if gold.empty:
        raise ValueError("No Gold records available")
    schema = StructType([
        StructField("account_id", StringType(), False),
        StructField("date", DateType(), False),
        *[StructField(name, DoubleType(), True) for name in numeric],
        *[StructField(name, StringType(), True) for name in categorical],
    ])
    records = []
    for record in gold[columns].to_dict(orient="records"):
        day = pd.Timestamp(record["date"])
        if pd.isna(day):
            raise ValueError("Invalid date in Gold")
        if day.tzinfo is not None or day != day.normalize():
            raise ValueError("Gold date must be normalized UTC midnight")
        # Keeping daily keys timezone-free through local Spark, regardless of
        # the workstation/JVM timezone (including daylight-saving changes).
        row = [str(record["account_id"]), day.date()]
        for name in numeric:
            value = record[name]
            if pd.isna(value):
                row.append(None)
            else:
                result = float(value)
                if not math.isfinite(result):
                    raise ValueError(f"Non-finite numeric Gold input: {name}")
                row.append(result)
        for name in categorical:
            value = record[name]
            row.append(None if pd.isna(value) else str(value))
        records.append(tuple(row))
    return spark.createDataFrame(records, schema=schema)


def run_databricks_inference(*, account_ids: Sequence[str], start_date: str,
                             end_date: str, model_name: str = MODEL_NAME,
                             model_version: str = MODEL_VERSION,
                             query_method: str = "connector", smoke: bool = False,
                             evaluate: bool = False,
                             output_dir: str | Path = "artifacts/inference",
                             spark: Any = None,
                             query_fn: Callable[[str], pd.DataFrame] | None = None,
                             bundle_loader: Callable[[str, str], Any] | None = None,
                             preprocess_fn: Callable[[Any, Mapping[str, Any]], Any] | None = None,
                             prediction_fn: Callable[..., tuple[pd.DataFrame, pd.DataFrame]] | None = None,
                             spark_frame_fn: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Score UC v1 on selected persisted Gold dates without remote writes."""
    request = validate_uc_request(
        account_ids=account_ids, start_date=start_date, end_date=end_date,
        model_name=model_name, model_version=model_version,
        query_method=query_method, smoke=smoke, evaluate=evaluate,
    )
    root = Path(output_dir)
    if not str(output_dir).strip() or (root.exists() and not root.is_dir()):
        raise ValueError("output_dir must be a writable directory")
    query_fn = query_fn or _query_function(request.query_method)
    bundle_loader = bundle_loader or _load_uc_bundle
    preprocess_fn = preprocess_fn or apply_saved_preprocessing
    prediction_fn = prediction_fn or causal_endpoint_predictions
    spark_frame_fn = spark_frame_fn or gold_to_spark

    # Loading registered model only once; SHA verification is performed by loader.
    bundle = bundle_loader(model_name, model_version)
    metadata = _require_uc_contract(bundle)
    observations = execute_queries(gold_query_plan(
        request, sequence_length=int(metadata["sequence_length"]),
        source_columns=[*metadata["model_columns"]["numeric_feature_columns"],
                        *metadata["model_columns"]["categorical_feature_columns"]],
    ), query_fn)
    # Rejecting accidental target exposure immediately after Gold acquisition.
    observations = observations.drop(columns=["risk_state", "risk_state_name"], errors="ignore")
    requested = pd.DataFrame(
        [(account, day) for account in request.accounts
         for day in pd.date_range(request.start, request.end, freq="D")],
        columns=["account_id", "date"],
    )
    requested["date"] = normalize_utc_daily_dates(requested["date"])
    available = requested.merge(observations[["account_id", "date"]],
                                on=["account_id", "date"], how="left", indicator=True,
                                validate="one_to_one")
    if not available["_merge"].eq("both").all():
        missing_keys = available.loc[available["_merge"] != "both", ["account_id", "date"]]
        raise ValueError(f"Missing persisted Gold observations for {len(missing_keys)} requested account/dates")
    from finance_ml.pipelines.training_runner import _create_local_spark_session
    owns_spark = False
    if spark is None:
        spark, owns_spark = _create_local_spark_session()
    original_error: BaseException | None = None
    try:
        gold_spark = spark_frame_fn(spark, observations, metadata)
        prepared = preprocess_fn(gold_spark, metadata)
        prepared_pd = prepared.toPandas()
        # Normalizing Spark DateType output and any timezone-aware overrides
        # to the same daily key representation as requested observations.
        prepared_pd["date"] = normalize_utc_daily_dates(prepared_pd["date"])
        predictions, coverage = prediction_fn(
            bundle.model, prepared_pd, requested,
            feature_columns=metadata["final_feature_columns"],
            sequence_length=int(metadata["sequence_length"]), n_classes=4,
        )
        for frame in (predictions, coverage):
            frame["cohort"] = [cohort(account, day.date())
                               for account, day in zip(frame["account_id"], pd.to_datetime(frame["date"]))]
            frame["account_population"] = ["known" if int(x.split("_")[1]) <= 2000 else "new"
                                           for x in frame["account_id"]]
            frame["observation_origin"] = "persisted_databricks_gold"
    except BaseException as exc:
        original_error = exc
        raise
    finally:
        if owns_spark:
            try:
                spark.stop()
            except Exception as cleanup_error:
                if original_error is not None:
                    original_error.add_note(f"Spark cleanup also failed: {cleanup_error}")
                else:
                    raise

    if len(coverage) != request.requested_rows or len(predictions) != int(coverage["status"].eq("predicted").sum()):
        raise AssertionError("UC prediction coverage does not match the request")

    evaluations: dict[str, dict] = {}
    if request.evaluate and not predictions.empty and truth_query_plan(request):
        truths = execute_queries(truth_query_plan(request), query_fn)
        if "risk_state" not in truths.columns:
            raise ValueError("Requested UC evaluation truth missing risk_state")
        truths = truths[["account_id", "date", "risk_state"]]
        for name, rows in predictions.groupby("cohort"):
            if name == "development_period":
                continue
            selected_truth = truths.merge(rows[["account_id", "date"]],
                                          on=["account_id", "date"], validate="one_to_one")
            if len(selected_truth) != len(rows):
                raise ValueError("UC ground truth coverage incomplete")
            evaluations[name] = _evaluate(rows, selected_truth, 4)

    directory = root / (
        f"infer_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:10]}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(directory / "predictions.csv", index=False)
    coverage.to_csv(directory / "coverage.csv", index=False)
    summary = {
        "request": uc_public_plan(request, model_name=model_name, model_version=model_version),
        "model_name": bundle.model_name, "model_version": bundle.model_version,
        "metadata_sha256": bundle.metadata_sha256,
        "model_features": len(metadata["final_feature_columns"]),
        "prediction_policy": "causal_endpoint_v1",
        "predictions": len(predictions), "requested": len(coverage),
        "insufficient_context": int(coverage["status"].ne("predicted").sum()),
        "cohort_coverage": {
            name: {"requested": len(rows),
                   "predicted": int(rows["status"].eq("predicted").sum()),
                   "insufficient_context": int(rows["status"].ne("predicted").sum())}
            for name, rows in coverage.groupby("cohort")
        },
        "evaluations": evaluations,
        "metric_policy": "causal_endpoint_v1; UC labels queried only with --evaluate; development excluded",
    }
    (directory / "inference_report.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return {"run_dir": directory, "summary": summary}
