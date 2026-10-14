"""Running local-bundle inference with causal, label-free holdout features.

Keeping generation, Spark preprocessing, fitted transforms, prediction and
reporting separate. Never fitting preprocessing on inference observations.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from finance_ml.data.synthetic_sources import generate_in_memory_synthetic_data
from finance_ml.models.local_bundle import inspect_local_bundle, load_local_bundle

LABELS = ("risk_state", "risk_state_name")
RISK_NAMES = ("normal", "elevated_risk", "liquidity_stress", "delinquency_risk")
MAX_ROWS = 20_000


def validate_request(*, bundle_dir: str | Path, holdout: str = "both",
                     existing_accounts: int = 3, historical_days: int = 60,
                     future_days: int = 14, new_accounts: int = 2,
                     seed: int = 42, output_dir: str | Path = "artifacts/inference",
                     smoke: bool = False) -> dict[str, Any]:
    """Validate before allocating Spark or importing TensorFlow."""
    metadata, manifest = inspect_local_bundle(bundle_dir)
    if holdout not in {"temporal", "account", "both"}:
        raise ValueError("holdout must be temporal, account, or both")
    for name, value, minimum in (("existing_accounts", existing_accounts, 1),
                                 ("historical_days", historical_days, 1),
                                 ("future_days", future_days, 1),
                                 ("new_accounts", new_accounts, 0),
                                 ("seed", seed, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if holdout in {"account", "both"} and not new_accounts:
        raise ValueError("new_accounts must be positive for account holdout")
    if historical_days < metadata["sequence_length"] - 1:
        raise ValueError("historical_days is too short for causal sequence context")
    effective = {
        "existing_accounts": min(existing_accounts, 2) if smoke else existing_accounts,
        "historical_days": historical_days,
        "future_days": min(future_days, 4) if smoke else future_days,
        "new_accounts": min(new_accounts, 1) if smoke else new_accounts,
    }
    # Simulator constructs full trajectories for new accounts, including warm-up.
    rows = (effective["existing_accounts"] + effective["new_accounts"]) * (
        effective["historical_days"] + effective["future_days"]
    )
    if rows > MAX_ROWS:
        raise ValueError(f"Synthetic inference exceeds the {MAX_ROWS:,}-row limit")
    if Path(output_dir).exists() and not Path(output_dir).is_dir():
        raise ValueError("output_dir points to an existing file")
    if not str(output_dir).strip():
        raise ValueError("output_dir must not be empty")
    return {"metadata": metadata, "manifest": manifest, "holdout": holdout,
            "seed": seed, "output_dir": str(output_dir), "smoke": smoke,
            **effective, "generated_rows": rows,
            "prediction_policy": "causal_endpoint_v1",
            "model_source": "local_bundle", "data_source": "synthetic"}


def _spark_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert numpy/pandas scalars before local Spark schema inference."""
    records = frame.to_dict(orient="records")
    for row in records:
        for name, value in row.items():
            if isinstance(value, np.generic):
                row[name] = value.item()
            elif isinstance(value, pd.Timestamp):
                row[name] = value.to_pydatetime()
    return records


def _build_gold(spark: Any, frame: pd.DataFrame) -> Any:
    """Transform unlabeled history and holdout together to preserve causality."""
    if frame.empty:
        raise ValueError("Cannot prepare an empty inference dataset")
    if any(key in frame.columns for key in LABELS):
        raise ValueError("Label leakage: inference input contains target columns")
    from finance_ml.lakehouse.build_silver import build_silver_dataframe
    from finance_ml.lakehouse.build_gold import build_gold_features
    bronze = spark.createDataFrame(_spark_records(frame))
    silver = build_silver_dataframe(bronze, require_labels=False)
    gold = build_gold_features(silver)
    if any(key in gold.columns for key in LABELS):
        raise ValueError("Label leakage: Gold features contain target columns")
    return gold


def apply_saved_preprocessing(gold: Any, metadata: dict[str, Any]) -> Any:
    """Replay training-fitted imputation, encoding, and scaling with fixed order."""
    from finance_ml.preprocessing.imputation import apply_numeric_imputation
    from finance_ml.preprocessing.encoding import apply_one_hot_encoding
    from finance_ml.preprocessing.scaling import apply_scaling
    model_columns = metadata["model_columns"]
    required = set(model_columns["numeric_feature_columns"]) | set(
        model_columns["categorical_feature_columns"]
    ) | {"account_id", "date"}
    missing = sorted(required - set(gold.columns))
    if missing:
        raise ValueError("Inference Gold schema missing columns: " + ", ".join(missing))
    transformed = apply_numeric_imputation(gold, metadata["medians"])
    transformed = apply_one_hot_encoding(transformed, metadata["category_levels"])
    transformed = apply_scaling(transformed, metadata["scaler_parameters"])
    columns = ["account_id", "date", *metadata["final_feature_columns"]]
    if len(columns) != len(set(columns)) or not set(columns) <= set(transformed.columns):
        raise ValueError("Saved feature order cannot be reconstructed")
    return transformed.select(*columns)


def causal_endpoint_predictions(model: Any, rows: pd.DataFrame,
                                requested: pd.DataFrame, *,
                                feature_columns: list[str], sequence_length: int,
                                n_classes: int, batch_size: int = 128) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict only at requested timestamps using observations up to each date.

    Avoiding boundary trimming: an endpoint window predicts its final timestep
    without requiring future observations. Scoring no target from future data.
    """
    keys = ["account_id", "date"]
    if rows.duplicated(keys).any() or requested.duplicated(keys).any():
        raise ValueError("Duplicate account/date keys are not allowed")
    if not set(keys + feature_columns) <= set(rows.columns):
        raise ValueError("Inference input is missing model features")
    if not set(keys) <= set(requested.columns):
        raise ValueError("Requested observations must have account_id and date")
    rows = rows.copy()
    requested = requested[keys].copy()
    rows["date"] = pd.to_datetime(rows["date"])
    requested["date"] = pd.to_datetime(requested["date"])
    requests = set(map(tuple, requested[keys].itertuples(index=False, name=None)))
    features = rows[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Inference features contain non-finite values")
    rows = rows.sort_values(keys).reset_index(drop=True)
    X, output_keys = [], []
    for group, group_rows in rows.groupby("account_id", sort=True):
        ordered = group_rows.sort_values("date")
        dates = ordered["date"].tolist()
        matrix = ordered[feature_columns].to_numpy(dtype=np.float32)
        for idx, date in enumerate(dates):
            if (group, date) not in requests:
                continue
            if idx + 1 < sequence_length:
                continue
            X.append(matrix[idx + 1 - sequence_length:idx + 1])
            output_keys.append((group, date))
    if X:
        predictions = np.asarray(model.predict(np.stack(X), batch_size=batch_size, verbose=0))
        if predictions.shape != (len(X), sequence_length, n_classes):
            raise ValueError("Model prediction shape disagrees with bundle metadata")
        probabilities = predictions[:, -1, :]
        if (not np.isfinite(probabilities).all() or (probabilities < -1e-6).any()
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-3)):
            raise ValueError("Model returned invalid class probabilities")
    else:
        probabilities = np.empty((0, n_classes))
    results = []
    for (account_id, date), probability in zip(output_keys, probabilities):
        label = int(np.argmax(probability))
        record = {"account_id": account_id, "date": date,
                  "predicted_class": label,
                  "predicted_state_name": RISK_NAMES[label]}
        record.update({f"prob_{name}": float(probability[i])
                       for i, name in enumerate(RISK_NAMES)})
        results.append(record)
    predictions_df = pd.DataFrame(results, columns=[*keys, "predicted_class",
        "predicted_state_name", *(f"prob_{x}" for x in RISK_NAMES)])
    coverage = requested.merge(predictions_df[keys], how="left", on=keys, indicator=True)
    coverage["status"] = np.where(coverage["_merge"] == "both", "predicted", "insufficient_context")
    coverage = coverage.drop(columns="_merge")
    # Refusing silently missing observations or duplicated requests.
    if len(coverage) != len(requested):
        raise AssertionError("Coverage accounting mismatch")
    return predictions_df.sort_values(keys).reset_index(drop=True), coverage.sort_values(keys).reset_index(drop=True)


def _evaluate(predictions: pd.DataFrame, truth: pd.DataFrame, n_classes: int) -> dict[str, Any]:
    """Evaluate strictly after prediction, never exposing labels to features."""
    labels = truth[["account_id", "date", "risk_state"]].copy()
    labels["date"] = pd.to_datetime(labels["date"])
    if labels.duplicated(["account_id", "date"]).any():
        raise ValueError("Duplicate truth keys")
    scored = predictions.merge(labels, on=["account_id", "date"], validate="one_to_one")
    if scored.empty:
        return {"n_evaluated": 0, "accuracy": None, "macro_f1": None}
    y_true = scored["risk_state"].astype(int).to_numpy()
    y_pred = scored["predicted_class"].astype(int).to_numpy()
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1
    per_class = {}
    f1s = []
    for i, name in enumerate(RISK_NAMES):
        tp = int(matrix[i, i])
        support = int(matrix[i].sum())
        predicted_count = int(matrix[:, i].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": f1, "support": support}
        f1s.append(f1)
    return {"n_evaluated": len(scored),
            "accuracy": float(np.mean(y_true == y_pred)),
            "macro_f1": float(np.mean(f1s)),
            "per_class": per_class, "confusion_matrix": matrix.tolist()}



def run_local_synthetic_inference(*, bundle_dir: str | Path, spark: Any = None,
                                  holdout: str = "both", existing_accounts: int = 3,
                                  historical_days: int = 60, future_days: int = 14,
                                  new_accounts: int = 2, seed: int = 42,
                                  output_dir: str | Path = "artifacts/inference",
                                  smoke: bool = False) -> dict[str, Any]:
    """Run requested holdouts, publishing local reports only after success."""
    request = validate_request(bundle_dir=bundle_dir, holdout=holdout,
        existing_accounts=existing_accounts, historical_days=historical_days,
        future_days=future_days, new_accounts=new_accounts, seed=seed,
        output_dir=output_dir, smoke=smoke)
    # Validating and loading model before starting Spark where possible.
    bundle = load_local_bundle(bundle_dir)
    from finance_ml.pipelines.training_runner import _create_local_spark_session
    owns_spark = False
    if spark is None:
        spark, owns_spark = _create_local_spark_session()
    initial_error = None
    try:
        generated = generate_in_memory_synthetic_data(
            n_existing_accounts=request["existing_accounts"],
            historical_days=request["historical_days"],
            future_days=request["future_days"],
            n_new_accounts=request["new_accounts"], random_seed=seed)
        # Giving new accounts a distinct namespace even when account counts
        # overlap those used by the previously trained synthetic model.
        # Renaming identifiers does not change financial trajectories or RNG.
        if not generated["account_observations"].empty:
            for key in ("account_observations", "account_truth"):
                generated[key] = generated[key].copy()
                generated[key]["account_id"] = generated[key]["account_id"].map(
                    lambda value: f"NEW_{value}"
                )
        results = {}
        context = generated["historical"].drop(columns=list(LABELS))
        for name in (["temporal", "account"] if holdout == "both" else [holdout]):
            if name == "temporal":
                observations = generated["temporal_observations"]
                context_and_future = pd.concat([context, observations], ignore_index=True)
                truth = generated["temporal_truth"]
            else:
                observations = generated["account_observations"]
                context_and_future = observations
                truth = generated["account_truth"]
            gold = _build_gold(spark, context_and_future)
            transformed = apply_saved_preprocessing(gold, bundle.metadata)
            frame = transformed.toPandas()
            predicted, coverage = causal_endpoint_predictions(
                bundle.model, frame, observations[["account_id", "date"]],
                feature_columns=bundle.metadata["final_feature_columns"],
                sequence_length=bundle.metadata["sequence_length"],
                n_classes=bundle.metadata["n_classes"])
            evaluation = _evaluate(predicted, truth, bundle.metadata["n_classes"])
            results[name] = (predicted, coverage, evaluation)
    except BaseException as exc:
        initial_error = exc
        raise
    finally:
        if owns_spark:
            try:
                spark.stop()
            except Exception as cleanup_error:
                if initial_error is not None:
                    initial_error.add_note(f"Spark cleanup also failed: {cleanup_error}")
                else:
                    raise
    root = Path(output_dir)
    run_dir = root / f"infer_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:10]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {"bundle_dir": str(Path(bundle_dir).resolve()), "policy": "causal_endpoint_v1",
               "holdout": holdout, "model_source": "local_bundle", "data_source": "synthetic",
               "request": {k: v for k, v in request.items() if k not in {"metadata", "manifest"}},
               "groups": {}}
    for name, (predicted, coverage, evaluation) in results.items():
        predicted.to_csv(run_dir / f"{name}_predictions.csv", index=False)
        coverage.to_csv(run_dir / f"{name}_coverage.csv", index=False)
        summary["groups"][name] = {"requested": len(coverage),
            "predicted": int((coverage["status"] == "predicted").sum()),
            "insufficient_context": int((coverage["status"] != "predicted").sum()),
            "evaluation": evaluation}
    (run_dir / "inference_report.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return {"run_dir": run_dir, "summary": summary}
