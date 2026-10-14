"""Run account/date-selected synthetic inference without modifying source tables.

The CLI selects accounts and an inclusive date interval. Cohort labels are
inferred from a checkpoint-bound, explicitly declared development profile.
Model inputs are strictly label-free; any simulated future extension is opt-in.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import pandas as pd

from finance_ml.data.synthetic_provenance import (
    SyntheticProvenance, load_profile, normalize_account_id, parse_date,
)
from finance_ml.models.local_bundle import inspect_local_bundle, load_local_bundle

MAX_REQUESTED_ROWS = 20_000
MAX_ACCOUNTS = 25
MAX_EXTENSION_DAYS = 60
LABELS = ("risk_state", "risk_state_name")


def cohort_for(profile: SyntheticProvenance, account_id: str, day: date) -> str:
    """Label the two-dimensional account/date evaluation cell or extension."""
    if day > profile.observed_through:
        return "simulated_extension"
    known = int(account_id.split("_")[1]) <= profile.development_accounts
    if known:
        return "development_period" if day <= profile.development_end else "date_holdout"
    return "account_holdout" if day <= profile.development_end else "joint_holdout"


def validate_account_request(*, bundle_dir: str | Path,
                             account_ids: Sequence[str], start_date: str,
                             end_date: str, profile_path: str | Path | None = None,
                             simulate_future: bool = False,
                             smoke: bool = False,
                             output_dir: str | Path = "artifacts/inference") -> dict[str, Any]:
    """Perform metadata, provenance, time-range and resource preflight."""
    if type(simulate_future) is not bool or type(smoke) is not bool:
        raise ValueError("Execution flags must be booleans")
    metadata, manifest = inspect_local_bundle(bundle_dir)
    profile = load_profile(bundle_dir, profile_path, manifest=manifest)
    if metadata["n_classes"] != 4:
        raise ValueError("Synthetic risk-classification inference requires exactly four classes")
    if profile.historical_days < metadata["sequence_length"]:
        raise ValueError("Declared development history is shorter than the saved model sequence")
    if not account_ids:
        raise ValueError("Supply at least one --account-id")
    ids = tuple(normalize_account_id(x) for x in account_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Account IDs must not be duplicated")
    if len(ids) > MAX_ACCOUNTS:
        raise ValueError(f"At most {MAX_ACCOUNTS} account IDs per local inference run")
    for value in ids:
        number = int(value.split("_")[1])
        if number > profile.development_accounts + profile.max_new_accounts:
            raise ValueError(f"{value} is outside the declared synthetic account population")
    start, end = parse_date(start_date, "start_date"), parse_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must not exceed end_date")
    if start < profile.start:
        raise ValueError(f"Requested dates precede the earliest available date {profile.start}")
    if simulate_future and end <= profile.observed_through:
        raise ValueError("--simulate-future is only valid for dates beyond available observations")
    if end > profile.observed_through:
        if not simulate_future:
            raise ValueError(
                f"Requested end {end} is beyond available observations "
                f"({profile.observed_through}). Explicitly opt in with --simulate-future "
                "to generate and classify simulated observations."
            )
        if (end - profile.observed_through).days > MAX_EXTENSION_DAYS:
            raise ValueError(f"Future simulation is limited to {MAX_EXTENSION_DAYS} days after available data")
    n_dates = (end - start).days + 1
    if n_dates * len(ids) > MAX_REQUESTED_ROWS:
        raise ValueError(f"Requested observations exceed {MAX_REQUESTED_ROWS:,}-row limit")
    if smoke and (len(ids) > 2 or n_dates > 7):
        raise ValueError("Smoke mode accepts at most 2 account IDs and 7 days; it never silently changes your request")
    root = Path(output_dir)
    if not str(output_dir).strip() or (root.exists() and not root.is_dir()):
        raise ValueError("output_dir must name a directory")
    # Also bounding contextual feature rows materialized on the local driver.
    context_days = (end - profile.start).days + 1
    if len(ids) * context_days > MAX_REQUESTED_ROWS:
        raise ValueError("Selected account histories exceed local inference row limit")
    warnings: list[str] = [
        "Account population, seed and simulator provenance are user-declared; "
        "the legacy bundle cannot independently verify these values."
    ]
    if any(int(x.split("_")[1]) <= profile.development_accounts for x in ids) and start <= profile.development_end:
        warnings.append("Known-account development dates are not independent holdout evaluation.")
    if end > profile.observed_through:
        warnings.append("Future financial FEATURES are simulated, not observed; these are not direct risk forecasts.")
    return {
        "profile": profile, "metadata": metadata, "manifest": manifest,
        "account_ids": ids, "start": start, "end": end,
        "simulate_future": simulate_future, "smoke": smoke,
        "output_dir": str(output_dir), "requested_rows": n_dates * len(ids),
        "warnings": warnings,
    }


def public_plan(request: dict) -> dict:
    """Return a JSON-safe plan with no Keras or Spark objects."""
    p = request["profile"]
    return {
        "account_ids": list(request["account_ids"]),
        "start_date": request["start"].isoformat(),
        "end_date": request["end"].isoformat(),
        "requested_rows": request["requested_rows"],
        "development_population": f"ACC_00001..ACC_{p.development_accounts:05d}",
        "development_end": p.development_end.isoformat(),
        "available_through": p.observed_through.isoformat(),
        "replay_start_date": p.start.isoformat(),
        "historical_days": p.historical_days,
        "new_account_anchor_days": p.anchor_days,
        "simulator_seed": p.seed,
        "simulator_version": "stateful_v1",
        "model_sha256": request["manifest"]["model"]["sha256"],
        "model_features": len(request["metadata"]["final_feature_columns"]),
        "sequence_length": request["metadata"]["sequence_length"],
        "prediction_policy": "causal_endpoint_v1",
        "data_source": "synthetic_replay",
        "model_source": "local_bundle",
        "simulated_future": request["simulate_future"],
        "smoke": request["smoke"],
        "warnings": request["warnings"],
    }


def run_account_inference(*, bundle_dir: str | Path,
                          account_ids: Sequence[str], start_date: str,
                          end_date: str, profile_path: str | Path | None = None,
                          simulate_future: bool = False, smoke: bool = False,
                          output_dir: str | Path = "artifacts/inference",
                          spark: Any = None) -> dict[str, Any]:
    """Score only requested account/dates, preserving causal feature context."""
    request = validate_account_request(
        bundle_dir=bundle_dir, account_ids=account_ids,
        start_date=start_date, end_date=end_date, profile_path=profile_path,
        simulate_future=simulate_future, smoke=smoke, output_dir=output_dir,
    )
    from finance_ml.data.synthetic_replay import replay_selected_accounts
    from finance_ml.pipelines.inference_runner import (
        _build_gold, apply_saved_preprocessing, causal_endpoint_predictions, _evaluate,
    )
    from finance_ml.pipelines.training_runner import _create_local_spark_session

    # Loading and validating the checkpoint before starting a JVM.
    bundle = load_local_bundle(bundle_dir)
    p = request["profile"]
    owns_spark = False
    if spark is None:
        spark, owns_spark = _create_local_spark_session()
    original_error: BaseException | None = None
    try:
        features, truth = replay_selected_accounts(p, request["account_ids"], request["end"])
        if set(features.columns) & set(LABELS):
            raise ValueError("Target leakage in simulator observations")
        gold = _build_gold(spark, features)
        prepared = apply_saved_preprocessing(gold, bundle.metadata)
        prepared_pd = prepared.toPandas()
        dates = pd.date_range(request["start"], request["end"], freq="D")
        desired = pd.DataFrame(
            [(account, day) for account in request["account_ids"] for day in dates],
            columns=["account_id", "date"],
        )
        # Verifying every requested date exists as a feature observation.
        available = features[["account_id", "date"]]
        matched = desired.merge(available, on=["account_id", "date"],
                                how="left", indicator=True, validate="one_to_one")
        if not matched["_merge"].eq("both").all():
            raise ValueError("Requested financial feature observations are unavailable")
        predictions, coverage = causal_endpoint_predictions(
            bundle.model, prepared_pd, desired,
            feature_columns=bundle.metadata["final_feature_columns"],
            sequence_length=bundle.metadata["sequence_length"],
            n_classes=bundle.metadata["n_classes"],
        )
        for frame in (predictions, coverage):
            frame["cohort"] = [cohort_for(p, acc, day.date())
                               for acc, day in zip(frame["account_id"], frame["date"]) ]
            frame["account_population"] = [
                "known" if int(x.split("_")[1]) <= p.development_accounts else "new"
                for x in frame["account_id"]
            ]
            frame["observation_origin"] = [
                "simulated_extension" if day.date() > p.observed_through else "synthetic_replay"
                for day in frame["date"]
            ]
        if len(coverage) != request["requested_rows"] or \
           int(coverage["status"].eq("predicted").sum()) != len(predictions):
            raise AssertionError("Incomplete or duplicated prediction coverage")
        # Strictly separating labels until this post-scoring evaluation stage.
        evaluations: dict[str, dict] = {}
        for name, part in predictions.groupby("cohort"):
            if name == "development_period":
                continue
            # Ground-truth states here are synthetic simulator diagnostics only.
            selected_truth = truth.merge(part[["account_id", "date"]],
                                         on=["account_id", "date"], validate="one_to_one")
            if len(selected_truth) != len(part):
                raise ValueError("Scored observations lack one-to-one simulator truth")
            evaluations[name] = _evaluate(part, selected_truth, bundle.metadata["n_classes"])
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

    directory = Path(output_dir) / (
        f"infer_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:10]}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(directory / "predictions.csv", index=False)
    coverage.to_csv(directory / "coverage.csv", index=False)
    summary = {
        "request": public_plan(request),
        "bundle": str(Path(bundle_dir).resolve()),
        "profile_provenance": "user_declared_unverified",
        "predictions": len(predictions),
        "requested": len(coverage),
        "insufficient_context": int(coverage["status"].ne("predicted").sum()),
        "cohort_coverage": {
            name: {
                "requested": int(len(rows)),
                "predicted": int(rows["status"].eq("predicted").sum()),
                "insufficient_context": int(rows["status"].ne("predicted").sum()),
            }
            for name, rows in coverage.groupby("cohort")
        },
        "evaluations": evaluations,
        "metric_policy": "causal_endpoint_v1; simulator ground truth only; development dates excluded",
    }
    (directory / "inference_report.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return {"run_dir": directory, "summary": summary}
