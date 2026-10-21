"""Orchestrating model inference on uploaded financial observations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from finance_ml.api.model_service import ModelService
from finance_ml.api.upload_pipeline import prepare_uploaded_gold
from finance_ml.pipelines.inference_runner import (
    apply_saved_preprocessing,
    causal_endpoint_predictions,
)


def _serialize_dated_records(frame: pd.DataFrame) -> list[dict]:
    """Converting prediction records into JSON-compatible values."""

    result = frame.copy()

    if not result.empty:
        result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")

    return result.to_dict(orient="records")


def predict_uploaded_accounts(
    frame: pd.DataFrame,
    *,
    spark: Any,
    service: ModelService,
) -> dict[str, Any]:
    """Predicting risk states using uploaded raw account-day data."""

    # Reusing the verified saved model and preprocessing metadata.
    bundle = service.bundle
    metadata = bundle.metadata
    sequence_length = metadata["sequence_length"]

    # Preparing Gold observations and assessing account context.
    gold, account_context = prepare_uploaded_gold(
        frame,
        spark=spark,
        sequence_length=sequence_length,
    )

    # Applying training-fitted transformations without refitting.
    transformed = apply_saved_preprocessing(
        gold,
        metadata,
    )

    # Materializing the bounded model-ready observations.
    model_input = transformed.toPandas()

    # Requesting predictions for all available account/date endpoints.
    requested = model_input[["account_id", "date"]].copy()

    # Reusing our established causal sequence-inference function.
    predictions, coverage = causal_endpoint_predictions(
        bundle.model,
        model_input,
        requested,
        feature_columns=metadata["final_feature_columns"],
        sequence_length=sequence_length,
        n_classes=metadata["n_classes"],
    )

    # Reporting accounts with insufficient sequence context.
    warnings = []

    for record in account_context.to_dict(orient="records"):

        if record["status"] == "insufficient_context":
            warnings.append(
                {
                    "account_id": record["account_id"],
                    "code": "insufficient_context",
                    "message": (
                        f"Account has {record['available_observations']} "
                        f"observations; at least {sequence_length} required."
                    ),
                }
            )

        if record["missing_calendar_days"] > 0:
            warnings.append(
                {
                    "account_id": record["account_id"],
                    "code": "calendar_gaps",
                    "message": (
                        f"Account has {record['missing_calendar_days']} "
                        "missing calendar days between observations."
                    ),
                }
            )

    # Summarizing account-level and date-level inference coverage.
    summary = {
        "accounts_received": len(account_context),
        "accounts_eligible": int((account_context["status"] == "eligible").sum()),
        "accounts_insufficient_context": int(
            (account_context["status"] == "insufficient_context").sum()
        ),
        "observations_after_deduplication": len(coverage),
        "predictions": len(predictions),
        "dates_insufficient_context": int(
            (coverage["status"] == "insufficient_context").sum()
        ),
    }

    return {
        "data_source": "uploaded_csv",
        "cohort_status": "unverified_external_upload",
        "sequence_length": sequence_length,
        "summary": summary,
        "warnings": warnings,
        "account_context": account_context.to_dict(orient="records"),
        "coverage": _serialize_dated_records(coverage),
        "predictions": _serialize_dated_records(predictions),
    }
