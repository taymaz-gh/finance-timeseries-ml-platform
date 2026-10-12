"""Running bounded, account-wise batches with one pinned registered model load."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from finance_ml.models.inference import predict_and_aggregate
from finance_ml.models.loading import load_model_bundle
from finance_ml.pipelines.inference_pipeline import (
    INFERENCE_CONFIG,
    build_unlabeled_sequences,
    transform_for_inference,
)


def run_batched_inference_pipeline(
    gold_df,
    *,
    config: dict[str, Any] | None = None,
    accounts_per_batch: int = 50,
    on_batch: Callable[[str, list[dict]], None] | None = None,
    registry_client=None,
    tracking_client=None,
    model_loader=None,
) -> dict[str, Any]:
    """Score whole-account batches without reloading the model each time.

    Args:
        gold_df: Spark Gold context, with all dates needed for each account.
        config: Same registered-model and preprocessing options as single-run inference.
        accounts_per_batch: Maximum distinct accounts processed per batch.
        on_batch: Optional callback (batch index string, prediction records). Called
            synchronously for each completed batch. When provided, prediction
            records are NOT retained in the returned result, bounding driver memory.
            A callback should persist/checkpoint its records durably if needed.

    Notes:
        This is batch, not streaming, inference. Each account must fit in driver
        memory individually. No table or model is written by this function.
        Do not split one account across batches: sequence boundaries must remain
        intact. Use boundary_width=0 for complete seven-step-window coverage.
    """
    if isinstance(accounts_per_batch, bool) or not isinstance(accounts_per_batch, int) or accounts_per_batch < 1:
        raise ValueError("accounts_per_batch must be a positive integer")
    if on_batch is not None and not callable(on_batch):
        raise TypeError("on_batch must be callable")
    options = {**INFERENCE_CONFIG, **(dict(config) if config else {})}
    group_col = options["group_col"]
    time_col = options["time_col"]
    if group_col not in gold_df.columns or time_col not in gold_df.columns:
        raise ValueError("Gold input must contain account and timestamp columns")

    bundle = load_model_bundle(
        options["model_name"], options["model_version"],
        registry_client=registry_client,
        tracking_client=tracking_client,
        model_loader=model_loader,
    )
    m = bundle.metadata
    sequence_length = int(m["sequence_length"])
    stride = int(m["stride"])
    boundary_width = options["boundary_width"]
    if boundary_width is None:
        boundary_width = int(m["boundary_width"])
    if (
        not isinstance(boundary_width, int) or isinstance(boundary_width, bool)
        or not 0 <= boundary_width < (sequence_length + 1) // 2
    ):
        raise ValueError("Invalid boundary_width")
    batch_size = options["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    # Collecting only distinct account IDs, not Gold observations or sequences.
    account_ids = [row[0] for row in gold_df.select(group_col).distinct().orderBy(group_col).collect()]
    if not account_ids:
        raise ValueError("No accounts available for inference")

    from pyspark.sql import functions as F

    retained = [] if on_batch is None else None
    n_sequences = n_predictions = n_batches = 0
    for start in range(0, len(account_ids), accounts_per_batch):
        batch_ids = account_ids[start:start + accounts_per_batch]
        scoped = gold_df.filter(F.col(group_col).isin(batch_ids))
        prepared = transform_for_inference(scoped, m, group_col=group_col, time_col=time_col)
        X, sequence_metadata = build_unlabeled_sequences(
            prepared, feature_columns=m["final_feature_columns"],
            group_col=group_col, time_col=time_col,
            sequence_length=sequence_length, stride=stride,
        )
        predictions = predict_and_aggregate(
            bundle.model, X, sequence_metadata,
            sequence_length=sequence_length,
            boundary_width=boundary_width, batch_size=batch_size,
        )
        n_sequences += len(X)
        n_predictions += len(predictions)
        n_batches += 1
        if on_batch is not None:
            on_batch(str(n_batches), predictions)
        else:
            retained.extend(predictions)

    return {
        "predictions": retained,
        "n_sequences": n_sequences,
        "n_predictions": n_predictions,
        "n_accounts": len(account_ids),
        "n_batches": n_batches,
        "model_name": bundle.model_name,
        "model_version": bundle.model_version,
        "metadata_sha256": bundle.metadata_sha256,
        "feature_columns": m["final_feature_columns"],
    }
