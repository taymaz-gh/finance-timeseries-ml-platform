"""Preparing reproducible, label-free temporal and new-account holdouts.

The original historical tables and registered model remain read-only.
Generated labels are returned separately and never enter Bronze/Silver/Gold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from finance_ml.data.generate_synthetic import generate_holdout_timeseries
from finance_ml.data.inject_quality_issues import inject_data_quality_issues
from finance_ml.lakehouse.build_gold import build_gold_features
from finance_ml.lakehouse.build_silver import build_silver_dataframe


LABEL_COLUMNS = ("risk_state", "risk_state_name")
DEFAULT_TABLES = {
    "temporal_bronze": "finance_ml.bronze.account_events_temporal_holdout",
    "temporal_silver": "finance_ml.silver.account_events_temporal_holdout_clean",
    "temporal_gold": "finance_ml.gold.account_features_temporal_holdout",
    "temporal_truth": "finance_ml.gold.account_labels_temporal_holdout",
    "account_bronze": "finance_ml.bronze.account_events_account_holdout",
    "account_silver": "finance_ml.silver.account_events_account_holdout_clean",
    "account_gold": "finance_ml.gold.account_features_account_holdout",
    "account_truth": "finance_ml.gold.account_labels_account_holdout",
}
PROTECTED_TABLES = {
    "finance_ml.bronze.account_events_raw",
    "finance_ml.silver.account_events_clean",
    "finance_ml.gold.account_features",
}
_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class HoldoutGroup:
    """Keeping a holdout group's layers and uncoupled ground truth together."""

    bronze: object
    silver: object
    gold: object
    context: object
    truth: object
    expected_rows: int


def _validate_tables(tables: dict[str, str], history_silver_table: str) -> None:
    if set(tables) != set(DEFAULT_TABLES):
        raise ValueError(f"Expected table keys: {sorted(DEFAULT_TABLES)}")
    names = [name.lower() for name in tables.values()]
    if len(set(names)) != len(names):
        raise ValueError("All holdout table names must be distinct")
    for key, name in tables.items():
        if not isinstance(name, str) or not _TABLE_NAME.fullmatch(name):
            raise ValueError(f"Invalid catalog.schema.table for {key}: {name!r}")
        if name.lower() in PROTECTED_TABLES or name.lower() == history_silver_table.lower():
            raise ValueError(f"Refusing to write original historical table: {name}")
        if "holdout" not in name.lower().split(".")[-1]:
            raise ValueError(f"Holdout destination must include 'holdout': {name}")


def _spark_unlabeled(spark, clean: pd.DataFrame, *, corrupt: bool, seed: int):
    from pyspark.sql import functions as F

    # Removing simulator targets BEFORE Bronze creation or quality injection.
    predictors = clean.drop(columns=list(LABEL_COLUMNS)).copy()
    if corrupt:
        predictors = inject_data_quality_issues(predictors, random_seed=seed)
    # Converting timestamps into native Python datetime objects.
    predictors["date"] = pd.to_datetime(
        predictors["date"]
    ).map(lambda timestamp: timestamp.to_pydatetime())
    return spark.createDataFrame(predictors).withColumn("date", F.to_timestamp("date"))


def _truth_frame(spark, labeled: pd.DataFrame):
    from pyspark.sql import functions as F

    truth = labeled[["account_id", "date", *LABEL_COLUMNS]].copy()
    # Converting ground-truth timestamps consistently.
    truth["date"] = pd.to_datetime(
        truth["date"]
    ).map(lambda timestamp: timestamp.to_pydatetime())
    return spark.createDataFrame(truth).withColumn("date", F.to_timestamp("date"))


def _assert_complete_keys(frame, expected: int, label: str) -> None:
    from pyspark.sql import functions as F

    # This is an opt-in Spark action that validates coverage before table writes.
    count = frame.count()
    if count != expected:
        raise ValueError(f"{label}: expected {expected:,} rows; found {count:,}")
    if frame.filter(F.col("account_id").isNull() | F.col("date").isNull()).limit(1).count():
        raise ValueError(f"{label}: null primary key")
    if frame.groupBy("account_id", "date").count().filter(F.col("count") != 1).limit(1).count():
        raise ValueError(f"{label}: duplicate account/date")


def prepare_holdout_data(
    spark,
    *,
    n_existing_accounts: int = 2000,
    historical_days: int = 60,
    n_future_days: int = 60,
    n_new_accounts: int = 1000,
    start_date: str = "2026-01-01",
    random_seed: int = 42,
    corruption_seed_temporal: int = 124,
    corruption_seed_account: int = 125,
    inject_quality_issues: bool = True,
    history_silver_table: str = "finance_ml.silver.account_events_clean",
    history_context_days: int = 6,
    validate: bool = True,
    persist: bool = False,
    tables: dict[str, str] | None = None,
) -> dict:
    """Preparing two independent holdout paths, optionally writing only safe tables.

    Temporal: existing accounts on new dates, with historical Silver joined
    BEFORE imputation and Gold feature engineering. New-account: full 120-day
    trajectories with no cross-account context. Labels are stored separately.

    Caveat: this is a batch/driver-oriented toy pipeline. Eight sequential
    overwrite writes are not atomic. Review destination tables before opting in.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    if not isinstance(history_context_days, int) or isinstance(history_context_days, bool) or history_context_days < 6:
        raise ValueError("history_context_days must be an integer >= 6 for the saved seven-step LSTM")
    targets = {**DEFAULT_TABLES, **(tables or {})}
    _validate_tables(targets, history_silver_table)

    simulated = generate_holdout_timeseries(
        n_existing_accounts=n_existing_accounts,
        historical_days=historical_days,
        holdout_days=n_future_days,
        n_new_accounts=n_new_accounts,
        start_date=start_date,
        random_seed=random_seed,
    )
    historical_pd = simulated["historical"]
    temporal_pd = simulated["temporal_holdout"]
    account_pd = simulated["account_holdout"]
    cutoff = pd.Timestamp(historical_pd["date"].max()).to_pydatetime()
    first_future = pd.Timestamp(temporal_pd["date"].min()).to_pydatetime()
    if first_future.date() <= cutoff.date():
        raise ValueError("Temporal holdout overlaps historical dates")

    # Historical data are READ ONLY and used solely for causal feature context.
    history = spark.table(history_silver_table).drop(*LABEL_COLUMNS)
    if validate:
        if history.select("account_id").distinct().count() != n_existing_accounts:
            raise ValueError("Historical Silver account count differs from replay")
        if history.agg(F.max("date")).first()[0].date() != cutoff.date():
            raise ValueError("Historical Silver cutoff differs from replay")

    temporal_bronze = _spark_unlabeled(
        spark, temporal_pd, corrupt=inject_quality_issues, seed=corruption_seed_temporal,
    )
    account_bronze = _spark_unlabeled(
        spark, account_pd, corrupt=inject_quality_issues, seed=corruption_seed_account,
    )

    # Cleaning history and future together allows causal forward fill across cutoff.
    temporal_raw_with_history = history.unionByName(temporal_bronze, allowMissingColumns=True)
    temporal_silver_all = build_silver_dataframe(temporal_raw_with_history, require_labels=False)
    temporal_silver = temporal_silver_all.filter(F.col("date") > F.lit(cutoff))
    temporal_gold_all = build_gold_features(temporal_silver_all)
    temporal_gold = temporal_gold_all.filter(F.col("date") > F.lit(cutoff))

    historical_context = (
        temporal_gold_all.filter(F.col("date") <= F.lit(cutoff))
        .withColumn("_rank", F.row_number().over(
            Window.partitionBy("account_id").orderBy(F.col("date").desc())
        ))
        .filter(F.col("_rank") <= history_context_days)
        .drop("_rank")
    )
    temporal_context = historical_context.unionByName(temporal_gold)

    account_silver = build_silver_dataframe(account_bronze, require_labels=False)
    account_gold = build_gold_features(account_silver)
    # All unseen-account records belong to this holdout; no historical table union.
    account_context = account_gold

    groups = {
        "temporal": HoldoutGroup(
            bronze=temporal_bronze, silver=temporal_silver,
            gold=temporal_gold, context=temporal_context,
            truth=_truth_frame(spark, temporal_pd), expected_rows=len(temporal_pd),
        ),
        "account": HoldoutGroup(
            bronze=account_bronze, silver=account_silver,
            gold=account_gold, context=account_context,
            truth=_truth_frame(spark, account_pd), expected_rows=len(account_pd),
        ),
    }
    if validate:
        for name, group in groups.items():
            for layer in ("silver", "gold", "truth"):
                _assert_complete_keys(getattr(group, layer), group.expected_rows, f"{name}.{layer}")
            for layer in ("bronze", "silver", "gold", "context"):
                if any(col in getattr(group, layer).columns for col in LABEL_COLUMNS):
                    raise ValueError(f"Label leakage detected in {name}.{layer}")
        _assert_complete_keys(temporal_context, len(temporal_pd) + n_existing_accounts * history_context_days, "temporal.context")

    if persist:
        # Explicit opt-in only. Do not use this option for first-time inspection.
        for name, group in groups.items():
            for layer in ("bronze", "silver", "gold", "truth"):
                getattr(group, layer).write.mode("overwrite").saveAsTable(targets[f"{name}_{layer}"])

    return {
        **groups,
        "history_end": cutoff,
        "future_start": first_future,
        "tables": targets,
        "persisted": persist,
        "historical_replay": historical_pd,  # clean, labeled, NOT persisted
    }
