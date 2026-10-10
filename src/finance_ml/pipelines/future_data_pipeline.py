"""Orchestrating separate Bronze/Silver/Gold future inference datasets."""

from __future__ import annotations

from finance_ml.data.generate_synthetic import generate_future_observations
from finance_ml.lakehouse.build_silver import build_silver_dataframe
from finance_ml.lakehouse.build_gold import build_gold_features

FUTURE_TABLES = {
    "bronze": "finance_ml.bronze.account_events_future",
    "silver": "finance_ml.silver.account_events_future_clean",
    "gold": "finance_ml.gold.account_features_future",
    "context": "finance_ml.gold.account_features_inference_context",
}


def prepare_future_data(
    spark,
    *,
    history_silver_table="finance_ml.silver.account_events_clean",
    n_future_days=14,
    seed=84,
    persist=False,
    tables=None,
):
    """Continue accounts approximately; preserve original tables unchanged.

    Returning Bronze, future Silver, future Gold and scoring context Spark
    frames. Writing only to separate future-specific tables if persist=True.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    targets = {**FUTURE_TABLES, **(tables or {})}
    if n_future_days < 1:
        raise ValueError("n_future_days must be positive")
    if len(set(targets.values())) != len(targets):
        raise ValueError("Each future output table must be unique")
    if any(name == history_silver_table for name in targets.values()):
        raise ValueError("Refusing to overwrite the training Silver table")
    if any(name in {"finance_ml.gold.account_features", "finance_ml.bronze.account_events_raw"}
           for name in targets.values()):
        raise ValueError("Refusing to overwrite original training tables")

    history = spark.table(history_silver_table)
    maximum_date = history.agg(F.max("date").alias("latest")).first()["latest"]
    if maximum_date is None:
        raise ValueError("The historical Silver table is empty")
    latest = (
        history.withColumn(
            "_rank", F.row_number().over(
                Window.partitionBy("account_id").orderBy(F.col("date").desc())
            ))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
        .toPandas()
    )
    # Generating future observations from the most recent observed states.
    future_pd = generate_future_observations(latest, n_days=n_future_days, random_seed=seed)
    if future_pd.empty:
        raise ValueError("No future account rows generated")
    if future_pd["date"].min().date() <= maximum_date.date():
        raise ValueError("Future observations overlap historical dates")
    future_pd["date"] = future_pd["date"].dt.to_pydatetime()
    bronze = spark.createDataFrame(future_pd)
    silver = build_silver_dataframe(bronze, require_labels=False)

    # Joining historical Silver records before calculating lags and rolling windows.
    combined_silver = history.unionByName(silver, allowMissingColumns=True)
    engineered = build_gold_features(combined_silver)
    future_gold = engineered.filter(F.col("date") > F.lit(maximum_date))

    # Retaining six historical Gold timesteps per account for seven-day windows.
    # A history-only ranking avoids dropping older context as future rows arrive.
    old_context = (
        engineered.filter(F.col("date") <= F.lit(maximum_date))
        .withColumn("_rank", F.row_number().over(
            Window.partitionBy("account_id").orderBy(F.col("date").desc())
        ))
        .filter(F.col("_rank") <= 6)
        .drop("_rank")
    )
    scoring_context = old_context.unionByName(future_gold, allowMissingColumns=True)

    if persist:
        for name, frame in (
            ("bronze", bronze), ("silver", silver),
            ("gold", future_gold), ("context", scoring_context),
        ):
            frame.write.mode("overwrite").saveAsTable(targets[name])

    return {
        "bronze": bronze,
        "silver": silver,
        "gold": future_gold,
        "context": scoring_context,
        "future_start": future_pd["date"].min(),
        "history_end": maximum_date,
        "tables": targets,
    }
