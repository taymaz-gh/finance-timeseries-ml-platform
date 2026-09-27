"""Running Gold feature engineering and persisting the Gold table."""

from __future__ import annotations

from finance_ml.features.build_gold import build_gold_features


def run_gold_pipeline(
    spark,
    source_table: str = "finance_ml.silver.account_events_clean",
    target_table: str = "finance_ml.gold.account_features",
):
    """
    Loading Silver data, building Gold features, and persisting the result.

    Args:
        spark:
            Active Spark session.
        source_table:
            Fully qualified Silver source table.
        target_table:
            Fully qualified Gold destination table.

    Returns:
        Persisted Gold Spark DataFrame.
    """
    silver_df = spark.table(source_table)

    gold_df = build_gold_features(silver_df)

    (gold_df.write.mode("overwrite").saveAsTable(target_table))

    return gold_df
