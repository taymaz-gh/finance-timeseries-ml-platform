"""Running the Silver transformation and persisting the Silver table."""

from __future__ import annotations

from finance_ml.lakehouse.build_silver import build_silver_dataframe


def run_silver_pipeline(
    spark,
    source_table: str = "finance_ml.bronze.account_events_raw",
    target_table: str = "finance_ml.silver.account_events_clean",
):
    """
    Loading Bronze data, building Silver data, and persisting the result.

    Args:
        spark:
            Active Spark session.
        source_table:
            Fully qualified Bronze source table.
        target_table:
            Fully qualified Silver destination table.

    Returns:
        Persisted Silver Spark DataFrame.
    """
    bronze_df = spark.table(source_table)

    silver_df = build_silver_dataframe(bronze_df)

    (silver_df.write.mode("overwrite").saveAsTable(target_table))

    return silver_df
