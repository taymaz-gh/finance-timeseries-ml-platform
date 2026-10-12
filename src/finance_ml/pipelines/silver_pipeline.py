"""Running Silver transformations and persisting the Silver table."""

from __future__ import annotations

from finance_ml.lakehouse.build_silver import build_silver_dataframe


def run_silver_pipeline(
    spark,
    source_table: str = "finance_ml.bronze.account_events_raw",
    target_table: str = "finance_ml.silver.account_events_clean",
    *,
    require_labels: bool = True,
):
    """Transform Bronze to Silver, preserving labeled training as default."""
    bronze_df = spark.table(source_table)

    if require_labels:
        silver_df = build_silver_dataframe(bronze_df)
    else:
        silver_df = build_silver_dataframe(
            bronze_df,
            require_labels=False,
        )

    silver_df.write.mode("overwrite").saveAsTable(target_table)
    return silver_df
