"""Building cleaned Silver account-event records for training or inference."""

from __future__ import annotations


def build_silver_dataframe(bronze_df, *, require_labels: bool = True):
    """Clean Bronze data, optionally requiring supervised target labels.

    Preserving the original default for training; allowing missing target
    columns for unlabeled future inference. Never fitting target labels.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    silver_df = (
        bronze_df
        .withColumn(
            "cash_outflow",
            F.when(F.col("cash_outflow") < 0, F.lit(None))
            .otherwise(F.col("cash_outflow")),
        )
        .withColumn(
            "credit_utilization",
            F.when(F.col("credit_utilization") > 1, F.lit(None))
            .otherwise(F.col("credit_utilization")),
        )
    )

    label_columns = ("risk_state", "risk_state_name")
    if require_labels:
        absent = set(label_columns) - set(bronze_df.columns)
        if absent:
            raise ValueError(f"Labeled Silver data missing target columns: {sorted(absent)}")
        priority = (
            F.col("risk_state").isNull().cast("int")
            + F.col("risk_state_name").isNull().cast("int")
        )
    else:
        # Prioritizing complete predictors when label-free duplicate rows exist.
        priority = sum(
            (F.col(c).isNull().cast("int") for c in bronze_df.columns),
            F.lit(0),
        )

    dedup_window = (
        Window.partitionBy("account_id", "date")
        .orderBy(
            priority.asc(),
            *[F.col(column).asc_nulls_last() for column in bronze_df.columns],
        )
    )
    silver_df = (
        silver_df.withColumn("_row_number", F.row_number().over(dedup_window))
        .filter(F.col("_row_number") == 1)
        .drop("_row_number")
    )

    account_window = Window.partitionBy("account_id")
    silver_df = silver_df.withColumn(
        "account_type",
        F.coalesce(
            F.col("account_type"),
            F.first("account_type", ignorenulls=True).over(account_window),
            F.lit("unknown"),
        ),
    )
    time_window = (
        Window.partitionBy("account_id")
        .orderBy("date")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    for column in ["balance", "credit_utilization", "payment_ratio", "days_past_due"]:
        silver_df = silver_df.withColumn(
            column,
            F.coalesce(
                F.col(column),
                F.last(F.col(column), ignorenulls=True).over(time_window),
            ),
        )
    if require_labels:
        silver_df = silver_df.filter(
            F.col("risk_state").isNotNull()
            & F.col("risk_state_name").isNotNull()
        )
    return silver_df
