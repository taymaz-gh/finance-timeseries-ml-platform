"""Building the cleaned Silver account-event dataset."""

from __future__ import annotations


def build_silver_dataframe(bronze_df):
    """
    Cleaning Bronze account-event data into a Silver Spark DataFrame.

    Args:
        bronze_df:
            Input Spark DataFrame containing Bronze account-event data.

    Returns:
        Cleaned Spark DataFrame suitable for the Silver layer.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    silver_df = (
        bronze_df
        # Converting structurally invalid values to NULL.
        .withColumn(
            "cash_outflow",
            F.when(
                F.col("cash_outflow") < 0,
                F.lit(None),
            ).otherwise(F.col("cash_outflow")),
        ).withColumn(
            "credit_utilization",
            F.when(
                F.col("credit_utilization") > 1,
                F.lit(None),
            ).otherwise(F.col("credit_utilization")),
        )
    )

    # Removing duplicate account-date rows deterministically.
    dedup_window = Window.partitionBy(
        "account_id",
        "date",
    ).orderBy(*[F.col(column).asc_nulls_last() for column in bronze_df.columns])

    silver_df = (
        silver_df.withColumn(
            "_row_number",
            F.row_number().over(dedup_window),
        )
        .filter(F.col("_row_number") == 1)
        .drop("_row_number")
    )

    # Filling the static account type from another known row
    # belonging to the same account.
    account_window = Window.partitionBy("account_id")

    silver_df = silver_df.withColumn(
        "account_type",
        F.coalesce(
            F.col("account_type"),
            F.first(
                "account_type",
                ignorenulls=True,
            ).over(account_window),
            F.lit("unknown"),
        ),
    )

    # Using only earlier observations from the same account.
    time_window = (
        Window.partitionBy("account_id")
        .orderBy("date")
        .rowsBetween(
            Window.unboundedPreceding,
            -1,
        )
    )

    state_columns = [
        "balance",
        "credit_utilization",
        "payment_ratio",
        "days_past_due",
    ]

    for column in state_columns:
        previous_observed = F.last(
            F.col(column),
            ignorenulls=True,
        ).over(time_window)

        silver_df = silver_df.withColumn(
            column,
            F.coalesce(
                F.col(column),
                previous_observed,
            ),
        )

    # Removing rows with missing target labels.
    silver_df = silver_df.filter(
        F.col("risk_state").isNotNull() & F.col("risk_state_name").isNotNull()
    )

    return silver_df
