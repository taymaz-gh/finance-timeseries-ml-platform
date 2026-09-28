"""Building model-ready Gold features from cleaned Silver data."""

from __future__ import annotations


def build_gold_features(silver_df):
    """
    Engineering causal Gold features from a Silver Spark DataFrame.

    Args:
        silver_df:
            Cleaned Silver Spark DataFrame.

    Returns:
        Spark DataFrame containing original cleaned columns and engineered
        causal features.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    ordered_window = Window.partitionBy("account_id").orderBy("date")

    seconds_per_day = 24 * 60 * 60

    rolling_7d_window = (
        Window
        .partitionBy("account_id")
        .orderBy(
            F.col("date").cast("long")
        )
        .rangeBetween(
            -6 * seconds_per_day,
            0,
        )
    )

    gold_df = (
        silver_df
        # Same-day derived features.
        .withColumn(
            "net_cash_flow",
            F.col("cash_inflow") - F.col("cash_outflow"),
        )
        .withColumn(
            "payment_gap",
            F.col("amount_due") - F.col("amount_paid"),
        )
        # One-step lag features.
        .withColumn(
            "previous_balance",
            F.lag("balance", 1).over(ordered_window),
        )
        .withColumn(
            "previous_credit_utilization",
            F.lag(
                "credit_utilization",
                1,
            ).over(ordered_window),
        )
        .withColumn(
            "previous_payment_ratio",
            F.lag(
                "payment_ratio",
                1,
            ).over(ordered_window),
        )
        .withColumn(
            "previous_days_past_due",
            F.lag(
                "days_past_due",
                1,
            ).over(ordered_window),
        )
        # One-step change features.
        .withColumn(
            "balance_change",
            F.col("balance") - F.col("previous_balance"),
        )
        .withColumn(
            "credit_utilization_change",
            F.col("credit_utilization") - F.col("previous_credit_utilization"),
        )
        .withColumn(
            "payment_ratio_change",
            F.col("payment_ratio") - F.col("previous_payment_ratio"),
        )
        # Seven-day trailing rolling features.
        .withColumn(
            "rolling_7d_mean_balance",
            F.avg("balance").over(rolling_7d_window),
        )
        .withColumn(
            "rolling_7d_mean_cash_outflow",
            F.avg("cash_outflow").over(rolling_7d_window),
        )
        .withColumn(
            "rolling_7d_mean_payment_ratio",
            F.avg("payment_ratio").over(rolling_7d_window),
        )
        .withColumn(
            "rolling_7d_max_days_past_due",
            F.max("days_past_due").over(rolling_7d_window),
        )
    )

    return gold_df
