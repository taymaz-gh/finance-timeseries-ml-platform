"""Leakage-safe numeric imputation utilities for model datasets."""

from __future__ import annotations


def fit_numeric_medians(
    train_df,
    numeric_columns: list[str],
    *,
    indicator_columns: list[str] | None = None,
) -> dict[str, float]:
    """Fit training-only medians for observed or explicitly selected features.

    When indicator_columns is None, preserve the original policy:
    fitting medians only for columns with missing training values.

    When indicator_columns is supplied, fit medians for those columns
    regardless of whether missing values occur in this sample.
    """
    import math

    from pyspark.sql import functions as F

    if indicator_columns is None:
        # Discovering missingness from training observations only.
        selected_columns = [
            column
            for column in numeric_columns
            if train_df.filter(F.col(column).isNull()).limit(1).count() > 0
        ]
    else:
        # Maintaining a stable, explicitly declared indicator schema.
        selected_columns = list(indicator_columns)

        if len(selected_columns) != len(set(selected_columns)):
            raise ValueError("indicator_columns contains duplicate feature names")

        unknown = sorted(set(selected_columns) - set(numeric_columns))
        if unknown:
            raise ValueError(f"Unknown numeric indicator columns: {unknown}")

    medians: dict[str, float] = {}

    for column in selected_columns:
        median_value = train_df.select(
            F.percentile_approx(
                column,
                0.5,
            ).alias("median")
        ).first()["median"]

        if median_value is None or not math.isfinite(float(median_value)):
            raise ValueError(f"Cannot compute a valid training median for '{column}'")

        medians[column] = float(median_value)

    return medians


def apply_numeric_imputation(
    df,
    medians: dict[str, float],
):
    """
    Apply fitted medians and create missing-indicator features.

    Args:
        df:
            Spark DataFrame to transform.
        medians:
            Training-derived median values keyed by feature name.

    Returns:
        Spark DataFrame containing:
            - original numeric columns with missing values imputed
            - `<column>_missing` indicator columns
    """
    from pyspark.sql import functions as F

    result_df = df

    for column, median_value in medians.items():
        missing_indicator = f"{column}_missing"

        result_df = result_df.withColumn(
            missing_indicator,
            F.col(column).isNull().cast("int"),
        ).withColumn(
            column,
            F.coalesce(
                F.col(column),
                F.lit(median_value),
            ),
        )

    return result_df
