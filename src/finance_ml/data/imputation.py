"""Leakage-safe numeric imputation utilities for model datasets."""

from __future__ import annotations


def fit_numeric_medians(
    train_df,
    numeric_columns: list[str],
) -> dict[str, float]:
    """
    Fit median imputation values using training data only.

    Args:
        train_df:
            Training Spark DataFrame.
        numeric_columns:
            Numeric feature columns to inspect.

    Returns:
        Dictionary mapping columns with missing values to their
        training-derived median.
    """
    from pyspark.sql import functions as F

    missing_columns = []

    for column in numeric_columns:
        has_missing = train_df.filter(F.col(column).isNull()).limit(1).count() > 0

        if has_missing:
            missing_columns.append(column)

    medians = {}

    for column in missing_columns:
        median_value = train_df.select(
            F.percentile_approx(
                column,
                0.5,
            ).alias("median")
        ).first()["median"]

        if median_value is None:
            raise ValueError(
                f"Cannot compute median for '{column}' because "
                "the training column contains no non-null values."
            )

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
