"""Leakage-safe categorical encoding (dummy variable / one-hot encoding) utilities for model datasets."""

from __future__ import annotations

from collections.abc import Iterable


def fit_category_levels(
    train_df,
    categorical_columns: Iterable[str],
) -> dict[str, list[str]]:
    """
    Learn categorical levels from training data only.

    Args:
        train_df:
            Training Spark DataFrame.
        categorical_columns:
            Categorical feature columns to encode.

    Returns:
        Dictionary mapping each categorical column to its sorted
        training-derived category levels.

    Raises:
        ValueError:
            If a requested categorical column is missing or contains
            no non-null category values.
    """
    from pyspark.sql import functions as F

    category_levels: dict[str, list[str]] = {}

    for column in categorical_columns:
        if column not in train_df.columns:
            raise ValueError(
                f"Categorical column '{column}' does not exist in the DataFrame."
            )

        levels = [
            row[column]
            for row in (
                train_df.select(column)
                .where(F.col(column).isNotNull())
                .distinct()
                .orderBy(column)
                .collect()
            )
        ]

        if not levels:
            raise ValueError(
                f"Categorical column '{column}' contains no non-null values."
            )

        category_levels[column] = levels

    return category_levels


def apply_one_hot_encoding(
    df,
    category_levels: dict[str, list[str]],
):
    """
    Apply one-hot encoding using training-derived category levels.

    Args:
        df:
            Spark DataFrame to transform.
        category_levels:
            Training-derived category levels for each categorical column.

    Returns:
        Spark DataFrame containing one-hot encoded indicator columns.

    Notes:
        Categories not observed during training produce zeros across all
        one-hot columns for that categorical feature.
    """
    from pyspark.sql import functions as F

    result_df = df

    for column, levels in category_levels.items():
        if column not in result_df.columns:
            raise ValueError(
                f"Categorical column '{column}' does not exist in the DataFrame."
            )

        for level in levels:
            safe_level = str(level).strip().lower().replace(" ", "_").replace("-", "_")

            encoded_column = f"{column}_{safe_level}"

            result_df = result_df.withColumn(
                encoded_column,
                (F.col(column) == F.lit(level)).cast("int"),
            )

    return result_df
