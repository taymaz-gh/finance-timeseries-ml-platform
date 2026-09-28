"""Generic chronological splitting utilities for time-series data."""

from __future__ import annotations

from math import floor


def temporal_split(
    df,
    time_col: str,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
):
    """
    Split a Spark DataFrame chronologically using distinct timestamps.

    The split is based on the ordered unique values of `time_col`, not on
    row counts. All rows sharing the same timestamp are therefore assigned
    to the same partition.

    Args:
        df:
            Input Spark DataFrame.
        time_col:
            Name of the timestamp/date column.
        train_fraction:
            Fraction of distinct timestamps assigned to training.
        validation_fraction:
            Fraction of distinct timestamps assigned to validation.

    Returns:
        Tuple containing:
            - training DataFrame
            - validation DataFrame
            - test DataFrame
            - training end timestamp
            - validation end timestamp

    Raises:
        ValueError:
            If the time column does not exist, split fractions are invalid,
            or there are too few distinct timestamps to create all splits.
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' does not exist in the DataFrame.")

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1.")

    timestamps = [
        row[time_col]
        for row in (
            df.select(time_col)
            .where(df[time_col].isNotNull())
            .distinct()
            .orderBy(time_col)
            .collect()
        )
    ]

    n_timestamps = len(timestamps)

    if n_timestamps < 3:
        raise ValueError("At least 3 distinct non-null timestamps are required.")

    n_train = floor(n_timestamps * train_fraction)

    n_validation = floor(n_timestamps * validation_fraction)

    n_test = n_timestamps - n_train - n_validation

    if n_train < 1 or n_validation < 1 or n_test < 1:
        raise ValueError(
            "Split fractions produce an empty train, validation, or test split."
        )

    train_end = timestamps[n_train - 1]

    validation_end = timestamps[n_train + n_validation - 1]

    train_df = df.filter(df[time_col] <= train_end)

    validation_df = df.filter(
        (df[time_col] > train_end) & (df[time_col] <= validation_end)
    )

    test_df = df.filter(df[time_col] > validation_end)

    return (
        train_df,
        validation_df,
        test_df,
        train_end,
        validation_end,
    )
