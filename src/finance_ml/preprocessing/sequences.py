"""Sequence-construction utilities for time-series classification."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def build_sequences(
    df,
    feature_columns: Iterable[str],
    target_col: str,
    group_col: str,
    time_col: str,
    sequence_length: int,
    stride: int = 1,
):
    """
    Build fixed-length sequence-to-sequence samples from ordered panel data.

    Each returned sample contains:

        X: shape (sequence_length, n_features)
        y: shape (sequence_length,)

    Sequences are created independently inside each `group_col` value and
    therefore never cross entity boundaries.

    Args:
        df:
            Spark DataFrame containing model-ready features and targets.
        feature_columns:
            Numerical feature columns used as model inputs.
        target_col:
            Target column containing one label per timestep.
        group_col:
            Entity identifier used to separate independent time series.
        time_col:
            Timestamp/date column used for chronological ordering.
        sequence_length:
            Number of timesteps per sequence.
        stride:
            Number of rows to advance between consecutive windows.

    Returns:
        Tuple containing:
            - X as a NumPy array with shape
              (n_sequences, sequence_length, n_features)
            - y as a NumPy array with shape
              (n_sequences, sequence_length)
            - metadata list containing group and time boundaries

    Raises:
        ValueError:
            If required columns are missing or sequence parameters are invalid.
    """
    feature_columns = list(feature_columns)

    required_columns = {
        *feature_columns,
        target_col,
        group_col,
        time_col,
    }

    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")

    if stride < 1:
        raise ValueError("stride must be at least 1.")

    selected_df = df.select(
        group_col,
        time_col,
        *feature_columns,
        target_col,
    ).orderBy(
        group_col,
        time_col,
    )

    rows = selected_df.collect()

    grouped_rows: dict[object, list] = {}

    for row in rows:
        group_value = row[group_col]

        grouped_rows.setdefault(
            group_value,
            [],
        ).append(row)

    X_sequences = []
    y_sequences = []
    metadata = []

    for group_value, group_rows in grouped_rows.items():
        n_rows = len(group_rows)

        if n_rows < sequence_length:
            continue

        for start_index in range(
            0,
            n_rows - sequence_length + 1,
            stride,
        ):
            end_index = start_index + sequence_length

            window_rows = group_rows[start_index:end_index]

            X_window = [
                [float(row[column]) for column in feature_columns]
                for row in window_rows
            ]

            y_window = [int(row[target_col]) for row in window_rows]

            X_sequences.append(X_window)

            y_sequences.append(y_window)

            metadata.append(
                {
                    "group": group_value,
                    "start_time": window_rows[0][time_col],
                    "end_time": window_rows[-1][time_col],
                    "times": [row[time_col] for row in window_rows],
                }
            )

    X = np.asarray(
        X_sequences,
        dtype=np.float32,
    )

    y = np.asarray(
        y_sequences,
        dtype=np.int64,
    )

    return X, y, metadata
