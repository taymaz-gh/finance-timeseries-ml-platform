"""Utilities for defining model-ready feature and target columns."""

from __future__ import annotations


def get_model_columns(
    df,
    target_col: str = "risk_state",
    target_name_col: str = "risk_state_name",
    id_columns: tuple[str, ...] = ("account_id", "date"),
    categorical_columns: tuple[str, ...] = ("account_type",),
):
    """
    Identify model feature, target, identifier, and categorical columns.

    Args:
        df:
            Input Spark DataFrame.
        target_col:
            Numeric target column used for supervised learning.
        target_name_col:
            Human-readable target label column.
        id_columns:
            Columns retained for traceability but excluded from model inputs.
        categorical_columns:
            Feature columns requiring explicit encoding before modeling.

    Returns:
        Dictionary containing:
            - feature_columns
            - numeric_feature_columns
            - categorical_feature_columns
            - target_column
            - target_name_column
            - id_columns

    Raises:
        ValueError:
            If any required column is missing from the DataFrame.
    """
    required_columns = {
        target_col,
        target_name_col,
        *id_columns,
        *categorical_columns,
    }

    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    excluded_columns = {
        target_col,
        target_name_col,
        *id_columns,
    }

    feature_columns = [
        column for column in df.columns if column not in excluded_columns
    ]

    categorical_feature_columns = [
        column for column in categorical_columns if column in feature_columns
    ]

    numeric_feature_columns = [
        column
        for column in feature_columns
        if column not in categorical_feature_columns
    ]

    return {
        "feature_columns": feature_columns,
        "numeric_feature_columns": numeric_feature_columns,
        "categorical_feature_columns": categorical_feature_columns,
        "target_column": target_col,
        "target_name_column": target_name_col,
        "id_columns": list(id_columns),
    }
