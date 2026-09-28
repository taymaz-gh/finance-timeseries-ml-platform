"""Utilities for constructing the final numerical model feature schema."""

from __future__ import annotations


def build_final_feature_columns(
    numeric_feature_columns: list[str],
    medians: dict[str, float],
    category_levels: dict[str, list[str]],
) -> list[str]:
    """
    Build the final numerical feature-column list used by the model.

    The final schema contains:

    - original numeric features
    - missing-indicator columns for numerics that were imputed
    - one-hot encoded categorical features

    Args:
        numeric_feature_columns:
            Original numerical model features.
        medians:
            Training-derived medians. Each key identifies a numeric feature
            for which a missing-indicator column was created.
        category_levels:
            Training-derived categorical levels used for one-hot encoding.

    Returns:
        Ordered list of final numerical feature-column names.
    """
    final_columns = list(numeric_feature_columns)

    missing_indicator_columns = [f"{column}_missing" for column in medians]

    final_columns.extend(missing_indicator_columns)

    for column, levels in category_levels.items():
        for level in levels:
            safe_level = str(level).strip().lower().replace(" ", "_").replace("-", "_")

            final_columns.append(f"{column}_{safe_level}")

    return final_columns
