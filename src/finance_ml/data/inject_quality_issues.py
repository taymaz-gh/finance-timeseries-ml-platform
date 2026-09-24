"""Injecting controlled data-quality issues into synthetic financial data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_data_quality_issues(
    df: pd.DataFrame,
    missing_rate: float = 0.01,
    duplicate_rate: float = 0.005,
    invalid_rate: float = 0.002,
    random_seed: int = 123,
) -> pd.DataFrame:
    """
    Injecting controlled imperfections into clean synthetic financial data.

    Args:
        df:
            Clean synthetic financial time-series data.
        missing_rate:
            Fraction of rows selected for missing-value corruption
            in each chosen column.
        duplicate_rate:
            Fraction of rows duplicated.
        invalid_rate:
            Fraction of rows selected for deliberately invalid values.
        random_seed:
            Seed used to make corruption reproducible.

    Returns:
        A corrupted copy representing raw Bronze-style data.
    """
    rng = np.random.default_rng(random_seed)

    bronze_df = df.copy()

    n_rows = len(bronze_df)

    # Injecting missing values into selected numeric columns.
    numeric_missing_cols = [
        "balance",
        "cash_inflow",
        "cash_outflow",
        "credit_utilization",
        "payment_ratio",
    ]

    for column in numeric_missing_cols:
        n_missing = max(
            1,
            int(n_rows * missing_rate),
        )

        missing_indices = rng.choice(
            bronze_df.index,
            size=n_missing,
            replace=False,
        )

        bronze_df.loc[
            missing_indices,
            column,
        ] = np.nan

    # Injecting missing categorical values.
    n_missing_categories = max(
        1,
        int(n_rows * missing_rate),
    )

    category_indices = rng.choice(
        bronze_df.index,
        size=n_missing_categories,
        replace=False,
    )

    bronze_df.loc[
        category_indices,
        "account_type",
    ] = np.nan

    # Injecting impossible negative monetary values.
    n_invalid = max(
        1,
        int(n_rows * invalid_rate),
    )

    negative_indices = rng.choice(
        bronze_df.index,
        size=n_invalid,
        replace=False,
    )

    bronze_df.loc[
        negative_indices,
        "cash_outflow",
    ] = -np.abs(
        bronze_df.loc[
            negative_indices,
            "cash_outflow",
        ]
    )

    # Injecting invalid credit utilization above 100%.
    utilization_indices = rng.choice(
        bronze_df.index,
        size=n_invalid,
        replace=False,
    )

    bronze_df.loc[
        utilization_indices,
        "credit_utilization",
    ] = rng.uniform(
        1.05,
        1.50,
        size=n_invalid,
    )

    # Creating duplicate raw records.
    n_duplicates = max(
        1,
        int(n_rows * duplicate_rate),
    )

    duplicate_indices = rng.choice(
        bronze_df.index,
        size=n_duplicates,
        replace=False,
    )

    duplicate_rows = bronze_df.loc[duplicate_indices].copy()

    bronze_df = pd.concat(
        [
            bronze_df,
            duplicate_rows,
        ],
        ignore_index=True,
    )

    # Shuffling rows to mimic unordered ingestion.
    bronze_df = bronze_df.sample(
        frac=1.0,
        random_state=random_seed,
    ).reset_index(drop=True)

    return bronze_df
