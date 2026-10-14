"""Preparing in-memory synthetic datasets for local demonstrations.

Reusing the established financial time-series simulator while separating
future predictor observations from their evaluation ground truth.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from finance_ml.data.generate_synthetic import generate_holdout_timeseries


LABEL_COLUMNS = ("risk_state", "risk_state_name")
MAX_DEMO_ROWS = 20_000


def generate_in_memory_synthetic_data(
    *,
    n_existing_accounts: int = 3,
    historical_days: int = 10,
    future_days: int = 4,
    n_new_accounts: int = 2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Generate reproducible historical and label-free future observations.

    Returning historical labeled data for future training experiments,
    future predictors without target columns, and separate ground truth.
    Never persisting the generated data.
    """
    integer_parameters = {
        "n_existing_accounts": n_existing_accounts,
        "historical_days": historical_days,
        "future_days": future_days,
        "n_new_accounts": n_new_accounts,
    }

    for key, value in integer_parameters.items():
        if type(value) is not int or value < (0 if key == "n_new_accounts" else 1):
            raise ValueError(f"{key} has an invalid count")

    if type(random_seed) is not int or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")

    total_rows = n_existing_accounts * (
        historical_days + future_days
    ) + n_new_accounts * (historical_days + future_days)

    if total_rows > MAX_DEMO_ROWS:
        raise ValueError(f"Local demonstration exceeds the {MAX_DEMO_ROWS:,}-row limit")

    generated = generate_holdout_timeseries(
        n_existing_accounts=n_existing_accounts,
        historical_days=historical_days,
        holdout_days=future_days,
        n_new_accounts=n_new_accounts,
        random_seed=random_seed,
    )

    def separate_future(
        labeled: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Separate future input features from evaluation labels."""
        truth = labeled[["account_id", "date", *LABEL_COLUMNS]].copy()

        observations = labeled.drop(columns=list(LABEL_COLUMNS)).copy()

        return observations, truth

    temporal, temporal_truth = separate_future(generated["temporal_holdout"])
    account, account_truth = separate_future(generated["account_holdout"])

    return {
        "historical": generated["historical"].copy(),
        "temporal_observations": temporal,
        "account_observations": account,
        "temporal_truth": temporal_truth,
        "account_truth": account_truth,
    }
