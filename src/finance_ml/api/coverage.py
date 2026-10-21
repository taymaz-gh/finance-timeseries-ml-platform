"""Assessing per-account historical context for model inference."""

from __future__ import annotations

import pandas as pd


def assess_account_context(
    observations: pd.DataFrame,
    *,
    sequence_length: int,
) -> pd.DataFrame:
    """Assessing sequence availability for each account."""

    # Validating the model's sequence-length requirement.
    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")

    required = {"account_id", "date"}

    if not required.issubset(observations.columns):
        raise ValueError("Missing required account_id or date column")

    if observations.empty:
        raise ValueError("No observations were provided")

    frame = observations[["account_id", "date"]].copy()

    # Validating account identifiers.
    valid_ids = frame["account_id"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )

    if not valid_ids.all():
        raise ValueError("Invalid account_id values")

    # Standardizing daily dates.
    try:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid observation dates") from exc

    if (
        frame["date"].isna().any()
        or not frame["date"].eq(frame["date"].dt.normalize()).all()
    ):
        raise ValueError("Dates must represent valid daily observations")

    # Rejecting unresolved duplicate account-date keys.
    if frame.duplicated(["account_id", "date"]).any():
        raise ValueError("Duplicate account/date observations")

    reports = []

    # Evaluating each account independently.
    for account_id, group in frame.groupby("account_id", sort=True):

        dates = group["date"].sort_values()
        available = len(dates)

        # Counting missing calendar days between observations.
        missing_days = int((dates.diff().dt.days.dropna() - 1).clip(lower=0).sum())

        reports.append(
            {
                "account_id": account_id,
                "available_observations": available,
                "required_observations": sequence_length,
                "missing_for_minimum": max(0, sequence_length - available),
                "eligible_endpoints": max(0, available - sequence_length + 1),
                "missing_calendar_days": missing_days,
                "status": (
                    "eligible"
                    if available >= sequence_length
                    else "insufficient_context"
                ),
            }
        )

    return pd.DataFrame(reports)
