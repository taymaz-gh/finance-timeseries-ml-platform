"""Validating uploaded raw financial account-day CSV data."""

from __future__ import annotations

from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd


MAX_CSV_ROWS = 20_000

NUMERIC_COLUMNS = (
    "balance",
    "cash_inflow",
    "cash_outflow",
    "transaction_count",
    "credit_utilization",
    "amount_due",
    "amount_paid",
    "payment_ratio",
    "days_past_due",
)

REQUIRED_COLUMNS = (
    "account_id",
    "date",
    "account_type",
    *NUMERIC_COLUMNS,
)

TARGET_COLUMNS = {"risk_state", "risk_state_name"}


def read_account_csv(source: BinaryIO | TextIO) -> pd.DataFrame:
    """Reading and validating unlabeled raw account-day observations."""

    # Reading uploaded values as strings before validating their types.
    try:
        frame = pd.read_csv(
            source,
            dtype="string",
            encoding="utf-8-sig",
            nrows=MAX_CSV_ROWS + 1,
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeError,
    ) as exc:
        raise ValueError("Invalid or empty CSV file") from exc

    # Enforcing bounded input size.
    if frame.empty:
        raise ValueError("CSV contains no observations")

    if len(frame) > MAX_CSV_ROWS:
        raise ValueError(f"CSV exceeds the {MAX_CSV_ROWS:,}-row limit")

    # Standardizing column-name whitespace.
    frame.columns = frame.columns.str.strip()

    if not frame.columns.is_unique:
        raise ValueError("Duplicate CSV column names")

    # Preventing target labels from entering inference.
    if TARGET_COLUMNS.intersection(frame.columns):
        raise ValueError("Uploaded CSV must not contain risk-state target columns")

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(REQUIRED_COLUMNS))

    if missing or unexpected:
        raise ValueError(
            f"Invalid raw CSV schema. " f"Missing: {missing}; unexpected: {unexpected}"
        )

    frame = frame[list(REQUIRED_COLUMNS)].copy()

    # Validating account identifiers without assuming synthetic IDs.
    frame["account_id"] = frame["account_id"].str.strip()

    if frame["account_id"].isna().any() or frame["account_id"].eq("").any():
        raise ValueError("Invalid or missing account_id")

    # Requiring valid ISO calendar dates.
    valid_date_format = frame["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}")

    if not valid_date_format.fillna(False).all():
        raise ValueError("Dates must use YYYY-MM-DD")

    try:
        frame["date"] = pd.to_datetime(
            frame["date"],
            format="%Y-%m-%d",
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid calendar dates") from exc

    # Preserving missing account types for Silver cleaning.
    frame["account_type"] = frame["account_type"].str.strip().replace("", pd.NA)

    # Validating numeric features while preserving missing values.
    for column in NUMERIC_COLUMNS:
        try:
            values = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric values in column: {column}") from exc

        if not np.isfinite(values.dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite numeric values in column: {column}")

        frame[column] = values

    return frame
