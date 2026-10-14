"""Read-only account/date selection for the verified UC LSTM v1 cohort.

The fixed table map and population boundaries come from notebook 10. This
module contains no write statements and makes no network calls during preflight.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import re
from typing import Callable, Sequence

import pandas as pd

from finance_ml.data.synthetic_provenance import normalize_account_id, parse_date

MODEL_NAME = "finance_ml.models.lstm_risk_classifier"
MODEL_VERSION = "1"
DEVELOPMENT_ACCOUNTS = 2000
TOTAL_ACCOUNTS = 3000
START = date(2026, 1, 1)
DEV_END = date(2026, 3, 1)
AVAILABLE_END = date(2026, 4, 30)
HISTORY_GOLD = "finance_ml.gold.account_features"
DATE_GOLD = "finance_ml.gold.account_features_temporal_holdout"
NEW_GOLD = "finance_ml.gold.account_features_account_holdout"
DATE_TRUTH = "finance_ml.gold.account_labels_temporal_holdout"
NEW_TRUTH = "finance_ml.gold.account_labels_account_holdout"
MAX_ACCOUNTS = 25
MAX_ROWS = 20_000


@dataclass(frozen=True)
class UCRequest:
    """Validated, strictly bounded, immutable remote inference selection."""
    accounts: tuple[str, ...]
    start: date
    end: date
    smoke: bool
    query_method: str
    evaluate: bool

    @property
    def requested_rows(self) -> int:
        return len(self.accounts) * ((self.end - self.start).days + 1)


def cohort(account_id: str, day: date) -> str:
    """Classify a requested key relative to the original UC training data."""
    number = int(account_id.split("_")[1])
    if number <= DEVELOPMENT_ACCOUNTS:
        return "development_period" if day <= DEV_END else "date_holdout"
    return "account_holdout" if day <= DEV_END else "joint_holdout"


def validate_uc_request(*, account_ids: Sequence[str], start_date: str,
                        end_date: str, query_method: str = "connector",
                        smoke: bool = False, evaluate: bool = False,
                        model_name: str = MODEL_NAME,
                        model_version: str = MODEL_VERSION) -> UCRequest:
    """Reject unsupported model populations and unavailable dates *offline*."""
    if model_name != MODEL_NAME or str(model_version) != MODEL_VERSION:
        raise ValueError("The documented UC cohort supports only finance_ml.models.lstm_risk_classifier version 1")
    if query_method not in ("connector", "sqlalchemy"):
        raise ValueError("Standalone Databricks inference uses --query-method connector or sqlalchemy")
    if type(smoke) is not bool or type(evaluate) is not bool:
        raise ValueError("Smoke and evaluate flags must be boolean")
    if not account_ids:
        raise ValueError("Provide at least one --account-id")
    accounts = tuple(normalize_account_id(raw) for raw in account_ids)
    if len(accounts) != len(set(accounts)):
        raise ValueError("Duplicate account IDs")
    if len(accounts) > MAX_ACCOUNTS:
        raise ValueError(f"Maximum {MAX_ACCOUNTS} account IDs per local session")
    if any(int(a.split("_")[1]) > TOTAL_ACCOUNTS for a in accounts):
        raise ValueError("Account ID exceeds the original persisted cohort (ACC_03000)")
    start, end = parse_date(start_date, "start_date"), parse_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must not exceed end_date")
    if start < START or end > AVAILABLE_END:
        raise ValueError(f"Persisted UC observations cover {START} through {AVAILABLE_END}; no synthetic fallback")
    n_days = (end - start).days + 1
    if n_days * len(accounts) > MAX_ROWS:
        raise ValueError(f"Selected observations exceed {MAX_ROWS:,} rows")
    if smoke and (len(accounts) > 2 or n_days > 7):
        raise ValueError("Smoke accepts at most two accounts and seven days; it never changes the request")
    return UCRequest(accounts, start, end, smoke, query_method, evaluate)


def _quote_accounts(accounts: Sequence[str]) -> str:
    # Account IDs were validated against a strict numeric identifier pattern.
    return ", ".join(f"'{account}'" for account in accounts)


def build_select(table: str, accounts: Sequence[str], start: date, end: date,
                 *, columns: str = "*") -> str:
    """Create a bounded SELECT from a static allowlisted table and column list."""
    if table not in {HISTORY_GOLD, DATE_GOLD, NEW_GOLD, DATE_TRUTH, NEW_TRUTH}:
        raise ValueError("Unsupported UC source table")
    column_list = [column.strip() for column in columns.split(",")]
    if not column_list or (columns != "*" and (
        any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in column_list)
        or len(set(column_list)) != len(column_list)
        or column_list[:2] != ["account_id", "date"]
    )):
        raise ValueError("Unsafe or unsupported SELECT columns")
    if not accounts or start > end:
        raise ValueError("SQL selection requires account IDs and an increasing date window")
    end_exclusive = end + timedelta(days=1)
    return (
        f"SELECT {columns} FROM {table} "
        f"WHERE account_id IN ({_quote_accounts(accounts)}) "
        f"AND date >= TIMESTAMP '{start.isoformat()} 00:00:00' "
        f"AND date < TIMESTAMP '{end_exclusive.isoformat()} 00:00:00' "
        f"ORDER BY account_id, date"
    )


def gold_query_plan(request: UCRequest, *, sequence_length: int = 7,
                    source_columns: Sequence[str] | None = None) -> list[tuple[str, str]]:
    """Read the six preceding days, crossing the development cutoff safely."""
    if type(sequence_length) is not int or sequence_length < 1 or sequence_length > 120:
        raise ValueError("Invalid model sequence_length")
    begin = max(START, request.start - timedelta(days=sequence_length - 1))
    columns = "*" if source_columns is None else ", ".join(["account_id", "date", *source_columns])
    plan: list[tuple[str, str]] = []
    known = [a for a in request.accounts if int(a.split("_")[1]) <= DEVELOPMENT_ACCOUNTS]
    new = [a for a in request.accounts if int(a.split("_")[1]) > DEVELOPMENT_ACCOUNTS]
    if known and begin <= DEV_END:
        plan.append((HISTORY_GOLD, build_select(HISTORY_GOLD, known, begin, min(DEV_END, request.end), columns=columns)))
    if known and request.end > DEV_END:
        plan.append((DATE_GOLD, build_select(DATE_GOLD, known, max(begin, DEV_END + timedelta(days=1)), request.end, columns=columns)))
    if new:
        plan.append((NEW_GOLD, build_select(NEW_GOLD, new, begin, request.end, columns=columns)))
    return plan


def truth_query_plan(request: UCRequest) -> list[tuple[str, str]]:
    """Select only requested post-development ground truth for opt-in metrics."""
    if not request.evaluate:
        return []
    plan = []
    known = [a for a in request.accounts if int(a.split("_")[1]) <= DEVELOPMENT_ACCOUNTS]
    new = [a for a in request.accounts if int(a.split("_")[1]) > DEVELOPMENT_ACCOUNTS]
    if known and request.end > DEV_END:
        plan.append((DATE_TRUTH, build_select(DATE_TRUTH, known,
                    max(request.start, DEV_END + timedelta(days=1)), request.end,
                    columns="account_id, date, risk_state")))
    if new:
        plan.append((NEW_TRUTH, build_select(NEW_TRUTH, new, request.start, request.end,
                    columns="account_id, date, risk_state")))
    return plan


def normalize_utc_daily_dates(values: pd.Series) -> pd.Series:
    """Normalize Databricks UTC-aware/naive daily keys to naive UTC midnight.

    SQL Connector timestamps can carry ``Etc/UTC`` whereas CLI-generated dates
    and Spark DateType output have no timezone. Converting to UTC before
    dropping tz information preserves the actual calendar day and prevents
    pandas merge type mismatches. Non-midnight timestamps are rejected rather
    than silently truncating timestamps to a different observation key.
    """
    dates = pd.to_datetime(values, errors="raise", utc=True)
    if dates.isna().any():
        raise ValueError("Databricks date keys contain null timestamps")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Databricks daily date keys must represent midnight UTC")
    return dates.dt.tz_localize(None)


def execute_queries(plan: list[tuple[str, str]], query_fn: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
    """Combine read-only bounded query results without inventing absent rows."""
    frames = []
    for table, sql in plan:
        result = query_fn(sql)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"Databricks query did not return a pandas DataFrame for {table}")
        if result.empty:
            continue
        if not {"account_id", "date"} <= set(result.columns):
            raise ValueError(f"Databricks {table} missing required key columns")
        frames.append(result.copy())
    if not frames:
        raise ValueError("No persisted Databricks observations found for the requested account/date range")
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = normalize_utc_daily_dates(combined["date"])
    if combined["date"].isna().any() or combined["account_id"].isna().any():
        raise ValueError("Databricks query returned null account/date keys")
    if combined.duplicated(["account_id", "date"]).any():
        raise ValueError("Overlapping or duplicated Databricks account/date records")
    return combined.sort_values(["account_id", "date"]).reset_index(drop=True)
