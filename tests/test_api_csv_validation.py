"""Testing validation of uploaded raw financial CSV files."""

from io import StringIO

import pandas as pd
import pytest

from finance_ml.api.csv_validation import read_account_csv


BASE_ROW = {
    "account_id": "ACC_00001",
    "date": "2026-01-01",
    "account_type": "checking",
    "balance": 1200.0,
    "cash_inflow": 500.0,
    "cash_outflow": 300.0,
    "transaction_count": 8,
    "credit_utilization": 0.35,
    "amount_due": 150.0,
    "amount_paid": 120.0,
    "payment_ratio": 0.8,
    "days_past_due": 0,
}


def make_csv(rows: list[dict]) -> StringIO:
    """Creating an in-memory CSV file for testing."""
    return StringIO(pd.DataFrame(rows).to_csv(index=False))


def test_valid_single_account() -> None:
    """Accepting a correctly structured account observation."""

    result = read_account_csv(make_csv([BASE_ROW]))

    assert len(result) == 1
    assert result.loc[0, "account_id"] == "ACC_00001"
    assert result.loc[0, "balance"] == 1200.0
    assert result.loc[0, "date"] == pd.Timestamp("2026-01-01")


def test_multiple_accounts() -> None:
    """Preserving observations from multiple accounts."""

    rows = [
        BASE_ROW,
        {**BASE_ROW, "account_id": "ACC_00002"},
        {**BASE_ROW, "date": "2026-01-02"},
    ]

    result = read_account_csv(make_csv(rows))

    assert len(result) == 3
    assert result["account_id"].nunique() == 2


def test_missing_values_preserved() -> None:
    """Preserving missing predictors for downstream imputation."""

    row = {
        **BASE_ROW,
        "balance": None,
        "account_type": None,
    }

    result = read_account_csv(make_csv([row]))

    assert pd.isna(result.loc[0, "balance"])
    assert pd.isna(result.loc[0, "account_type"])


def test_invalid_financial_values_preserved() -> None:
    """Deferring financial-domain corrections to Silver preprocessing."""

    row = {
        **BASE_ROW,
        "cash_outflow": -100.0,
        "credit_utilization": 1.5,
    }

    result = read_account_csv(make_csv([row]))

    assert result.loc[0, "cash_outflow"] == -100.0
    assert result.loc[0, "credit_utilization"] == 1.5


def test_duplicate_account_dates_preserved() -> None:
    """Leaving deduplication to the existing Silver pipeline."""

    result = read_account_csv(make_csv([BASE_ROW, BASE_ROW.copy()]))

    assert len(result) == 2


def test_missing_required_column() -> None:
    """Rejecting files without the complete raw schema."""

    row = BASE_ROW.copy()
    del row["cash_inflow"]

    with pytest.raises(ValueError, match="Invalid raw CSV schema"):
        read_account_csv(make_csv([row]))


def test_unexpected_column() -> None:
    """Rejecting unsupported input columns."""

    row = {**BASE_ROW, "unknown_feature": 42}

    with pytest.raises(ValueError, match="Invalid raw CSV schema"):
        read_account_csv(make_csv([row]))


def test_target_leakage_rejected() -> None:
    """Preventing target labels from entering inference requests."""

    row = {**BASE_ROW, "risk_state": 2}

    with pytest.raises(ValueError, match="target columns"):
        read_account_csv(make_csv([row]))


@pytest.mark.parametrize(
    "invalid_date",
    ["01/01/2026", "2026-02-30"],
)
def test_invalid_dates_rejected(invalid_date: str) -> None:
    """Rejecting incorrectly formatted or impossible calendar dates."""

    row = {**BASE_ROW, "date": invalid_date}

    with pytest.raises(ValueError, match="Dates|calendar dates"):
        read_account_csv(make_csv([row]))


def test_invalid_numeric_value() -> None:
    """Rejecting values that cannot be converted to numbers."""

    row = {**BASE_ROW, "balance": "not_a_number"}

    with pytest.raises(ValueError, match="Invalid numeric"):
        read_account_csv(make_csv([row]))


def test_nonfinite_numeric_value() -> None:
    """Rejecting infinite numeric values."""

    row = {**BASE_ROW, "balance": "inf"}

    with pytest.raises(ValueError, match="Non-finite"):
        read_account_csv(make_csv([row]))


def test_empty_csv() -> None:
    """Rejecting a CSV containing headers but no records."""

    empty_csv = StringIO(pd.DataFrame(columns=BASE_ROW.keys()).to_csv(index=False))

    with pytest.raises(ValueError, match="no observations"):
        read_account_csv(empty_csv)


def test_row_limit(monkeypatch) -> None:
    """Rejecting CSV files exceeding the configured row limit."""

    monkeypatch.setattr(
        "finance_ml.api.csv_validation.MAX_CSV_ROWS",
        2,
    )

    rows = [BASE_ROW.copy() for _ in range(3)]

    with pytest.raises(ValueError, match="row limit"):
        read_account_csv(make_csv(rows))
