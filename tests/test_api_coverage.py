"""Testing per-account sequence eligibility and coverage reporting."""

import pandas as pd
import pytest

from finance_ml.api.coverage import assess_account_context


def make_account_rows(account_id: str, n_days: int) -> list[dict]:
    """Generating daily account keys for testing."""

    dates = pd.date_range("2026-01-01", periods=n_days)

    return [
        {"account_id": account_id, "date": date.date().isoformat()} for date in dates
    ]


def test_multiple_accounts_independently() -> None:
    """Checking eligibility without rejecting other valid accounts."""

    rows = (
        make_account_rows("ACC_001", 10)
        + make_account_rows("ACC_002", 7)
        + make_account_rows("ACC_003", 4)
    )

    # Testing that input row ordering does not affect the results.
    report = assess_account_context(
        pd.DataFrame(rows[::-1]),
        sequence_length=7,
    ).set_index("account_id")

    assert report.loc["ACC_001", "status"] == "eligible"
    assert report.loc["ACC_001", "eligible_endpoints"] == 4

    assert report.loc["ACC_002", "status"] == "eligible"
    assert report.loc["ACC_002", "eligible_endpoints"] == 1

    assert report.loc["ACC_003", "status"] == "insufficient_context"
    assert report.loc["ACC_003", "missing_for_minimum"] == 3
    assert report.loc["ACC_003", "eligible_endpoints"] == 0


@pytest.mark.parametrize(
    "sequence_length, expected_endpoints",
    [(4, 4), (7, 1), (8, 0)],
)
def test_dynamic_sequence_length(
    sequence_length: int,
    expected_endpoints: int,
) -> None:
    """Checking minimum history against the selected model contract."""

    frame = pd.DataFrame(make_account_rows("ACC_001", 7))

    report = assess_account_context(
        frame,
        sequence_length=sequence_length,
    )

    assert report.loc[0, "required_observations"] == sequence_length
    assert report.loc[0, "eligible_endpoints"] == expected_endpoints


def test_missing_calendar_day() -> None:
    """Detecting calendar gaps separately from sequence length."""

    dates = [
        "2026-01-01",
        "2026-01-02",
        "2026-01-04",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
    ]

    frame = pd.DataFrame(
        {
            "account_id": ["ACC_001"] * len(dates),
            "date": dates,
        }
    )

    report = assess_account_context(frame, sequence_length=7)

    assert report.loc[0, "missing_calendar_days"] == 1
    assert report.loc[0, "status"] == "eligible"


def test_duplicate_account_dates_rejected() -> None:
    """Rejecting unresolved duplicate account/date keys."""

    rows = make_account_rows("ACC_001", 7)
    rows.append(rows[0].copy())

    with pytest.raises(ValueError, match="Duplicate"):
        assess_account_context(
            pd.DataFrame(rows),
            sequence_length=7,
        )


def test_invalid_date_rejected() -> None:
    """Rejecting impossible dates."""

    frame = pd.DataFrame(
        {
            "account_id": ["ACC_001"],
            "date": ["2026-02-30"],
        }
    )

    with pytest.raises(ValueError, match="Invalid observation dates"):
        assess_account_context(frame, sequence_length=7)


def test_missing_account_id_rejected() -> None:
    """Rejecting missing account identifiers."""

    frame = pd.DataFrame(
        {
            "account_id": [None],
            "date": ["2026-01-01"],
        }
    )

    with pytest.raises(ValueError, match="Invalid account_id"):
        assess_account_context(frame, sequence_length=7)


@pytest.mark.parametrize("sequence_length", [0, -1, True])
def test_invalid_sequence_length_rejected(sequence_length) -> None:
    """Rejecting invalid model sequence configurations."""

    frame = pd.DataFrame(make_account_rows("ACC_001", 7))

    with pytest.raises(ValueError, match="sequence_length"):
        assess_account_context(
            frame,
            sequence_length=sequence_length,
        )
