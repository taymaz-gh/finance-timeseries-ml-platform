"""Checking historical replay and deterministic holdout simulation."""
import numpy as np
import pandas as pd
import pytest
from finance_ml.data.generate_synthetic import (
    generate_financial_timeseries,
    generate_holdout_timeseries,
)


def _generate(**overrides):
    return generate_holdout_timeseries(
        n_existing_accounts=3, historical_days=6, holdout_days=4,
        n_new_accounts=2, start_date="2026-01-01", random_seed=42,
        **overrides,
    )


def test_original_generator_reference_row():
    data = generate_financial_timeseries(n_accounts=2, n_days=60)
    first = data.iloc[0]
    assert (first.account_id, first.account_type, first.balance) == ("ACC_00001", "credit", 7624.34)
    assert first.cash_inflow == 150.02
    assert first.cash_outflow == 160.29
    assert first.transaction_count == 12
    assert first.risk_state == 0
    assert first.risk_state_name == "normal"


def test_historical_prefix_unchanged_when_more_accounts_added():
    original = generate_financial_timeseries(n_accounts=2, n_days=6)
    more = generate_financial_timeseries(n_accounts=3, n_days=6)
    pd.testing.assert_frame_equal(original, more.iloc[:12].reset_index(drop=True), check_exact=True)


def test_legacy_shared_rng_changes_second_account_when_days_change():
    short = generate_financial_timeseries(n_accounts=2, n_days=6)
    long = generate_financial_timeseries(n_accounts=2, n_days=10)
    pd.testing.assert_frame_equal(short.iloc[:6].reset_index(drop=True), long.iloc[:6].reset_index(drop=True))
    assert not short.iloc[6:12].reset_index(drop=True).equals(long.iloc[10:16].reset_index(drop=True))


def test_replayed_history_equals_original_exactly():
    result = _generate()
    reference = generate_financial_timeseries(n_accounts=3, n_days=6)
    pd.testing.assert_frame_equal(result["historical"], reference, check_exact=True)


def test_holdout_account_date_ranges_and_uniqueness():
    result = _generate()
    assert {k: len(v) for k, v in result.items()} == {
        "historical": 18, "temporal_holdout": 12, "account_holdout": 20
    }
    assert set(result["historical"].account_id) == {"ACC_00001", "ACC_00002", "ACC_00003"}
    assert set(result["temporal_holdout"].account_id) == set(result["historical"].account_id)
    assert set(result["account_holdout"].account_id) == {"ACC_00004", "ACC_00005"}
    assert result["temporal_holdout"].date.min() == pd.Timestamp("2026-01-07")
    assert result["temporal_holdout"].date.max() == pd.Timestamp("2026-01-10")
    assert result["account_holdout"].date.min() == pd.Timestamp("2026-01-01")
    assert result["account_holdout"].date.max() == pd.Timestamp("2026-01-10")
    for frame in result.values():
        assert not frame.duplicated(["account_id", "date"]).any()
        assert set(["risk_state", "risk_state_name"]).issubset(frame.columns)


def test_generation_is_deterministic():
    first = _generate()
    second = _generate()
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name], check_exact=True)


def test_new_account_count_does_not_change_existing_trajectories():
    first = generate_holdout_timeseries(n_existing_accounts=3, historical_days=6, holdout_days=4, n_new_accounts=0)
    second = _generate()
    for name in ("historical", "temporal_holdout"):
        pd.testing.assert_frame_equal(first[name], second[name], check_exact=True)
    assert first["account_holdout"].empty
    assert list(first["account_holdout"].columns) == list(first["historical"].columns)


def test_holdout_horizon_does_not_change_existing_history_or_future_prefix():
    short = generate_holdout_timeseries(n_existing_accounts=3, historical_days=6, holdout_days=2, n_new_accounts=2)
    long = _generate()
    pd.testing.assert_frame_equal(short["historical"], long["historical"], check_exact=True)
    # Comparing each account's first two future records.
    expected = long["temporal_holdout"].groupby("account_id", sort=False).head(2).reset_index(drop=True)
    pd.testing.assert_frame_equal(short["temporal_holdout"], expected, check_exact=True)


@pytest.mark.parametrize("field,value", [
    ("n_existing_accounts", 0), ("historical_days", 0),
    ("holdout_days", 0), ("n_new_accounts", -1),
    ("n_existing_accounts", True), ("n_new_accounts", True),
])
def test_invalid_counts_rejected(field, value):
    args = dict(n_existing_accounts=3, historical_days=6, holdout_days=4, n_new_accounts=2)
    args[field] = value
    with pytest.raises(ValueError):
        generate_holdout_timeseries(**args)
