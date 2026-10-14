"""Prove account-specific replay matches the original seeded simulator."""
from datetime import date

import pandas as pd

from finance_ml.data.generate_synthetic import generate_holdout_timeseries
from finance_ml.data.synthetic_provenance import SyntheticProvenance
from finance_ml.data.synthetic_replay import replay_selected_accounts


def _profile():
    return SyntheticProvenance("1" * 64, 3, 6, 42, date(2026, 1, 1), date(2026, 1, 10), 1000)


def test_original_known_and_new_account_rows_match():
    reference = generate_holdout_timeseries(n_existing_accounts=3, historical_days=6,
                                             holdout_days=4, n_new_accounts=2, random_seed=42)
    all_reference = pd.concat(reference.values(), ignore_index=True).drop(
        columns=["risk_state", "risk_state_name"])
    accounts = ("ACC_00001", "ACC_00004", "ACC_00005")
    features, truth = replay_selected_accounts(_profile(), accounts, date(2026, 1, 10))
    assert set(features.account_id) == set(accounts)
    assert {"risk_state", "risk_state_name"}.isdisjoint(features.columns)
    assert {"risk_state", "risk_state_name"}.issubset(truth.columns)
    for account in accounts:
        observed = features[features.account_id == account].reset_index(drop=True)
        expected = all_reference[all_reference.account_id == account].reset_index(drop=True)
        pd.testing.assert_frame_equal(observed, expected)


def test_account_replay_independent_of_requested_horizon_and_other_ids():
    account = "ACC_00005"
    p = _profile()
    first, _ = replay_selected_accounts(p, (account,), date(2026, 1, 10))
    later, _ = replay_selected_accounts(p, ("ACC_00004", account), date(2026, 1, 14))
    same_period = later[(later.account_id == account) & (later.date <= pd.Timestamp("2026-01-10"))]
    pd.testing.assert_frame_equal(first.reset_index(drop=True), same_period.reset_index(drop=True))
    assert len(later[later.account_id == account]) == 14


def test_known_history_matches_legacy_history_exactly():
    from finance_ml.data.generate_synthetic import generate_financial_timeseries
    p = _profile()
    replay, _ = replay_selected_accounts(p, ("ACC_00002",), date(2026, 1, 6))
    historical = generate_financial_timeseries(n_accounts=3, n_days=6, random_seed=42)
    historical = historical[historical.account_id == "ACC_00002"].drop(
        columns=["risk_state", "risk_state_name"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(replay.reset_index(drop=True), historical)
