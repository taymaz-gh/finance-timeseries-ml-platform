"""Checking deterministic, label-free synthetic future generation."""
import pandas as pd
import pytest
from finance_ml.data.generate_synthetic import generate_future_observations


def _latest():
    return pd.DataFrame([
        dict(account_id="ACC_00001", date="2026-03-01", account_type="checking",
             balance=5000.0, credit_utilization=0.4, payment_ratio=0.85,
             days_past_due=0, amount_due=100.0),
        dict(account_id="ACC_00002", date="2026-03-01", account_type="credit",
             balance=2000.0, credit_utilization=0.7, payment_ratio=0.55,
             days_past_due=10, amount_due=150.0),
    ])


def test_future_is_deterministic_and_label_free():
    a = generate_future_observations(_latest(), n_days=14, random_seed=84)
    b = generate_future_observations(_latest().iloc[::-1], n_days=14, random_seed=84)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 28
    assert "risk_state" not in a.columns
    assert "risk_state_name" not in a.columns
    assert a.date.min() == pd.Timestamp("2026-03-02")
    assert a.date.max() == pd.Timestamp("2026-03-15")
    assert set(a.account_id) == {"ACC_00001", "ACC_00002"}


def test_future_rejects_invalid_states():
    with pytest.raises(ValueError, match="one row per account"):
        generate_future_observations(pd.concat([_latest(), _latest()]), n_days=3)
    with pytest.raises(ValueError, match="positive"):
        generate_future_observations(_latest(), n_days=0)
