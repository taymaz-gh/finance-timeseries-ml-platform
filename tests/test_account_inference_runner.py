"""Test account/date classification and opt-in rules without starting Spark."""
from datetime import date

import pytest

from finance_ml.data.synthetic_provenance import SyntheticProvenance
from finance_ml.pipelines import account_inference_runner as run


def _setup(monkeypatch):
    p = SyntheticProvenance("1" * 64, 8, 60, 42, date(2026, 1, 1),
                            date(2026, 4, 30), 1000)
    monkeypatch.setattr(run, "inspect_local_bundle", lambda path: (
        {"sequence_length": 7, "n_classes": 4, "final_feature_columns": ["balance"]},
        {"model": {"sha256": "1" * 64}},
    ))
    monkeypatch.setattr(run, "load_profile", lambda *a, **k: p)
    return p


def _validate(**kwargs):
    return run.validate_account_request(bundle_dir="ignored", account_ids=["521"],
          start_date="2026-05-01", end_date="2026-05-03", **kwargs)


def test_cohort_matrix(monkeypatch):
    p = _setup(monkeypatch)
    assert run.cohort_for(p, "ACC_00001", date(2026, 2, 10)) == "development_period"
    assert run.cohort_for(p, "ACC_00001", date(2026, 3, 5)) == "date_holdout"
    assert run.cohort_for(p, "ACC_00521", date(2026, 2, 10)) == "account_holdout"
    assert run.cohort_for(p, "ACC_00521", date(2026, 3, 5)) == "joint_holdout"
    assert run.cohort_for(p, "ACC_00001", date(2026, 5, 1)) == "simulated_extension"


def test_future_requires_explicit_opt_in(monkeypatch):
    _setup(monkeypatch)
    with pytest.raises(ValueError, match="--simulate-future"):
        _validate()
    r = _validate(simulate_future=True)
    assert r["account_ids"] == ("ACC_00521",)
    assert any("not direct risk forecasts" in x for x in r["warnings"])


def test_before_first_date_is_rejected(monkeypatch):
    _setup(monkeypatch)
    with pytest.raises(ValueError, match="earliest available"):
        run.validate_account_request(bundle_dir="ignored", account_ids=["1"],
                                     start_date="2025-12-31", end_date="2026-01-02")


def test_historical_development_warning(monkeypatch):
    _setup(monkeypatch)
    r = run.validate_account_request(bundle_dir="ignored", account_ids=["1"],
                                     start_date="2026-02-01", end_date="2026-02-02")
    assert any("not independent holdout" in w for w in r["warnings"])


def test_account_cap_and_duplicate_rejection(monkeypatch):
    _setup(monkeypatch)
    with pytest.raises(ValueError, match="population"):
        run.validate_account_request(bundle_dir="ignored", account_ids=["1009"],
                                     start_date="2026-04-01", end_date="2026-04-02")
    with pytest.raises(ValueError, match="duplicated"):
        run.validate_account_request(bundle_dir="ignored", account_ids=["1", "ACC_00001"],
                                     start_date="2026-04-01", end_date="2026-04-02")


def test_smoke_does_not_silently_change_request(monkeypatch):
    _setup(monkeypatch)
    with pytest.raises(ValueError, match="Smoke mode"):
        run.validate_account_request(bundle_dir="ignored", account_ids=["1"],
                                     start_date="2026-04-01", end_date="2026-04-15", smoke=True)
    r = run.validate_account_request(bundle_dir="ignored", account_ids=["1", "9"],
                                     start_date="2026-04-01", end_date="2026-04-02", smoke=True)
    assert r["requested_rows"] == 4


def test_extension_is_limited(monkeypatch):
    _setup(monkeypatch)
    with pytest.raises(ValueError, match="limited"):
        run.validate_account_request(bundle_dir="ignored", account_ids=["1"],
                                     start_date="2026-07-01", end_date="2026-07-02",
                                     simulate_future=True)


def test_unneeded_simulation_flag_is_rejected(monkeypatch):
    _setup(monkeypatch)
    with pytest.raises(ValueError, match="only valid"):
        run.validate_account_request(bundle_dir="ignored", account_ids=["1"],
                                     start_date="2026-03-01", end_date="2026-03-02",
                                     simulate_future=True)
