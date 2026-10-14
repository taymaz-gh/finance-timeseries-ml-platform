"""Verify sidecar provenance cannot silently change the model association."""
from datetime import date
import json

import pytest

from finance_ml.data import synthetic_provenance as prov

SHA = "1" * 64


def _patch_manifest(monkeypatch):
    monkeypatch.setattr(prov, "inspect_local_bundle", lambda path: ({}, {"model": {"sha256": SHA}}))


def test_create_and_load_profile_without_rewriting_bundle(tmp_path, monkeypatch):
    _patch_manifest(monkeypatch)
    p = prov.create_profile(tmp_path, development_accounts=8, historical_days=60,
                            seed=42, start_date="2026-01-01", observed_through="2026-04-30")
    assert p.name == "synthetic_provenance.json"
    obj = prov.load_profile(tmp_path)
    assert obj.development_end == date(2026, 3, 1)
    assert obj.anchor_days == 120
    assert obj.development_accounts == 8
    with pytest.raises(FileExistsError):
        prov.create_profile(tmp_path, development_accounts=8, historical_days=60,
                            seed=42, start_date="2026-01-01", observed_through="2026-04-30")


def test_reject_changed_model_hash(tmp_path, monkeypatch):
    _patch_manifest(monkeypatch)
    p = prov.create_profile(tmp_path, development_accounts=8, historical_days=60,
                            seed=42, start_date="2026-01-01", observed_through="2026-04-30")
    data = json.loads(p.read_text())
    data["model_sha256"] = "2" * 64
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different Keras checkpoint"):
        prov.load_profile(tmp_path)


@pytest.mark.parametrize("argument", ["ACC_00000", "foo", "ACC_521", "ACC_-21", "ACC_000001", " 521"])
def test_invalid_account_ids(argument):
    with pytest.raises(ValueError):
        prov.normalize_account_id(argument)


def test_normalized_account_id():
    assert prov.normalize_account_id("521") == "ACC_00521"
    assert prov.normalize_account_id("ACC_00521") == "ACC_00521"


def test_invalid_date():
    with pytest.raises(ValueError, match="calendar"):
        prov.parse_date("2026-02-30", "start")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        prov.parse_date("2026-1-1", "start")


def test_legacy_provenance_warns_explicitly(tmp_path, monkeypatch):
    _patch_manifest(monkeypatch)
    p = prov.create_profile(tmp_path, development_accounts=8, historical_days=60,
                            seed=42, start_date="2026-01-01", observed_through="2026-04-30")
    assert json.loads(p.read_text())["provenance_level"] == "user_declared_unverified"
