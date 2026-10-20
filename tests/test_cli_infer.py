"""Check redesigned Typer commands without TensorFlow or a Spark JVM."""
from datetime import date
from unittest.mock import Mock
from click.utils import strip_ansi

from typer.testing import CliRunner

from finance_ml.cli import app
from finance_ml.data.synthetic_provenance import SyntheticProvenance
from finance_ml.pipelines import account_inference_runner as runner

cli = CliRunner()


def test_missing_account_or_date_is_rejected():
    r = cli.invoke(app, ["infer", "--bundle", "dummy", "--dry-run"])
    assert r.exit_code != 0
    assert "--account-id" in strip_ansi(r.output)


def test_dry_run_does_not_score(monkeypatch):
    p = SyntheticProvenance("1" * 64, 8, 60, 42, date(2026, 1, 1), date(2026, 4, 30), 1000)
    monkeypatch.setattr(runner, "inspect_local_bundle", lambda path: (
        {"sequence_length": 7, "n_classes": 4, "final_feature_columns": ["balance"]},
        {"model": {"sha256": "1" * 64}},
    ))
    monkeypatch.setattr(runner, "load_profile", lambda *a, **k: p)
    dispatch = Mock()
    monkeypatch.setattr(runner, "run_account_inference", dispatch)
    r = cli.invoke(app, ["infer", "--bundle", "dummy", "--account-id", "521",
                          "--start-date", "2026-03-05", "--end-date", "2026-03-07", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "ACC_00521" in r.output
    assert "No Spark session" in r.output
    dispatch.assert_not_called()


def test_inference_dispatch_preserves_account_date_request(monkeypatch):
    p = SyntheticProvenance("1" * 64, 8, 60, 42, date(2026, 1, 1), date(2026, 4, 30), 1000)
    monkeypatch.setattr(runner, "inspect_local_bundle", lambda path: (
        {"sequence_length": 7, "n_classes": 4, "final_feature_columns": ["balance"]},
        {"model": {"sha256": "1" * 64}},
    ))
    monkeypatch.setattr(runner, "load_profile", lambda *a, **k: p)
    dispatch = Mock(return_value={"run_dir": "artifacts/sample", "summary": {
        "requested": 3, "predictions": 3, "insufficient_context": 0}})
    monkeypatch.setattr(runner, "run_account_inference", dispatch)
    r = cli.invoke(app, ["infer", "--bundle", "dummy", "--account-id", "521",
                          "--start-date", "2026-03-05", "--end-date", "2026-03-07", "--smoke"])
    assert r.exit_code == 0, r.output
    assert dispatch.call_args.kwargs["account_ids"] == ["521"]
    assert dispatch.call_args.kwargs["start_date"] == "2026-03-05"
    assert "3/3" in r.output


def test_databricks_requires_explicit_uc_model_source():
    r = cli.invoke(app, ["infer", "--bundle", "dummy", "--account-id", "1",
                         "--start-date", "2026-03-05", "--end-date", "2026-03-07",
                         "--data-source", "databricks"])
    assert r.exit_code == 2
    assert "requires --model-source uc" in r.output


def test_profile_cli_writes_no_model(monkeypatch):
    from finance_ml.data import synthetic_provenance as prov
    create = Mock(return_value="path/provenance.json")
    monkeypatch.setattr(prov, "create_profile", create)
    r = cli.invoke(app, ["profile-init", "--bundle", "dummy", "--development-accounts", "8",
                         "--historical-days", "60", "--observed-through", "2026-04-30"])
    assert r.exit_code == 0, r.output
    create.assert_called_once()
    assert "user-declared" in r.output


def test_dry_run_and_smoke_mutually_exclusive():
    r = cli.invoke(app, ["infer", "--bundle", "dummy", "--account-id", "1",
                         "--start-date", "2026-03-01", "--end-date", "2026-03-02",
                         "--dry-run", "--smoke"])
    assert r.exit_code == 2
    assert "not both" in r.output
