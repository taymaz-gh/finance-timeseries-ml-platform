"""Testing persistence of API inference results."""

import json

import pandas as pd
import pytest

from finance_ml.api.result_store import save_inference_result


@pytest.fixture
def prediction_result() -> dict:
    """Creating a valid seven-observation inference result."""

    account_id = "ACC_00001"

    dates = [f"2026-01-{day:02d}" for day in range(1, 8)]

    return {
        "data_source": "uploaded_csv",
        "cohort_status": "unverified_external_upload",
        "sequence_length": 7,
        "summary": {
            "accounts_received": 1,
            "accounts_eligible": 1,
            "accounts_insufficient_context": 0,
            "observations_after_deduplication": 7,
            "predictions": 1,
            "dates_insufficient_context": 6,
        },
        "warnings": [],
        "account_context": [
            {
                "account_id": account_id,
                "available_observations": 7,
                "required_observations": 7,
                "missing_for_minimum": 0,
                "eligible_endpoints": 1,
                "missing_calendar_days": 0,
                "status": "eligible",
            }
        ],
        "coverage": [
            {
                "account_id": account_id,
                "date": date,
                "status": ("predicted" if index == 6 else "insufficient_context"),
            }
            for index, date in enumerate(dates)
        ],
        "predictions": [
            {
                "account_id": account_id,
                "date": dates[-1],
                "predicted_class": 0,
                "predicted_state_name": "normal",
                "prob_normal": 0.8,
                "prob_elevated_risk": 0.1,
                "prob_liquidity_stress": 0.06,
                "prob_delinquency_risk": 0.04,
            }
        ],
    }


def test_persists_complete_inference_run(
    tmp_path,
    prediction_result,
) -> None:
    """Saving predictions, coverage, context and metadata."""

    output = save_inference_result(
        prediction_result,
        model_sha256="a" * 64,
        output_dir=tmp_path,
    )

    run_dir = tmp_path / "runs" / output["run_id"]

    assert run_dir.is_dir()

    expected_files = {
        "predictions.csv",
        "coverage.csv",
        "account_context.csv",
        "summary.json",
    }

    assert {path.name for path in run_dir.iterdir()} == expected_files

    predictions = pd.read_csv(run_dir / "predictions.csv")
    coverage = pd.read_csv(run_dir / "coverage.csv")
    context = pd.read_csv(run_dir / "account_context.csv")

    assert len(predictions) == 1
    assert len(coverage) == 7
    assert len(context) == 1

    assert predictions.loc[0, "predicted_class"] == 0
    assert predictions.loc[0, "prob_normal"] == pytest.approx(0.8)

    assert coverage["status"].value_counts().to_dict() == {
        "insufficient_context": 6,
        "predicted": 1,
    }

    assert (
        predictions["run_id"].iloc[0]
        == coverage["run_id"].iloc[0]
        == context["run_id"].iloc[0]
        == output["run_id"]
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["model_sha256"] == "a" * 64
    assert summary["run_id"] == output["run_id"]
    assert summary["summary"]["predictions"] == 1


def test_latest_files_are_created(
    tmp_path,
    prediction_result,
) -> None:
    """Publishing stable files for BI consumers."""

    save_inference_result(
        prediction_result,
        model_sha256="a" * 64,
        output_dir=tmp_path,
    )

    for filename in (
        "predictions.csv",
        "coverage.csv",
        "account_context.csv",
        "summary.json",
    ):
        assert (tmp_path / f"latest_{filename}").is_file()


def test_new_run_preserves_history_and_updates_latest(
    tmp_path,
    prediction_result,
) -> None:
    """Preserving historical runs while replacing latest results."""

    first = save_inference_result(
        prediction_result,
        model_sha256="a" * 64,
        output_dir=tmp_path,
    )

    # Simulating a different valid prediction in the next run.
    second_result = {
        **prediction_result,
        "predictions": [
            {
                **prediction_result["predictions"][0],
                "predicted_class": 1,
                "predicted_state_name": "elevated_risk",
                "prob_normal": 0.1,
                "prob_elevated_risk": 0.8,
                "prob_liquidity_stress": 0.06,
                "prob_delinquency_risk": 0.04,
            }
        ],
    }

    second = save_inference_result(
        second_result,
        model_sha256="a" * 64,
        output_dir=tmp_path,
    )

    assert first["run_id"] != second["run_id"]

    # Checking that the first run remains unchanged.
    historical = pd.read_csv(tmp_path / "runs" / first["run_id"] / "predictions.csv")

    assert historical.loc[0, "predicted_class"] == 0

    # Checking that the latest file reflects the second run.
    latest = pd.read_csv(tmp_path / "latest_predictions.csv")

    assert latest.loc[0, "predicted_class"] == 1
    assert latest.loc[0, "run_id"] == second["run_id"]

    latest_summary = json.loads(
        (tmp_path / "latest_summary.json").read_text(encoding="utf-8")
    )

    assert latest_summary["run_id"] == second["run_id"]


def test_invalid_result_is_not_persisted(
    tmp_path,
    prediction_result,
) -> None:
    """Rejecting invalid API responses before writing files."""

    invalid = {
        **prediction_result,
        "sequence_length": 0,
    }

    with pytest.raises(ValueError):
        save_inference_result(
            invalid,
            model_sha256="a" * 64,
            output_dir=tmp_path,
        )

    assert not (tmp_path / "runs").exists()


def test_empty_predictions_preserve_csv_headers(
    tmp_path,
    prediction_result,
) -> None:
    """Supporting runs where no account has sufficient context."""

    result = {
        **prediction_result,
        "predictions": [],
        "summary": {
            **prediction_result["summary"],
            "accounts_eligible": 0,
            "accounts_insufficient_context": 1,
            "predictions": 0,
            "dates_insufficient_context": 7,
        },
        "account_context": [
            {
                **prediction_result["account_context"][0],
                "status": "insufficient_context",
                "available_observations": 7,
                "required_observations": 8,
                "missing_for_minimum": 1,
                "eligible_endpoints": 0,
            }
        ],
        "sequence_length": 8,
        "coverage": [
            {**record, "status": "insufficient_context"}
            for record in prediction_result["coverage"]
        ],
    }

    save_inference_result(
        result,
        model_sha256="a" * 64,
        output_dir=tmp_path,
    )

    predictions = pd.read_csv(tmp_path / "latest_predictions.csv")

    assert predictions.empty
    assert "run_id" in predictions.columns
    assert "predicted_class" in predictions.columns
