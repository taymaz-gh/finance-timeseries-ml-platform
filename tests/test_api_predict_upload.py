"""Testing the HTTP endpoint for uploaded financial predictions."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from finance_ml.api.main import app
from finance_ml.data.generate_synthetic import generate_financial_timeseries


def make_mock_prediction_response() -> dict:
    """Creating a valid, synthetic API prediction response."""

    accounts = [f"ACC_{i:05d}" for i in range(1, 4)]

    observations = [
        (account, f"2026-01-{day:02d}", day)
        for account in accounts
        for day in range(1, 15)
    ]

    return {
        "data_source": "uploaded_csv",
        "cohort_status": "unverified_external_upload",
        "sequence_length": 7,
        "summary": {
            "accounts_received": 3,
            "accounts_eligible": 3,
            "accounts_insufficient_context": 0,
            "observations_after_deduplication": 42,
            "predictions": 24,
            "dates_insufficient_context": 18,
        },
        "warnings": [],
        "account_context": [
            {
                "account_id": account,
                "available_observations": 14,
                "required_observations": 7,
                "missing_for_minimum": 0,
                "eligible_endpoints": 8,
                "missing_calendar_days": 0,
                "status": "eligible",
            }
            for account in accounts
        ],
        "coverage": [
            {
                "account_id": account,
                "date": date,
                "status": ("predicted" if day >= 7 else "insufficient_context"),
            }
            for account, date, day in observations
        ],
        "predictions": [
            {
                "account_id": account,
                "date": date,
                "predicted_class": 0,
                "predicted_state_name": "normal",
                "prob_normal": 1.0,
                "prob_elevated_risk": 0.0,
                "prob_liquidity_stress": 0.0,
                "prob_delinquency_risk": 0.0,
            }
            for account, date, day in observations
            if day >= 7
        ],
    }


@pytest.fixture
def sample_csv() -> bytes:
    """Generating reproducible unlabeled account observations."""

    frame = generate_financial_timeseries(
        n_accounts=3,
        n_days=14,
        start_date="2026-01-01",
        random_seed=123,
    )

    frame = frame.drop(columns=["risk_state", "risk_state_name"])

    return frame.to_csv(index=False).encode("utf-8")


@pytest.fixture
def configured_api(monkeypatch):
    """Starting FastAPI with mocked model, inference, and persistence."""

    monkeypatch.setenv(
        "FINANCE_ML_API_BUNDLE_DIR",
        "test-bundle",
    )

    fake_service = Mock()

    fake_service.bundle.manifest = {"model": {"sha256": "a" * 64}}

    with (
        patch(
            "finance_ml.api.main.ModelService.from_directory",
            return_value=fake_service,
        ) as mock_loader,
        patch(
            "finance_ml.api.spark_runtime.SparkRuntime.predict",
            return_value=make_mock_prediction_response(),
        ) as mock_predict,
        patch(
            "finance_ml.api.main.save_inference_result",
            return_value={
                "run_id": "run_test_001",
                "created_at_utc": "2026-10-21T12:00:00+00:00",
                "run_dir": "unused-test-path",
            },
        ) as mock_save,
        TestClient(app) as client,
    ):
        yield (
            client,
            fake_service,
            mock_loader,
            mock_predict,
            mock_save,
        )


def test_prediction_upload_success(
    configured_api,
    sample_csv,
) -> None:
    """Passing validated CSV observations to the inference runtime."""

    client, service, loader, predict, save = configured_api

    response = client.post(
        "/predict/upload",
        files={"file": ("accounts.csv", sample_csv, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()

    # Checking that the prediction response remains intact.
    expected = make_mock_prediction_response()

    for key, value in expected.items():
        assert body[key] == value

    # Checking that the persisted run identity is returned.
    assert body["run_id"] == "run_test_001"
    assert "created_at_utc" in body

    # Not exposing internal filesystem paths.
    assert "run_dir" not in body

    # Verifying that persistence was called once.
    save.assert_called_once()

    assert save.call_args.args[0] == expected
    assert save.call_args.kwargs["model_sha256"] == "a" * 64

    # Checking that the model was initialized once.
    loader.assert_called_once_with("test-bundle")

    # Checking that the inference runtime was invoked once.
    predict.assert_called_once()

    frame, received_service = predict.call_args.args

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 42
    assert frame["account_id"].nunique() == 3

    assert received_service is service

    # Preventing target labels from entering inference.
    assert "risk_state" not in frame.columns
    assert "risk_state_name" not in frame.columns


def test_prediction_requires_loaded_model(
    monkeypatch,
    sample_csv,
) -> None:
    """Returning HTTP 503 when no model is configured."""

    monkeypatch.delenv(
        "FINANCE_ML_API_BUNDLE_DIR",
        raising=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/predict/upload",
            files={"file": ("accounts.csv", sample_csv, "text/csv")},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded."


def test_prediction_rejects_invalid_csv(
    configured_api,
    sample_csv,
) -> None:
    """Rejecting invalid raw data before invoking inference."""

    client, _, _, predict, save = configured_api

    from io import BytesIO

    frame = pd.read_csv(BytesIO(sample_csv))
    frame = frame.drop(columns=["balance"])

    invalid_csv = frame.to_csv(index=False).encode("utf-8")

    response = client.post(
        "/predict/upload",
        files={"file": ("accounts.csv", invalid_csv, "text/csv")},
    )

    assert response.status_code == 422
    assert "balance" in response.json()["detail"]

    predict.assert_not_called()
    save.assert_not_called()


def test_prediction_rejects_non_csv(
    configured_api,
    sample_csv,
) -> None:
    """Rejecting unsupported files before executing inference."""

    client, _, _, predict, save = configured_api

    response = client.post(
        "/predict/upload",
        files={"file": ("accounts.txt", sample_csv, "text/plain")},
    )

    assert response.status_code == 415
    predict.assert_not_called()
    save.assert_not_called()


def test_prediction_failure_hides_internal_details(
    configured_api,
    sample_csv,
) -> None:
    """Returning a safe HTTP error when inference unexpectedly fails."""

    client, _, _, predict, save = configured_api

    # Simulating an internal processing failure.
    predict.side_effect = RuntimeError(
        "Internal connection details must remain private"
    )

    response = client.post(
        "/predict/upload",
        files={"file": ("accounts.csv", sample_csv, "text/csv")},
    )

    assert response.status_code == 500

    assert response.json() == {"detail": "Prediction could not be completed."}

    # Ensuring internal exception messages are not disclosed.
    assert "Internal connection details" not in response.text

    predict.assert_called_once()

    save.assert_not_called()
