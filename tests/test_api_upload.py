"""Testing the FastAPI CSV-upload validation endpoint."""

from io import BytesIO

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from finance_ml.api.main import app
from finance_ml.data.generate_synthetic import generate_financial_timeseries


@pytest.fixture
def client(monkeypatch):
    """Starting the API without requiring a saved Keras model."""

    monkeypatch.delenv("FINANCE_ML_API_BUNDLE_DIR", raising=False)

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_csv() -> bytes:
    """Generating reproducible, unlabeled financial observations."""

    frame = generate_financial_timeseries(
        n_accounts=3,
        n_days=14,
        start_date="2026-01-01",
        random_seed=123,
    )

    frame = frame.drop(columns=["risk_state", "risk_state_name"])

    return frame.to_csv(index=False).encode("utf-8")


def upload(client, content: bytes, filename: str = "accounts.csv"):
    """Submitting CSV content through the HTTP upload interface."""

    return client.post(
        "/uploads/validate",
        files={"file": (filename, content, "text/csv")},
    )


def test_valid_csv_upload(client, sample_csv) -> None:
    """Accepting multiple accounts through the HTTP endpoint."""

    response = upload(client, sample_csv)

    assert response.status_code == 200
    assert response.json() == {
        "status": "valid",
        "filename": "accounts.csv",
        "observations": 42,
        "accounts": 3,
        "duplicate_account_dates": 0,
    }


def test_duplicate_account_dates(client, sample_csv) -> None:
    """Reporting duplicate records without silently removing them."""

    frame = pd.read_csv(BytesIO(sample_csv))

    frame = pd.concat(
        [frame, frame.iloc[[0]]],
        ignore_index=True,
    )

    response = upload(
        client,
        frame.to_csv(index=False).encode("utf-8"),
    )

    assert response.status_code == 200
    assert response.json()["duplicate_account_dates"] == 1


def test_missing_required_column(client, sample_csv) -> None:
    """Rejecting CSV data with incomplete raw features."""

    frame = pd.read_csv(BytesIO(sample_csv))
    frame = frame.drop(columns=["balance"])

    response = upload(
        client,
        frame.to_csv(index=False).encode("utf-8"),
    )

    assert response.status_code == 422
    assert "balance" in response.json()["detail"]


def test_target_leakage_rejected(client, sample_csv) -> None:
    """Rejecting target labels in uploaded inference data."""

    frame = pd.read_csv(BytesIO(sample_csv))
    frame["risk_state"] = 0

    response = upload(
        client,
        frame.to_csv(index=False).encode("utf-8"),
    )

    assert response.status_code == 422
    assert "target columns" in response.json()["detail"]


def test_invalid_file_extension(client, sample_csv) -> None:
    """Rejecting files without a CSV extension."""

    response = upload(
        client,
        sample_csv,
        filename="accounts.txt",
    )

    assert response.status_code == 415


def test_empty_file(client) -> None:
    """Rejecting empty uploaded files."""

    response = upload(client, b"")

    assert response.status_code == 422


def test_upload_size_limit(client, sample_csv, monkeypatch) -> None:
    """Enforcing the configured upload-size limit."""

    monkeypatch.setattr(
        "finance_ml.api.main.MAX_UPLOAD_BYTES",
        100,
    )

    response = upload(client, sample_csv)

    assert response.status_code == 413


def test_missing_file(client) -> None:
    """Requiring a file in the multipart request."""

    response = client.post("/uploads/validate")

    assert response.status_code == 422
