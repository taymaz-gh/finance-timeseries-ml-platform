"""Testing FastAPI application lifespan and model readiness."""

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from finance_ml.api.main import app


def test_api_without_model(monkeypatch) -> None:
    """Checking API liveness when no model is configured."""

    # Explicitly disabling model loading, including .env fallback.
    monkeypatch.setenv("FINANCE_ML_API_BUNDLE_DIR", "")

    # Starting and stopping the FastAPI application.
    with TestClient(app) as client:

        health = client.get("/health")
        ready = client.get("/ready")
        model = client.get("/model")

        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        assert ready.status_code == 503
        assert ready.json()["detail"] == "Model is not loaded."

        assert model.status_code == 503


def test_api_loads_model_once(monkeypatch) -> None:
    """Checking model initialization, reuse, and metadata reporting."""

    # Configuring a simulated model-bundle directory.
    monkeypatch.setenv(
        "FINANCE_ML_API_BUNDLE_DIR",
        "test-bundle",
    )

    # Creating a simulated model service.
    fake_service = Mock()

    fake_service.describe.return_value = {
        "model_source": "local",
        "sequence_length": 7,
        "n_features": 41,
        "n_classes": 4,
        "model_sha256": "a" * 64,
    }

    # Replacing real model loading with our simulated service.
    with patch(
        "finance_ml.api.main.ModelService.from_directory",
        return_value=fake_service,
    ) as mock_loader:

        with TestClient(app) as client:

            ready = client.get("/ready")
            model = client.get("/model")

            assert ready.status_code == 200
            assert ready.json() == {"status": "ready"}

            assert model.status_code == 200
            assert model.json()["sequence_length"] == 7
            assert model.json()["n_features"] == 41
            assert model.json()["n_classes"] == 4

        # Verifying that model loading occurred only once.
        mock_loader.assert_called_once_with("test-bundle")

    # Checking application cleanup after shutdown.
    assert app.state.model_service is None
