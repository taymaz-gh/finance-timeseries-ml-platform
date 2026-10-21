"""Testing the FastAPI health-check endpoint."""

from fastapi.testclient import TestClient

from finance_ml.api.main import app


def test_health_endpoint() -> None:
    """Checking the HTTP status and JSON response."""
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
