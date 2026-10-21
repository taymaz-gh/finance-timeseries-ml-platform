"""Testing the Streamlit dashboard without a live backend or Spark runtime."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from streamlit.testing.v1 import AppTest

import finance_ml.dashboard.app as dashboard


APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "finance_ml"
    / "dashboard"
    / "app.py"
)


# Allowing extra time for Streamlit under full-suite resource contention.
APP_TEST_TIMEOUT = 60

def make_response(
    status_code: int,
    body: dict,
    *,
    url: str = "http://127.0.0.1:8000/ready",
) -> httpx.Response:
    """Constructing realistic HTTP responses for mocked transports."""
    return httpx.Response(
        status_code,
        json=body,
        request=httpx.Request("GET", url),
    )


@pytest.fixture
def sample_result() -> dict:
    """Providing a compact schema-compatible API response for rendering."""
    return {
        "run_id": "run_example_001",
        "summary": {
            "accounts_received": 1,
            "predictions": 2,
            "dates_insufficient_context": 6,
        },
        "coverage": [
            {
                "account_id": "ACC_00001",
                "date": "2026-01-06",
                "status": "insufficient_context",
            },
            {
                "account_id": "ACC_00001",
                "date": "2026-01-07",
                "status": "predicted",
            },
            {
                "account_id": "ACC_00001",
                "date": "2026-01-08",
                "status": "predicted",
            },
        ],
        "predictions": [
            {
                "account_id": "ACC_00001",
                "date": "2026-01-07",
                "predicted_class": 0,
                "predicted_state_name": "normal",
                "prob_normal": 0.70,
                "prob_elevated_risk": 0.20,
                "prob_liquidity_stress": 0.08,
                "prob_delinquency_risk": 0.02,
            },
            {
                "account_id": "ACC_00001",
                "date": "2026-01-08",
                "predicted_class": 1,
                "predicted_state_name": "elevated_risk",
                "prob_normal": 0.15,
                "prob_elevated_risk": 0.65,
                "prob_liquidity_stress": 0.15,
                "prob_delinquency_risk": 0.05,
            },
        ],
    }


def test_streamlit_app_shows_upload_when_backend_ready(monkeypatch) -> None:
    """Rendering the actual Streamlit entrypoint with a simulated ready API."""

    def mock_get(url: str, **kwargs) -> httpx.Response:
        if url.endswith("/ready"):
            return make_response(200, {"status": "ready"}, url=url)
        if url.endswith("/model"):
            return make_response(
                200,
                {
                    "model_source": "local",
                    "sequence_length": 7,
                    "n_features": 41,
                    "n_classes": 4,
                    "model_sha256": "a" * 64,
                },
                url=url,
            )
        raise AssertionError(f"Unexpected backend GET: {url}")

    monkeypatch.setattr(httpx, "get", mock_get)

    app = AppTest.from_file(APP_PATH, default_timeout=APP_TEST_TIMEOUT).run()

    assert not app.exception
    assert len(app.success) == 1
    assert len(app.file_uploader) == 1
    assert len(app.info) == 1  # Prompt to upload before inference


def test_streamlit_app_handles_unreachable_backend(monkeypatch) -> None:
    """Showing a useful error instead of requiring a running API server."""

    def disconnected(url: str, **kwargs):
        raise httpx.ConnectError("Connection refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", disconnected)

    app = AppTest.from_file(APP_PATH, default_timeout=APP_TEST_TIMEOUT).run()

    assert not app.exception
    assert len(app.error) == 1
    assert "Cannot connect" in app.error[0].value


def test_streamlit_app_handles_unready_model(monkeypatch) -> None:
    """Explaining model-not-loaded status to the dashboard user."""

    def mock_get(url: str, **kwargs) -> httpx.Response:
        return make_response(503, {"detail": "Model is not loaded."}, url=url)

    monkeypatch.setattr(httpx, "get", mock_get)

    app = AppTest.from_file(APP_PATH, default_timeout=APP_TEST_TIMEOUT).run()

    assert not app.exception
    assert len(app.error) == 1
    assert "no model" in app.error[0].value.lower()


def test_request_csv_uploads_bytes_and_returns_response(monkeypatch) -> None:
    """Sending the correct multipart file field to the API."""
    captured = {}

    def mock_post(url: str, **kwargs) -> httpx.Response:
        captured["url"] = url
        captured["files"] = kwargs["files"]
        captured["timeout"] = kwargs["timeout"]
        return httpx.Response(
            200,
            json={"status": "valid", "observations": 42, "accounts": 3},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", mock_post)

    result = dashboard.request_csv(
        "/uploads/validate", "sample_accounts.csv", b"csv-contents"
    )

    assert captured["url"] == f"{dashboard.API_URL}/uploads/validate"
    assert captured["files"] == {
        "file": ("sample_accounts.csv", b"csv-contents", "text/csv")
    }
    assert captured["timeout"] is not None
    assert result["observations"] == 42


def test_request_csv_exposes_safe_validation_failure(monkeypatch) -> None:
    """Displaying backend HTTP errors rather than treating them as success."""

    def mock_post(url: str, **kwargs) -> httpx.Response:
        return httpx.Response(
            422,
            json={"detail": "Invalid raw CSV schema"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", mock_post)

    with pytest.raises(RuntimeError, match="HTTP 422.*Invalid raw CSV schema"):
        dashboard.request_csv("/uploads/validate", "invalid.csv", b"invalid")


def test_request_csv_reports_connection_failure(monkeypatch) -> None:
    """Reporting network failures in user-understandable language."""

    def disconnected(url: str, **kwargs):
        raise httpx.ConnectError("Connection refused", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", disconnected)

    with pytest.raises(RuntimeError, match="Unable to communicate"):
        dashboard.request_csv("/predict/upload", "accounts.csv", b"csv")


def test_render_report_uses_real_response_for_charts_and_download(
    monkeypatch, sample_result
) -> None:
    """Rendering three charts and a CSV download from the API result."""

    fake_st = Mock()
    fake_st.columns.side_effect = lambda count: [Mock() for _ in range(count)]
    fake_st.selectbox.return_value = "ACC_00001"
    monkeypatch.setattr(dashboard, "st", fake_st)

    dashboard.render_report(sample_result)

    assert fake_st.plotly_chart.call_count == 3
    assert fake_st.dataframe.call_count == 1
    assert fake_st.download_button.call_count == 1
    download_kwargs = fake_st.download_button.call_args.kwargs
    assert b"ACC_00001" in download_kwargs["data"]
    assert b"prob_elevated_risk" in download_kwargs["data"]
    assert download_kwargs["file_name"] == "run_example_001_predictions.csv"


def test_render_report_explains_empty_prediction_set(monkeypatch, sample_result) -> None:
    """Providing a coverage warning when history is insufficient."""
    sample_result["predictions"] = []
    sample_result["summary"]["predictions"] = 0

    fake_st = Mock()
    fake_st.columns.side_effect = lambda count: [Mock() for _ in range(count)]
    monkeypatch.setattr(dashboard, "st", fake_st)

    dashboard.render_report(sample_result)

    fake_st.warning.assert_called_once()
    fake_st.plotly_chart.assert_not_called()
