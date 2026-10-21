"""Testing validation of API prediction requests and responses."""

from datetime import date

import pytest
from pydantic import ValidationError

from finance_ml.api.schemas import PredictionRequest, PredictionResponse


def test_valid_prediction_request() -> None:
    """Accepting chronological Gold observations."""
    request = PredictionRequest.model_validate(
        {
            "account_id": "ACC_00021",
            "observations": [
                {
                    "date": "2026-03-01",
                    "features": {
                        "balance": 1200.0,
                        "account_type": "checking",
                    },
                },
                {
                    "date": "2026-03-02",
                    "features": {
                        "balance": 1250.0,
                        "account_type": "checking",
                    },
                },
            ],
        }
    )

    assert request.account_id == "ACC_00021"
    assert request.observations[0].date == date(2026, 3, 1)
    assert request.observations[1].features["balance"] == 1250.0


@pytest.mark.parametrize(
    "dates",
    [
        ["2026-03-02", "2026-03-01"],
        ["2026-03-02", "2026-03-02"],
    ],
)
def test_reject_non_increasing_dates(dates: list[str]) -> None:
    """Rejecting unsorted or duplicated observation dates."""
    with pytest.raises(ValidationError, match="strictly increasing"):
        PredictionRequest.model_validate(
            {
                "account_id": "ACC_00021",
                "observations": [
                    {"date": day, "features": {"balance": 1000.0}} for day in dates
                ],
            }
        )


def test_reject_empty_features() -> None:
    """Rejecting observations without feature values."""
    with pytest.raises(ValidationError):
        PredictionRequest.model_validate(
            {
                "account_id": "ACC_00021",
                "observations": [{"date": "2026-03-01", "features": {}}],
            }
        )


def test_prediction_response() -> None:
    """Validating the structured inference response."""
    response = PredictionResponse(
        account_id="ACC_00021",
        date=date(2026, 3, 2),
        predicted_class=0,
        predicted_state_name="normal",
        probabilities={
            "normal": 0.8,
            "elevated_risk": 0.1,
            "liquidity_stress": 0.08,
            "delinquency_risk": 0.02,
        },
    )

    assert response.predicted_class == 0
    assert sum(response.probabilities.values()) == pytest.approx(1.0)
