"""Defining validated HTTP request and response schemas."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from typing import Literal


class GoldObservation(BaseModel):
    """Representing one engineered account-day observation."""

    model_config = ConfigDict(extra="forbid")

    date: date
    features: dict[str, float | str | None] = Field(min_length=1)


class PredictionRequest(BaseModel):
    """Representing chronological observations for one account."""

    model_config = ConfigDict(extra="forbid")

    account_id: str = Field(min_length=1)
    observations: list[GoldObservation] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dates(self):
        """Requiring strictly increasing observation dates."""
        dates = [observation.date for observation in self.observations]

        if any(later <= earlier for earlier, later in zip(dates, dates[1:])):
            raise ValueError("Observation dates must be strictly increasing")

        return self


class PredictionResponse(BaseModel):
    """Representing the latest account-day prediction."""

    account_id: str
    date: date
    predicted_class: int
    predicted_state_name: str
    probabilities: dict[str, float]


class UploadValidationResponse(BaseModel):
    """Describing the result of validating an uploaded CSV."""

    status: Literal["valid"]
    filename: str
    observations: int = Field(ge=0)
    accounts: int = Field(ge=0)
    duplicate_account_dates: int = Field(ge=0)


class AccountContextReport(BaseModel):
    """Describing sequence availability for one account."""

    account_id: str
    available_observations: int = Field(ge=0)
    required_observations: int = Field(ge=1)
    missing_for_minimum: int = Field(ge=0)
    eligible_endpoints: int = Field(ge=0)
    missing_calendar_days: int = Field(ge=0)
    status: Literal["eligible", "insufficient_context"]


class PredictionWarning(BaseModel):
    """Describing an account-level inference warning."""

    account_id: str
    code: Literal["insufficient_context", "calendar_gaps"]
    message: str


class PredictionCoverage(BaseModel):
    """Describing inference availability for one account/date."""

    account_id: str
    date: date
    status: Literal["predicted", "insufficient_context"]


class AccountRiskPrediction(BaseModel):
    """Describing one predicted account-day risk state."""

    account_id: str
    date: date

    predicted_class: int = Field(ge=0, le=3)
    predicted_state_name: Literal[
        "normal",
        "elevated_risk",
        "liquidity_stress",
        "delinquency_risk",
    ]

    prob_normal: float = Field(ge=0, le=1)
    prob_elevated_risk: float = Field(ge=0, le=1)
    prob_liquidity_stress: float = Field(ge=0, le=1)
    prob_delinquency_risk: float = Field(ge=0, le=1)


class PredictionSummary(BaseModel):
    """Summarizing account-level and date-level inference coverage."""

    accounts_received: int = Field(ge=0)
    accounts_eligible: int = Field(ge=0)
    accounts_insufficient_context: int = Field(ge=0)
    observations_after_deduplication: int = Field(ge=0)
    predictions: int = Field(ge=0)
    dates_insufficient_context: int = Field(ge=0)


class UploadPredictionResponse(BaseModel):
    """Describing the complete CSV-upload inference response."""

    data_source: Literal["uploaded_csv"]
    cohort_status: Literal["unverified_external_upload"]
    sequence_length: int = Field(ge=1)

    summary: PredictionSummary
    warnings: list[PredictionWarning]
    account_context: list[AccountContextReport]
    coverage: list[PredictionCoverage]
    predictions: list[AccountRiskPrediction]


class PersistedUploadPredictionResponse(UploadPredictionResponse):
    """Reporting predictions with their persisted inference-run identity."""

    run_id: str = Field(min_length=1)
    created_at_utc: datetime
