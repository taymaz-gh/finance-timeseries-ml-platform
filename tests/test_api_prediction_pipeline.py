"""Testing uploaded-account inference orchestration."""

from types import SimpleNamespace
from unittest.mock import Mock
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from finance_ml.api import prediction_pipeline
from finance_ml.api.coverage import assess_account_context


class FakeLSTM:
    """Simulating a four-class sequence-to-sequence classifier."""

    def __init__(self):
        self.calls = []

    def predict(self, x, **kwargs):
        """Returning deterministic class probabilities."""

        self.calls.append(x.copy())

        probabilities = np.zeros(
            (len(x), x.shape[1], 4),
            dtype=np.float32,
        )

        probabilities[..., 1] = 1.0

        return probabilities


def make_model_rows(lengths: dict[str, int]) -> pd.DataFrame:
    """Generating model-ready observations for several accounts."""

    records = []

    for account_id, n_days in lengths.items():

        for index in range(n_days):

            # Introducing one calendar gap for ACC_B.
            day_offset = index

            if account_id == "ACC_B" and index >= 3:
                day_offset += 1

            records.append(
                {
                    "account_id": account_id,
                    "date": datetime(2026, 1, 1) + timedelta(days=day_offset),
                    "feature_1": float(index),
                }
            )

    return pd.DataFrame(records)


def setup_pipeline(monkeypatch, rows, sequence_length):
    """Mocking expensive transformations while retaining real inference."""

    metadata = {
        "sequence_length": sequence_length,
        "final_feature_columns": ["feature_1"],
        "n_classes": 4,
    }

    model = FakeLSTM()

    service = SimpleNamespace(
        bundle=SimpleNamespace(
            model=model,
            metadata=metadata,
        )
    )

    report = assess_account_context(
        rows[["account_id", "date"]],
        sequence_length=sequence_length,
    )

    gold = object()

    mock_prepare = Mock(return_value=(gold, report))

    # Simulating a Spark DataFrame's toPandas() operation.
    transformed = Mock()
    transformed.toPandas.return_value = rows

    mock_preprocess = Mock(return_value=transformed)

    monkeypatch.setattr(
        prediction_pipeline,
        "prepare_uploaded_gold",
        mock_prepare,
    )

    monkeypatch.setattr(
        prediction_pipeline,
        "apply_saved_preprocessing",
        mock_preprocess,
    )

    return service, model, mock_prepare, mock_preprocess, gold


def test_multiple_accounts_partial_success(monkeypatch) -> None:
    """Predicting eligible accounts while reporting insufficient context."""

    rows = make_model_rows(
        {
            "ACC_A": 9,
            "ACC_B": 7,
            "ACC_C": 4,
        }
    )

    service, model, prepare, preprocess, gold = setup_pipeline(
        monkeypatch,
        rows,
        sequence_length=7,
    )

    spark = object()

    result = prediction_pipeline.predict_uploaded_accounts(
        rows,
        spark=spark,
        service=service,
    )

    summary = result["summary"]

    assert summary["accounts_received"] == 3
    assert summary["accounts_eligible"] == 2
    assert summary["accounts_insufficient_context"] == 1

    assert summary["observations_after_deduplication"] == 20
    assert summary["predictions"] == 4
    assert summary["dates_insufficient_context"] == 16

    assert len(result["predictions"]) == 4
    assert len(result["coverage"]) == 20

    # Checking the actual tensor passed to our fake LSTM.
    assert len(model.calls) == 1
    assert model.calls[0].shape == (4, 7, 1)

    # Checking prediction output and JSON-compatible dates.
    assert all(record["predicted_class"] == 1 for record in result["predictions"])

    assert all(
        record["predicted_state_name"] == "elevated_risk"
        for record in result["predictions"]
    )

    assert isinstance(result["predictions"][0]["date"], str)

    # Checking account-level warnings.
    warning_codes = {(item["account_id"], item["code"]) for item in result["warnings"]}

    assert ("ACC_C", "insufficient_context") in warning_codes
    assert ("ACC_B", "calendar_gaps") in warning_codes

    assert result["cohort_status"] == "unverified_external_upload"

    # Checking that existing pipeline stages are reused correctly.
    prepare.assert_called_once_with(
        rows,
        spark=spark,
        sequence_length=7,
    )

    preprocess.assert_called_once_with(
        gold,
        service.bundle.metadata,
    )


def test_all_accounts_insufficient(monkeypatch) -> None:
    """Returning coverage without calling the LSTM."""

    rows = make_model_rows(
        {
            "ACC_A": 3,
            "ACC_B": 2,
        }
    )

    service, model, *_ = setup_pipeline(
        monkeypatch,
        rows,
        sequence_length=7,
    )

    result = prediction_pipeline.predict_uploaded_accounts(
        rows,
        spark=object(),
        service=service,
    )

    assert result["summary"]["accounts_eligible"] == 0
    assert result["summary"]["predictions"] == 0

    assert result["predictions"] == []
    assert len(result["coverage"]) == 5

    assert all(item["status"] == "insufficient_context" for item in result["coverage"])

    # No complete sequence means no inference call.
    assert model.calls == []


@pytest.mark.parametrize(
    "sequence_length, expected_predictions",
    [
        (4, 5),
        (7, 2),
    ],
)
def test_model_defined_sequence_length(
    monkeypatch,
    sequence_length,
    expected_predictions,
) -> None:
    """Using saved model metadata instead of a hardcoded window length."""

    rows = make_model_rows({"ACC_A": 8})

    service, model, *_ = setup_pipeline(
        monkeypatch,
        rows,
        sequence_length=sequence_length,
    )

    result = prediction_pipeline.predict_uploaded_accounts(
        rows,
        spark=object(),
        service=service,
    )

    assert result["sequence_length"] == sequence_length

    assert result["summary"]["predictions"] == expected_predictions

    assert model.calls[0].shape == (
        expected_predictions,
        sequence_length,
        1,
    )


def test_preprocessing_failure_is_not_silently_ignored(
    monkeypatch,
) -> None:
    """Propagating feature-processing failures instead of reporting success."""

    rows = make_model_rows({"ACC_A": 8})

    service, model, _, preprocess, _ = setup_pipeline(
        monkeypatch,
        rows,
        sequence_length=7,
    )

    preprocess.side_effect = ValueError("Invalid saved preprocessing contract")

    with pytest.raises(ValueError, match="preprocessing contract"):
        prediction_pipeline.predict_uploaded_accounts(
            rows,
            spark=object(),
            service=service,
        )

    assert model.calls == []
