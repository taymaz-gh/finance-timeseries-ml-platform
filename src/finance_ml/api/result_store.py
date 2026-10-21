"""Persisting API inference results for reporting and visualization."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import pandas as pd

from finance_ml.api.schemas import (
    AccountContextReport,
    AccountRiskPrediction,
    PredictionCoverage,
    UploadPredictionResponse,
)


_STORE_LOCK = Lock()
DEFAULT_OUTPUT_DIR = Path("artifacts/api_inference")


def save_inference_result(
    result: dict[str, Any],
    *,
    model_sha256: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Saving an inference run and publishing its latest-result files."""

    # Validating and serializing the prediction response.
    validated = UploadPredictionResponse.model_validate(result)
    payload = validated.model_dump(mode="json")

    now = datetime.now(timezone.utc)
    run_id = f"run_{now.strftime('%Y%m%dT%H%M%S%fZ')}_" f"{uuid4().hex[:10]}"

    root = Path(output_dir)
    runs_root = root / "runs"

    # Defining the tabular outputs for Power BI and Streamlit.
    datasets = {
        "predictions.csv": (
            payload["predictions"],
            list(AccountRiskPrediction.model_fields),
        ),
        "coverage.csv": (
            payload["coverage"],
            list(PredictionCoverage.model_fields),
        ),
        "account_context.csv": (
            payload["account_context"],
            list(AccountContextReport.model_fields),
        ),
    }

    summary = {
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "model_sha256": model_sha256,
        "data_source": payload["data_source"],
        "cohort_status": payload["cohort_status"],
        "sequence_length": payload["sequence_length"],
        "summary": payload["summary"],
        "warnings": payload["warnings"],
    }

    # Serializing filesystem writes within the current API process.
    with _STORE_LOCK:
        runs_root.mkdir(parents=True, exist_ok=True)

        temporary_dir = runs_root / f".{run_id}.tmp"
        run_dir = runs_root / run_id
        temporary_dir.mkdir()

        try:
            # Writing separate BI-friendly tables.
            for filename, (records, columns) in datasets.items():
                rows = [{"run_id": run_id, **record} for record in records]

                pd.DataFrame(
                    rows,
                    columns=["run_id", *columns],
                ).to_csv(
                    temporary_dir / filename,
                    index=False,
                )

            # Recording inference provenance and coverage.
            (temporary_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False),
                encoding="utf-8",
            )

            # Publishing the complete historical run.
            temporary_dir.rename(run_dir)

            # Updating the stable paths used by dashboards.
            for filename in (*datasets, "summary.json"):
                source = run_dir / filename
                destination = root / f"latest_{filename}"
                temporary = root / f".{run_id}_{filename}.tmp"

                try:
                    shutil.copyfile(source, temporary)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)

        finally:
            # Removing unfinished historical-run staging files.
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)

    return {
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "run_dir": str(run_dir),
    }
