"""Defining the FastAPI application and its HTTP endpoints."""

from __future__ import annotations

import os
import pandas as pd
import logging

from io import BytesIO
from typing import Any
from fastapi import File, UploadFile
from starlette.concurrency import run_in_threadpool

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request

from pathlib import Path
from dotenv import load_dotenv

from finance_ml import __version__
from finance_ml.api.model_service import ModelService

from finance_ml.api.csv_validation import read_account_csv
from finance_ml.api.spark_runtime import SparkRuntime

from finance_ml.api.schemas import (
    UploadValidationResponse,
    UploadPredictionResponse,
    PersistedUploadPredictionResponse,
)

from finance_ml.api.result_store import save_inference_result


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Managing application-wide model and Spark resources."""

    # Initializing lightweight Spark resource management.
    app.state.spark_runtime = SparkRuntime()
    
    # Loading persistent local configuration when available.
    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(dotenv_path=project_root / ".env")

    # Loading the verified Keras model once at startup.
    bundle_dir = os.getenv("FINANCE_ML_API_BUNDLE_DIR")

    app.state.model_service = (
        ModelService.from_directory(bundle_dir) if bundle_dir else None
    )

    try:
        yield
    finally:
        # Releasing owned Spark resources during shutdown.
        try:
            app.state.spark_runtime.close()
        finally:
            app.state.model_service = None


# Creating the API application.
app = FastAPI(
    title="Financial Time-Series ML API",
    version=__version__,
    description="Model-serving API for financial time-series risk classification.",
    lifespan=lifespan,
)


@app.get("/health", tags=["Operations"])
def health() -> dict[str, str]:
    """Reporting API process liveness."""
    return {"status": "ok"}


@app.get("/ready", tags=["Operations"])
def ready(request: Request) -> dict[str, str]:
    """Checking whether the model is available for inference."""
    service = getattr(request.app.state, "model_service", None)

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return {"status": "ready"}


@app.get("/model", tags=["Model"])
def model_info(request: Request) -> dict[str, str | int]:
    """Reporting information about the loaded model."""
    service = getattr(request.app.state, "model_service", None)

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return service.describe()


# Limiting the CSV payload read into application memory.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB


async def _read_validated_upload(file: UploadFile) -> pd.DataFrame:
    """Reading and validating bounded raw account-day CSV uploads."""

    try:
        # Validating the uploaded file extension.
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=415,
                detail="Upload a .csv file.",
            )

        # Bounding the amount of uploaded data read into memory.
        content = await file.read(MAX_UPLOAD_BYTES + 1)

        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Uploaded CSV exceeds the 10 MiB limit.",
            )

        if not content:
            raise HTTPException(
                status_code=422,
                detail="Uploaded CSV is empty.",
            )

        # Reusing the existing parser without blocking the event loop.
        try:
            return await run_in_threadpool(
                read_account_csv,
                BytesIO(content),
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

    finally:
        # Releasing the temporary uploaded file.
        await file.close()


@app.post(
    "/uploads/validate",
    tags=["Uploads"],
    response_model=UploadValidationResponse,
)
async def validate_csv_upload(
    file: UploadFile = File(...),
) -> dict[str, str | int]:
    """Validating uploaded financial observations without running inference."""

    frame = await _read_validated_upload(file)

    duplicate_count = int(frame.duplicated(["account_id", "date"]).sum())

    return {
        "status": "valid",
        "filename": file.filename,
        "observations": len(frame),
        "accounts": int(frame["account_id"].nunique()),
        "duplicate_account_dates": duplicate_count,
    }


@app.post(
    "/predict/upload",
    tags=["Predictions"],
    response_model=PersistedUploadPredictionResponse,
)
async def predict_csv_upload(
    request: Request,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Predicting financial risk states from uploaded raw CSV data."""

    # Checking that the trained model is available.
    service = getattr(
        request.app.state,
        "model_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    # Checking that the Spark runtime is initialized.
    runtime = getattr(
        request.app.state,
        "spark_runtime",
        None,
    )

    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail="Inference runtime is unavailable.",
        )

    # Reading and validating the raw financial observations.
    frame = await _read_validated_upload(file)

    # Executing inference and persisting the results, while protecting internal error details.
    # Executing blocking Spark and Keras operations in a worker thread.
    try:
        result = await run_in_threadpool(
            runtime.predict,
            frame,
            service,
        )

        # Recording the predictions and verified model identity.
        stored = await run_in_threadpool(
            save_inference_result,
            result,
            model_sha256=service.bundle.manifest["model"]["sha256"],
        )

    except Exception as exc:
        logger.exception("CSV inference or result persistence failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed.",
        ) from exc

    # Returning inference results with their persisted run identity.
    return {
        **result,
        "run_id": stored["run_id"],
        "created_at_utc": stored["created_at_utc"],
    }
