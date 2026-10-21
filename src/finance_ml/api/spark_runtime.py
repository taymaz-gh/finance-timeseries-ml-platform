"""Managing reusable local Spark resources for API inference."""

from __future__ import annotations

from threading import Lock
from typing import Any

import pandas as pd

from finance_ml.api.model_service import ModelService
from finance_ml.api.prediction_pipeline import predict_uploaded_accounts


class SparkRuntime:
    """Managing one lazily initialized Spark session for API predictions."""

    def __init__(self) -> None:
        """Initializing resource references without starting Spark."""

        self._spark: Any = None
        self._owns_spark = False
        self._lock = Lock()

    def predict(
        self,
        frame: pd.DataFrame,
        service: ModelService,
    ) -> dict[str, Any]:
        """Running inference with a reusable Spark session."""

        # Serializing resource access and local model inference.
        with self._lock:

            # Starting Spark only when the first prediction arrives.
            if self._spark is None:
                from finance_ml.pipelines.training_runner import (
                    _create_local_spark_session,
                )

                self._spark, self._owns_spark = _create_local_spark_session()

            # Reusing our existing inference orchestration.
            return predict_uploaded_accounts(
                frame,
                spark=self._spark,
                service=service,
            )

    def close(self) -> None:
        """Releasing Spark only when this runtime created it."""

        with self._lock:
            try:
                if self._spark is not None and self._owns_spark:
                    self._spark.stop()
            finally:
                self._spark = None
                self._owns_spark = False
