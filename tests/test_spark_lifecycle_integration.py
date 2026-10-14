"""Testing reuse of an actual externally managed local Spark session."""

from __future__ import annotations

from finance_ml.config.training_config import TRAINING_CONFIG
from finance_ml.pipelines.training_runner import run_training


def test_runner_preserves_real_external_spark(spark) -> None:
    """Keeping an existing Spark session usable after training dispatch."""
    result = run_training(
        TRAINING_CONFIG,
        spark=spark,
        pipeline_fn=lambda *, config, spark: {
            "count": spark.range(3).count(),
        },
    )

    assert result["count"] == 3

    # Checking that the supplied session is still usable.
    assert spark.range(2).count() == 2
