"""Running reproducible financial time-series LSTM training workflows.

Supporting notebook/Databricks table queries and caller-supplied Gold data.
Keeping each run's checkpoint, preprocessing metadata, and learning curves
in a separate directory so inference cannot mix artifacts across runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from finance_ml.config.training_config import (
    TRAINING_CONFIG as CONFIG,
    validate_training_config as _validate_config,
)
from finance_ml.data.query import (
    query_with_spark,
    query_with_sql_connector,
    query_with_sqlalchemy,
)
from finance_ml.models.experiment import run_lstm_experiment
from finance_ml.models.local_bundle import publish_local_bundle
from finance_ml.models.registry import register_logged_model
from finance_ml.models.tracking import log_sequence_experiment
from finance_ml.pipelines.model_dataset_pipeline import prepare_model_dataset


def _acquire_gold_data(*, config: Mapping[str, Any], spark: Any) -> Any:
    """Query a Gold table and return a Spark DataFrame for preprocessing."""
    query = f"SELECT * FROM {config['source_table']}"
    method = config["query_method"]

    if method == "spark":
        if spark is None:
            raise ValueError("A SparkSession is required for query_method='spark'")
        return query_with_spark(spark, query)

    # Connector and SQLAlchemy supply pandas; preprocessing still needs Spark.
    pandas_df = (
        query_with_sql_connector(query)
        if method == "connector"
        else query_with_sqlalchemy(query)
    )
    if pandas_df.empty:
        raise ValueError("The source table returned no rows")
    if spark is None:
        raise ValueError(
            "A SparkSession is required to prepare the training dataset; "
            "the connector/SQLAlchemy methods supply pandas data only"
        )
    return spark.createDataFrame(pandas_df)


def _json_safe(value: Any) -> Any:
    """Convert fitted preprocessing metadata to portable JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Cannot serialize preprocessing value: {type(value).__name__}")


def _create_run_artifact_dir(base_dir: str | Path) -> Path:
    """Create a new, non-reused directory for every training run."""
    root = Path(base_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root / f"run_{run_id}_{uuid4().hex[:10]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_preprocessing_metadata(
    dataset: Mapping[str, Any],
    config: Mapping[str, Any],
    artifact_dir: Path,
    *,
    training_input: str = "databricks_table",
) -> Path:
    """Persist training-fitted transformations for a future inference run."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_table": config["source_table"],
        # Distinguishing configured table name from actual Gold data source.
        "training_input": training_input,
        "sequence_length": config["sequence_length"],
        "stride": config["stride"],
        "boundary_width": config["boundary_width"],
        "n_classes": config["n_classes"],
        "final_feature_columns": dataset["final_feature_columns"],
        "model_columns": dataset["model_columns"],
        "medians": dataset["medians"],
        "category_levels": dataset["category_levels"],
        "scaler_parameters": dataset["scaler_parameters"],
        "scaling_method": dataset["scaling_method"],
        "feature_schema_profile": config["feature_schema_profile"],
        "train_end": dataset["train_end"],
        "validation_end": dataset["validation_end"],
    }
    output_path = artifact_dir / "preprocessing_metadata.json"
    output_path.write_text(
        json.dumps(_json_safe(metadata), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def run_model_pipeline(
    config: Mapping[str, Any] | None = None,
    *,
    spark: Any = None,
    gold_df: Any = None,
) -> dict[str, Any]:
    """Query or accept Gold, preprocess, train, and optionally log/register.

    All query modes use Spark-based preprocessing. Supplying ``gold_df``
    bypasses data querying, as in synthetic training. Each call creates an
    isolated artifact directory. A local model bundle is published only
    when a Keras checkpoint was requested and training completed.
    """
    options = {**CONFIG, **(dict(config) if config is not None else {})}
    _validate_config(options)

    supplied_gold = gold_df is not None
    if gold_df is None:
        gold_df = _acquire_gold_data(config=options, spark=spark)

    dataset = prepare_model_dataset(
        gold_df,
        sequence_length=options["sequence_length"],
        stride=options["stride"],
        scaling_method=options["scaling_method"],
        train_fraction=options["train_fraction"],
        validation_fraction=options["validation_fraction"],
        feature_schema_profile=options["feature_schema_profile"],
    )
    feature_columns = dataset["final_feature_columns"]
    n_features = len(feature_columns)
    if not n_features:
        raise ValueError("The prepared dataset contains no model features")

    artifact_dir = _create_run_artifact_dir(options["artifact_dir"])
    metadata_path = _save_preprocessing_metadata(
        dataset, options, artifact_dir,
        training_input="supplied_gold_dataframe" if supplied_gold else "databricks_table",
    )
    is_final = options["run_mode"] == "final"
    parameters = {
        key: str(value) if isinstance(value, tuple) else value
        for key, value in options.items()
        if key not in {"mlflow_experiment_name", "registered_model_name"}
        and value is not None
    }
    parameters["n_features"] = n_features
    parameters["preprocessing_metadata_file"] = metadata_path.name

    bundle_manifest_path: Path | None = None

    def _train():
        nonlocal bundle_manifest_path
        result = run_lstm_experiment(
            x_train=dataset["X_train"],
            y_train=dataset["y_train"],
            metadata_train=dataset["train_metadata"],
            x_validation=dataset["X_validation"],
            y_validation=dataset["y_validation"],
            metadata_validation=dataset["validation_metadata"],
            x_test=dataset["X_test"],
            y_test=dataset["y_test"],
            metadata_test=dataset["test_metadata"],
            sequence_length=options["sequence_length"],
            n_features=n_features,
            n_classes=options["n_classes"],
            lstm_units=tuple(options["lstm_units"]),
            dropout_rate=options["dropout_rate"],
            learning_rate=options["learning_rate"],
            boundary_width=options["boundary_width"],
            epochs=options["epochs"],
            batch_size=options["batch_size"],
            patience=options["patience"],
            verbose=options["verbose"],
            reduce_lr_on_plateau=options["reduce_lr_on_plateau"],
            lr_factor=options["lr_factor"],
            lr_patience=options["lr_patience"],
            min_lr=options["min_lr"],
            save_best_model=options["save_keras_checkpoint"],
            evaluate_test=is_final,
            seed=options["seed"],
            artifact_dir=artifact_dir,
        )
        if options["save_keras_checkpoint"]:
            # Publishing the manifest last: incomplete training is not loadable.
            bundle_manifest_path = publish_local_bundle(
                artifact_dir, model=result.model
            )
        return result

    model_uri = None
    registered_version = None
    if options["log_to_mlflow"]:
        import mlflow

        if options["mlflow_experiment_name"]:
            mlflow.set_experiment(options["mlflow_experiment_name"])
        if mlflow.active_run() is not None:
            raise RuntimeError("Close the active MLflow run before starting the pipeline")
        with mlflow.start_run(run_name=f"LSTM_{options['run_mode']}"):
            result = _train()
            mlflow.set_tag("model_type", "LSTM")
            mlflow.set_tag("selection_stage", options["run_mode"])
            if is_final:
                mlflow.set_tag("final_model", "true")
            mlflow.log_artifact(str(metadata_path), artifact_path="training_artifacts")
            if bundle_manifest_path is not None:
                mlflow.log_artifact(
                    str(bundle_manifest_path), artifact_path="training_artifacts"
                )
            model_uri = log_sequence_experiment(
                result=result,
                parameters=parameters,
                log_model=is_final,
                model_name=options["mlflow_model_name"],
                model_input_example=dataset["X_validation"][:1] if is_final else None,
            )
    else:
        result = _train()

    if options["register_in_uc"]:
        if model_uri is None:
            raise RuntimeError("No logged MLflow model URI available for registration")
        registered_version = register_logged_model(
            model_uri=model_uri,
            registered_model_name=options["registered_model_name"],
        )

    print("Query method:", options["query_method"])
    print("Training input:", "provided Gold" if supplied_gold else "Databricks table")
    print("Mode:", options["run_mode"])
    print("Model features:", n_features)
    print("Validation macro-F1:", result.validation_evaluation["macro_f1"])
    if result.test_evaluation is not None:
        print("Test macro-F1:", result.test_evaluation["macro_f1"])
    print("MLflow model URI:", model_uri)
    print("UC model version:", getattr(registered_version, "version", None))
    if bundle_manifest_path is not None:
        print("Local model bundle:", artifact_dir)

    return {
        "experiment": result,
        "model_uri": model_uri,
        "registered_version": registered_version,
        "metadata_path": metadata_path,
        "feature_columns": feature_columns,
        "bundle_dir": artifact_dir if bundle_manifest_path is not None else None,
        "bundle_manifest_path": bundle_manifest_path,
    }


if __name__ == "__main__":
    run_model_pipeline(spark=globals().get("spark"))
