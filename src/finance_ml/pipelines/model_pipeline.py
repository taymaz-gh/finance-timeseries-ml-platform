"""Running reproducible financial time-series LSTM training workflows.

Edit CONFIG below for interactive use, or import run_model_pipeline(config=...)
from a notebook, scheduled job, or another Python module. Running this module
never starts training merely by importing it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from finance_ml.data.query import (
    query_with_spark,
    query_with_sql_connector,
    query_with_sqlalchemy,
)
from finance_ml.models.experiment import run_lstm_experiment
from finance_ml.models.registry import register_logged_model
from finance_ml.models.tracking import log_sequence_experiment
from finance_ml.pipelines.model_dataset_pipeline import prepare_model_dataset


# ============================================================
# USER CONFIGURATION (edit here; no secrets should be stored)
# ============================================================
CONFIG: dict[str, Any] = {
    "query_method": "spark",  # "spark", "connector", "sqlalchemy"
    "source_table": "finance_ml.gold.account_features",
    "sequence_length": 7,
    "stride": 1,
    "scaling_method": "mad",
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "n_classes": 4,
    "boundary_width": 2,
    "lstm_units": (64,),
    "dropout_rate": 0.30,
    "learning_rate": 1e-3,
    "epochs": 50,
    "batch_size": 256,
    "patience": 5,
    "seed": 42,
    "verbose": 1,
    "reduce_lr_on_plateau": False,
    "lr_factor": 0.5,
    "lr_patience": 2,
    "min_lr": 1e-6,
    "run_mode": "development",  # "development" or "final"
    "save_keras_checkpoint": True,
    "log_to_mlflow": True,
    "register_in_uc": False,
    "mlflow_experiment_name": None,  # Set to your Databricks experiment path
    "mlflow_model_name": "lstm_model",
    "registered_model_name": "finance_ml.models.lstm_risk_classifier",
    "artifact_dir": "artifacts/model_pipeline",
}


def _validate_config(config: Mapping[str, Any]) -> None:
    """Reject unsupported combinations before querying or training."""
    if config["query_method"] not in {"spark", "connector", "sqlalchemy"}:
        raise ValueError("query_method must be spark, connector, or sqlalchemy")
    if config["run_mode"] not in {"development", "final"}:
        raise ValueError("run_mode must be development or final")
    if not config["source_table"] or not all(
        part.replace("_", "").isalnum()
        for part in config["source_table"].split(".")
    ) or len(config["source_table"].split(".")) != 3:
        raise ValueError("source_table must be catalog.schema.table")
    if config["register_in_uc"] and not config["log_to_mlflow"]:
        raise ValueError("UC registration requires MLflow model logging")
    if config["register_in_uc"] and config["run_mode"] != "final":
        raise ValueError("UC registration is only supported for final runs")
    if config["run_mode"] == "final" and not config["log_to_mlflow"]:
        # A final evaluation need not be registered, but for this workflow
        # all final runs are tracked so their test metrics remain auditable.
        raise ValueError("Final runs must be logged to MLflow")
    if config["sequence_length"] <= 2 * config["boundary_width"]:
        raise ValueError("sequence_length must exceed twice boundary_width")


def _acquire_gold_data(*, config: Mapping[str, Any], spark: Any) -> Any:
    """Query a Gold table and return a Spark DataFrame for preprocessing."""
    query = f"SELECT * FROM {config['source_table']}"
    method = config["query_method"]

    if method == "spark":
        if spark is None:
            raise ValueError("A SparkSession is required for query_method='spark'")
        return query_with_spark(spark, query)

    # Both remote SQL methods return pandas; the existing training
    # preprocessing pipeline requires Spark, so normalize here.
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


def _save_preprocessing_metadata(dataset: Mapping[str, Any], config: Mapping[str, Any], artifact_dir: Path) -> Path:
    """Persist training-fitted transformations required for future inference."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_table": config["source_table"],
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
) -> dict[str, Any]:
    """Query, preprocess, train, evaluate, optionally log/register an LSTM.

    `spark` can be the active Databricks SparkSession. For all query modes,
    Spark is needed by the existing preprocessing implementation. Final
    mode evaluates the held-out test partition and may register in UC;
    development mode uses validation metrics only.

    Returns a dictionary with experiment result, MLflow model URI, optional
    registered version, and preprocessing metadata artifact location.
    """
    options = {**CONFIG, **(dict(config) if config is not None else {})}
    _validate_config(options)

    gold_df = _acquire_gold_data(config=options, spark=spark)
    dataset = prepare_model_dataset(
        gold_df,
        sequence_length=options["sequence_length"],
        stride=options["stride"],
        scaling_method=options["scaling_method"],
        train_fraction=options["train_fraction"],
        validation_fraction=options["validation_fraction"],
    )
    feature_columns = dataset["final_feature_columns"]
    n_features = len(feature_columns)
    if not n_features:
        raise ValueError("The prepared dataset contains no model features")

    artifact_dir = Path(options["artifact_dir"])
    metadata_path = _save_preprocessing_metadata(dataset, options, artifact_dir)
    is_final = options["run_mode"] == "final"
    parameters = {
        key: str(value) if isinstance(value, tuple) else value
        for key, value in options.items()
        if key not in {"mlflow_experiment_name", "registered_model_name"}
        and value is not None
    }
    parameters["n_features"] = n_features
    parameters["preprocessing_metadata_file"] = metadata_path.name

    def _train():
        return run_lstm_experiment(
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
            # The training utility reports its own artifacts to MLflow;
            # metadata is logged separately for later inference.
            mlflow.log_artifact(str(metadata_path), artifact_path="training_artifacts")
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
    print("Mode:", options["run_mode"])
    print("Model features:", n_features)
    print("Validation macro-F1:", result.validation_evaluation["macro_f1"])
    if result.test_evaluation is not None:
        print("Test macro-F1:", result.test_evaluation["macro_f1"])
    print("MLflow model URI:", model_uri)
    print("UC model version:", getattr(registered_version, "version", None))

    return {
        "experiment": result,
        "model_uri": model_uri,
        "registered_version": registered_version,
        "metadata_path": metadata_path,
        "feature_columns": feature_columns,
    }


if __name__ == "__main__":
    # Running on Databricks: use the active `spark` session if available.
    # Local runs need a configured SparkSession (and Databricks credentials).
    run_model_pipeline(spark=globals().get("spark"))
