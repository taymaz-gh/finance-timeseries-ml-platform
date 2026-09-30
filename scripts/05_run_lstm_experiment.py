"""Run an LSTM sequence-classification experiment on the Gold dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from pyspark.sql import SparkSession

from finance_ml.models.experiment import run_lstm_experiment
from finance_ml.pipelines.model_dataset_pipeline import (
    prepare_model_dataset,
)
from finance_ml.data.query import query_with_sql_connector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_GOLD_TABLE = "finance_ml.gold.account_features"

SEED = 42

SEQUENCE_LENGTH = 7
STRIDE = 1
N_CLASSES = 4

LSTM_UNITS = (64,)
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-3

BOUNDARY_WIDTH = 2

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 256
DEFAULT_PATIENCE = 5


def parse_args() -> argparse.Namespace:
    """Parse command-line experiment parameters."""
    parser = argparse.ArgumentParser(
        description="Run an LSTM sequence-classification experiment."
    )

    parser.add_argument(
        "--gold-table",
        default=DEFAULT_GOLD_TABLE,
        help="Fully qualified Gold table name.",
    )

    parser.add_argument(
        "--run-name",
        default="lstm_baseline",
        help="Name used for saved experiment artifacts.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Maximum number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training and prediction batch size.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early-stopping patience.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Keras verbosity level.",
    )

    return parser.parse_args()


def configure_reproducibility() -> None:
    """Configure random seeds for reproducible model experiments."""
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def load_gold_as_spark(
    spark: SparkSession,
    gold_table: str,
):
    """Load the Gold table through Databricks SQL and convert it to Spark."""
    query = f"""
        SELECT *
        FROM {gold_table}
        ORDER BY account_id, date
    """

    gold_pdf = query_with_sql_connector(query)

    print(f"Gold rows loaded: {len(gold_pdf):,}")

    print(f"Gold columns: {len(gold_pdf.columns)}")

    return spark.createDataFrame(gold_pdf)


def save_experiment_summary(
    *,
    run_name: str,
    result,
    dataset: dict,
    parameters: dict,
) -> None:
    """Save scalar experiment results and configuration as JSON."""
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "run_name": run_name,
        "parameters": parameters,
        "dataset": {
            "train_rows": int(dataset["X_train"].shape[0]),
            "validation_rows": int(dataset["X_validation"].shape[0]),
            "test_rows": int(dataset["X_test"].shape[0]),
            "sequence_length": int(dataset["X_train"].shape[1]),
            "n_features": int(dataset["X_train"].shape[2]),
        },
        "validation": {
            "accuracy": float(result.validation_evaluation["accuracy"]),
            "macro_f1": float(result.validation_evaluation["macro_f1"]),
            "n_predictions": int(result.validation_evaluation["n_predictions"]),
            "confusion_matrix": (
                result.validation_evaluation["confusion_matrix"].tolist()
            ),
        },
        "test": {
            "accuracy": float(result.test_evaluation["accuracy"]),
            "macro_f1": float(result.test_evaluation["macro_f1"]),
            "n_predictions": int(result.test_evaluation["n_predictions"]),
            "confusion_matrix": (result.test_evaluation["confusion_matrix"].tolist()),
        },
        "training": {
            "epochs_completed": len(result.history.history["loss"]),
            "history": {
                key: [float(value) for value in values]
                for key, values in (result.history.history.items())
            },
        },
    }

    output_path = ARTIFACTS_DIR / f"{run_name}_results.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(f"Results saved to: {output_path}")


def main() -> None:
    """Run the configured LSTM experiment."""
    args = parse_args()

    configure_reproducibility()

    # Ensuring local PySpark uses the active project interpreter.
    os.environ.setdefault(
        "PYSPARK_PYTHON",
        sys.executable,
    )
    os.environ.setdefault(
        "PYSPARK_DRIVER_PYTHON",
        sys.executable,
    )

    print(f"Python: {sys.version.split()[0]}")

    print(f"TensorFlow: {tf.__version__}")

    print(
        "GPUs:",
        tf.config.list_physical_devices("GPU"),
    )

    print(f"Gold table: {args.gold_table}")

    print(f"Sequence length: {SEQUENCE_LENGTH}")

    print(f"Stride: {STRIDE}")

    print(f"LSTM units: {LSTM_UNITS}")

    print(f"Boundary width: {BOUNDARY_WIDTH}")

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("finance-timeseries-lstm-experiment")
        .getOrCreate()
    )

    try:
        print("\nLoading Gold data...")

        gold_df = load_gold_as_spark(
            spark,
            args.gold_table,
        )

        print("\nPreparing model dataset...")

        dataset = prepare_model_dataset(
            gold_df,
            sequence_length=SEQUENCE_LENGTH,
            stride=STRIDE,
            scaling_method="mad",
        )

        print(f"X_train: {dataset['X_train'].shape}")

        print(f"X_validation: " f"{dataset['X_validation'].shape}")

        print(f"X_test: {dataset['X_test'].shape}")

        print(f"Number of features: " f"{len(dataset['final_feature_columns'])}")

        parameters = {
            "sequence_length": SEQUENCE_LENGTH,
            "stride": STRIDE,
            "n_classes": N_CLASSES,
            "lstm_units": list(LSTM_UNITS),
            "dropout_rate": DROPOUT_RATE,
            "learning_rate": LEARNING_RATE,
            "boundary_width": BOUNDARY_WIDTH,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "scaling_method": "mad",
            "seed": SEED,
        }

        print("\nStarting experiment...")

        result = run_lstm_experiment(
            x_train=dataset["X_train"],
            y_train=dataset["y_train"],
            x_validation=dataset["X_validation"],
            y_validation=dataset["y_validation"],
            metadata_validation=dataset["validation_metadata"],
            x_test=dataset["X_test"],
            y_test=dataset["y_test"],
            metadata_test=dataset["test_metadata"],
            sequence_length=SEQUENCE_LENGTH,
            n_features=len(dataset["final_feature_columns"]),
            n_classes=N_CLASSES,
            lstm_units=LSTM_UNITS,
            dropout_rate=DROPOUT_RATE,
            learning_rate=LEARNING_RATE,
            boundary_width=BOUNDARY_WIDTH,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            verbose=args.verbose,
        )

        model_path = ARTIFACTS_DIR / f"{args.run_name}.keras"

        ARTIFACTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.model.save(model_path)

        print(f"\nModel saved to: {model_path}")

        print("\nValidation results:")

        print(f"  Accuracy: " f"{result.validation_evaluation['accuracy']:.4f}")

        print(f"  Macro-F1: " f"{result.validation_evaluation['macro_f1']:.4f}")

        print(f"  Predictions: " f"{result.validation_evaluation['n_predictions']:,}")

        print("\nTest results:")

        print(f"  Accuracy: " f"{result.test_evaluation['accuracy']:.4f}")

        print(f"  Macro-F1: " f"{result.test_evaluation['macro_f1']:.4f}")

        print(f"  Predictions: " f"{result.test_evaluation['n_predictions']:,}")

        save_experiment_summary(
            run_name=args.run_name,
            result=result,
            dataset=dataset,
            parameters=parameters,
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()



"""

Example usage:

python scripts/05_run_lstm_experiment.py `
    --run-name lstm_smoke_test `
    --epochs 1 `
    --patience 0


See Spark job UI for progress and logs:

http://localhost:4040


To see the CPU and memory usage of the Python and Java processes, run:

Get-Process python,java -ErrorAction SilentlyContinue | Select-Object ProcessName,CPU,WorkingSet64

"""
