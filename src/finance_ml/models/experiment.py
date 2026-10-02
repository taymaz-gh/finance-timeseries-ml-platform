"""Experiment orchestration for sequence-classification models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from finance_ml.models.evaluation import (
    evaluate_sequence_predictions,
)
from finance_ml.models.lstm import (
    build_lstm_sequence_classifier,
)
from finance_ml.models.training import (
    train_sequence_model,
)

from finance_ml.models.artifacts import (
    save_training_artifacts,
)

@dataclass
class SequenceExperimentResult:
    """
    Container for sequence-model experiment outputs.

    Attributes:
        model:
            Trained Keras model.

        history:
            Keras training history.

        validation_evaluation:
            Aggregated validation metrics and predictions.

        test_evaluation:
            Aggregated test metrics and predictions.
    """

    model: tf.keras.Model
    history: tf.keras.callbacks.History
    validation_evaluation: dict
    test_evaluation: dict
    artifact_paths: dict[str, Path] | None = None


def run_lstm_experiment(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    metadata_train: list[dict],
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    metadata_validation: list[dict],
    x_test: np.ndarray,
    y_test: np.ndarray,
    metadata_test: list[dict],
    sequence_length: int,
    n_features: int,
    n_classes: int,
    lstm_units: tuple[int, ...] = (64,),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    boundary_width: int = 2,
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 5,
    verbose: int = 1,
    artifact_dir: str | Path | None = None,
) -> SequenceExperimentResult:
    """
    Run a complete LSTM sequence-classification experiment.

    The experiment performs the following steps:

        1. Build and compile the configurable LSTM classifier.
        2. Train it using validation macro-F1 for early stopping.
        3. Restore the best model weights.
        4. Save the training history and learning-curve artifacts,
           when an artifact directory is provided.
        5. Generate validation softmax probabilities.
        6. Trim sequence boundaries and aggregate overlapping
           validation predictions.
        7. Compute validation accuracy, macro-F1, and confusion matrix.
        8. Generate test softmax probabilities.
        9. Apply the same trimming and aggregation procedure to test data.
        10. Compute final test metrics.

    Args:
        x_train:
            Training feature tensor.

        y_train:
            Training integer target tensor.

        metadata_train:
            Training sequence metadata.

        x_validation:
            Validation feature tensor.

        y_validation:
            Validation integer target tensor.

        metadata_validation:
            Validation sequence metadata.

        x_test:
            Test feature tensor.

        y_test:
            Test integer target tensor.

        metadata_test:
            Test sequence metadata.

        sequence_length:
            Number of timesteps in each sequence.

        n_features:
            Number of model input features per timestep.

        n_classes:
            Number of target classes.

        lstm_units:
            Hidden-unit configuration for one or more stacked
            LSTM layers.

        dropout_rate:
            Dropout rate applied after each LSTM layer.

        learning_rate:
            Adam optimizer learning rate.

        boundary_width:
            Number of positions removed from each sequence edge
            before overlap aggregation.

        epochs:
            Maximum number of training epochs.

        batch_size:
            Training and prediction batch size.

        patience:
            Early-stopping patience measured in epochs without
            validation macro-F1 improvement.

        verbose:
            Keras verbosity level.

        artifact_dir:
            Optional directory in which training-history and
            learning-curve artifacts are saved. If ``None``, no
            training artifacts are saved.

    Returns:
        SequenceExperimentResult containing the trained model,
        training history, validation evaluation, and test evaluation.
    """
    model = build_lstm_sequence_classifier(
        sequence_length=sequence_length,
        n_features=n_features,
        n_classes=n_classes,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    history = train_sequence_model(
        model,
        x_train=x_train,
        y_train=y_train,
        metadata_train=metadata_train,
        x_validation=x_validation,
        y_validation=y_validation,
        metadata_validation=metadata_validation,
        sequence_length=sequence_length,
        n_classes=n_classes,
        boundary_width=boundary_width,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        verbose=verbose,
    )

    artifact_paths = None

    if artifact_dir is not None:
        artifact_paths = save_training_artifacts(
            history,
            artifact_dir,
        )

    validation_probabilities = model.predict(
        x_validation,
        batch_size=batch_size,
        verbose=0,
    )

    validation_evaluation = evaluate_sequence_predictions(
        validation_probabilities,
        y_validation,
        metadata_validation,
        sequence_length=sequence_length,
        n_classes=n_classes,
        boundary_width=boundary_width,
    )

    test_probabilities = model.predict(
        x_test,
        batch_size=batch_size,
        verbose=0,
    )

    test_evaluation = evaluate_sequence_predictions(
        test_probabilities,
        y_test,
        metadata_test,
        sequence_length=sequence_length,
        n_classes=n_classes,
        boundary_width=boundary_width,
    )

    return SequenceExperimentResult(
        model=model,
        history=history,
        validation_evaluation=validation_evaluation,
        test_evaluation=test_evaluation,
        artifact_paths=artifact_paths,
    )
