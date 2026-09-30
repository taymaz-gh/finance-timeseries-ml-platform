"""Training utilities for sequence-classification models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tensorflow as tf

from finance_ml.models.metrics import macro_f1_score
from finance_ml.models.prediction_aggregation import (
    aggregate_sequence_probabilities,
)


def _aggregate_true_labels(
    y_true: np.ndarray,
    metadata: Sequence[dict],
    *,
    sequence_length: int,
    boundary_width: int,
) -> dict[tuple[object, object], int]:
    """
    Build one ground-truth class label per retained original timestep.

    Overlapping sequences can contain the same original timestep multiple
    times. The repeated ground-truth labels must therefore agree.

    Args:
        y_true:
            Integer sequence labels with shape
            ``(n_sequences, sequence_length)``.

        metadata:
            Sequence metadata containing ``group`` and ``times``.

        sequence_length:
            Number of timesteps in each sequence.

        boundary_width:
            Number of positions excluded from each sequence boundary.

    Returns:
        Mapping from ``(group, time)`` to the corresponding true class.

    Raises:
        ValueError:
            If shapes, metadata, or repeated labels are inconsistent.
    """
    y_true = np.asarray(y_true)

    if y_true.ndim != 2:
        raise ValueError("y_true must have shape " "(n_sequences, sequence_length).")

    n_sequences, observed_sequence_length = y_true.shape

    if observed_sequence_length != sequence_length:
        raise ValueError("sequence_length does not match y_true.shape[1].")

    if len(metadata) != n_sequences:
        raise ValueError("metadata length must equal the number of sequences.")

    if boundary_width < 0:
        raise ValueError("boundary_width must be non-negative.")

    first_retained_position = boundary_width
    last_retained_position = sequence_length - boundary_width

    if first_retained_position >= last_retained_position:
        raise ValueError("boundary_width removes all sequence positions.")

    aggregated_true_labels: dict[
        tuple[object, object],
        int,
    ] = {}

    for sequence_index in range(n_sequences):
        sequence_metadata = metadata[sequence_index]

        if "group" not in sequence_metadata:
            raise ValueError("Each metadata item must contain 'group'.")

        if "times" not in sequence_metadata:
            raise ValueError("Each metadata item must contain 'times'.")

        times = sequence_metadata["times"]

        if len(times) != sequence_length:
            raise ValueError("Metadata 'times' length must equal sequence_length.")

        group_value = sequence_metadata["group"]

        for position in range(
            first_retained_position,
            last_retained_position,
        ):
            key = (
                group_value,
                times[position],
            )

            class_label = int(
                y_true[
                    sequence_index,
                    position,
                ]
            )

            if (
                key in aggregated_true_labels
                and aggregated_true_labels[key] != class_label
            ):
                raise ValueError(
                    "Overlapping sequences contain inconsistent "
                    f"ground-truth labels for {key}."
                )

            aggregated_true_labels[key] = class_label

    return aggregated_true_labels


class ValidationMacroF1(tf.keras.callbacks.Callback):
    """
    Compute validation macro-F1 after prediction aggregation.

    The callback performs model inference on the validation sequences at
    the end of each epoch, removes sequence-boundary predictions,
    aggregates overlapping probability vectors by original timestep,
    and computes macro-F1 on the resulting retained predictions.
    """

    def __init__(
        self,
        *,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
        metadata_validation: Sequence[dict],
        sequence_length: int,
        n_classes: int,
        boundary_width: int = 2,
        batch_size: int = 256,
        verbose: int = 1,
    ) -> None:
        """
        Initialize validation macro-F1 computation.

        Args:
            x_validation:
                Validation features with shape
                ``(n_sequences, sequence_length, n_features)``.

            y_validation:
                Integer validation targets with shape
                ``(n_sequences, sequence_length)``.

            metadata_validation:
                Metadata corresponding to validation sequences.

            sequence_length:
                Number of timesteps in every sequence.

            n_classes:
                Number of target classes.

            boundary_width:
                Number of sequence positions discarded from each edge
                before overlapping predictions are aggregated.

            batch_size:
                Batch size used for validation prediction.

            verbose:
                Whether to print validation macro-F1 after each epoch.
        """
        super().__init__()

        if n_classes <= 1:
            raise ValueError("n_classes must be greater than 1.")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.x_validation = np.asarray(x_validation)

        self.y_validation = np.asarray(y_validation)

        self.metadata_validation = list(metadata_validation)

        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.boundary_width = boundary_width
        self.batch_size = batch_size
        self.verbose = verbose

        self.true_labels = _aggregate_true_labels(
            self.y_validation,
            self.metadata_validation,
            sequence_length=self.sequence_length,
            boundary_width=self.boundary_width,
        )

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict | None = None,
    ) -> None:
        """Compute and expose aggregated validation macro-F1."""
        if logs is None:
            logs = {}

        probabilities = self.model.predict(
            self.x_validation,
            batch_size=self.batch_size,
            verbose=0,
        )

        aggregated_predictions = aggregate_sequence_probabilities(
            probabilities,
            self.metadata_validation,
            sequence_length=self.sequence_length,
            boundary_width=self.boundary_width,
        )

        y_true_aggregated = []
        y_pred_aggregated = []

        for prediction in aggregated_predictions:
            key = (
                prediction["group"],
                prediction["time"],
            )

            if key not in self.true_labels:
                raise ValueError(
                    "Aggregated prediction has no matching "
                    f"ground-truth label for {key}."
                )

            y_true_aggregated.append(self.true_labels[key])

            y_pred_aggregated.append(prediction["predicted_class"])

        if not y_true_aggregated:
            raise ValueError(
                "No validation timesteps remain after " "boundary trimming."
            )

        validation_macro_f1 = macro_f1_score(
            np.asarray(y_true_aggregated),
            np.asarray(y_pred_aggregated),
            n_classes=self.n_classes,
        )

        logs["val_macro_f1"] = validation_macro_f1

        if self.verbose:
            print(f" - val_macro_f1: " f"{validation_macro_f1:.4f}")


def train_sequence_model(
    model: tf.keras.Model,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    metadata_validation: Sequence[dict],
    sequence_length: int,
    n_classes: int,
    boundary_width: int = 2,
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 5,
    verbose: int = 1,
) -> tf.keras.callbacks.History:
    """
    Train a sequence classifier using aggregated validation macro-F1.

    Model selection uses validation macro-F1 after boundary trimming and
    overlapping-prediction aggregation. This matches the downstream
    evaluation convention used by the project.

    Args:
        model:
            Compiled Keras sequence-classification model.

        x_train:
            Training features.

        y_train:
            Integer training targets.

        x_validation:
            Validation features.

        y_validation:
            Integer validation targets.

        metadata_validation:
            Metadata corresponding to validation sequences.

        sequence_length:
            Number of timesteps in each sequence.

        n_classes:
            Number of target classes.

        boundary_width:
            Number of positions discarded from each sequence edge.

        epochs:
            Maximum number of training epochs.

        batch_size:
            Training and prediction batch size.

        patience:
            Number of epochs without improvement in validation
            macro-F1 before training stops.

        verbose:
            Keras training verbosity.

    Returns:
        Keras training history.

    Raises:
        ValueError:
            If training parameters are invalid.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if patience < 0:
        raise ValueError("patience must be non-negative.")

    validation_macro_f1_callback = ValidationMacroF1(
        x_validation=x_validation,
        y_validation=y_validation,
        metadata_validation=metadata_validation,
        sequence_length=sequence_length,
        n_classes=n_classes,
        boundary_width=boundary_width,
        batch_size=batch_size,
        verbose=verbose,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=patience,
        restore_best_weights=True,
        verbose=verbose,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(
            x_validation,
            y_validation,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            validation_macro_f1_callback,
            early_stopping,
        ],
        shuffle=True,
        verbose=verbose,
    )

    return history
