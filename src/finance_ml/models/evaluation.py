"""Evaluation utilities for sequence-classification models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from finance_ml.models.metrics import macro_f1_score
from finance_ml.models.prediction_aggregation import (
    aggregate_sequence_probabilities,
)


def aggregate_true_labels(
    y_true: np.ndarray,
    metadata: Sequence[dict],
    *,
    sequence_length: int,
    boundary_width: int = 2,
) -> dict[tuple[object, object], int]:
    """
    Aggregate true labels onto retained original timesteps.

    Overlapping sequence windows may contain the same original timestep
    multiple times. After boundary trimming, all retained occurrences of
    the same ``(group, time)`` pair must have the same true class label.

    Args:
        y_true:
            Integer target labels with shape
            ``(n_sequences, sequence_length)``.

        metadata:
            Metadata corresponding to the sequence windows.

            Each item must contain:
                - ``group``
                - ``times``

        sequence_length:
            Number of timesteps in each sequence.

        boundary_width:
            Number of positions excluded from both sequence edges.

    Returns:
        Mapping from ``(group, time)`` to the true integer class label.

    Raises:
        ValueError:
            If shapes, metadata, boundary settings, or repeated labels
            are inconsistent.
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

    true_labels: dict[
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

            if key in true_labels and true_labels[key] != class_label:
                raise ValueError(
                    "Overlapping sequences contain inconsistent "
                    f"ground-truth labels for {key}."
                )

            true_labels[key] = class_label

    return true_labels


def build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_classes: int,
) -> np.ndarray:
    """
    Build a multiclass confusion matrix.

    Rows correspond to true classes and columns correspond to predicted
    classes.

    Args:
        y_true:
            True integer class labels.

        y_pred:
            Predicted integer class labels.

        n_classes:
            Total number of target classes.

    Returns:
        Integer confusion matrix with shape
        ``(n_classes, n_classes)``.
    """
    y_true = np.asarray(y_true).reshape(-1)

    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same number of labels.")

    if n_classes <= 1:
        raise ValueError("n_classes must be greater than 1.")

    if np.any((y_true < 0) | (y_true >= n_classes)):
        raise ValueError("y_true contains class labels outside the valid range.")

    if np.any((y_pred < 0) | (y_pred >= n_classes)):
        raise ValueError("y_pred contains class labels outside the valid range.")

    confusion_matrix = np.zeros(
        shape=(
            n_classes,
            n_classes,
        ),
        dtype=np.int64,
    )

    for true_class, predicted_class in zip(
        y_true,
        y_pred,
        strict=True,
    ):
        confusion_matrix[
            true_class,
            predicted_class,
        ] += 1

    return confusion_matrix


def evaluate_sequence_predictions(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    metadata: Sequence[dict],
    *,
    sequence_length: int,
    n_classes: int,
    boundary_width: int = 2,
) -> dict:
    """
    Evaluate sequence predictions after trimming and aggregation.

    The evaluation procedure is:

        1. Discard predictions near sequence boundaries.
        2. Group predictions referring to the same original timestep.
        3. Average their class-probability vectors.
        4. Select the final class using ``argmax``.
        5. Match predictions with one true label per retained timestep.
        6. Compute timestep-level accuracy, macro-F1, and confusion matrix.

    This prevents overlapping sequence windows from counting the same
    real-world timestep multiple times.

    Args:
        probabilities:
            Model softmax probabilities with shape
            ``(n_sequences, sequence_length, n_classes)``.

        y_true:
            Integer target labels with shape
            ``(n_sequences, sequence_length)``.

        metadata:
            Sequence metadata corresponding to ``probabilities`` and
            ``y_true``.

        sequence_length:
            Number of timesteps in each sequence.

        n_classes:
            Number of target classes.

        boundary_width:
            Number of positions excluded from both sequence edges.

    Returns:
        Dictionary containing:

            ``accuracy``
                Accuracy over retained original timesteps.

            ``macro_f1``
                Macro-F1 over retained original timesteps.

            ``confusion_matrix``
                Array with true classes as rows and predicted classes
                as columns.

            ``y_true``
                Final retained true-label array.

            ``y_pred``
                Final retained predicted-label array.

            ``predictions``
                Aggregated prediction records.

            ``n_predictions``
                Number of unique retained original timesteps.

    Raises:
        ValueError:
            If predictions and true labels cannot be aligned.
    """
    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 3:
        raise ValueError(
            "probabilities must have shape "
            "(n_sequences, sequence_length, n_classes)."
        )

    if probabilities.shape[2] != n_classes:
        raise ValueError("n_classes does not match probabilities.shape[2].")

    aggregated_predictions = aggregate_sequence_probabilities(
        probabilities,
        list(metadata),
        sequence_length=sequence_length,
        boundary_width=boundary_width,
    )

    true_label_lookup = aggregate_true_labels(
        y_true,
        metadata,
        sequence_length=sequence_length,
        boundary_width=boundary_width,
    )

    final_true_labels = []
    final_predicted_labels = []

    for prediction in aggregated_predictions:
        key = (
            prediction["group"],
            prediction["time"],
        )

        if key not in true_label_lookup:
            raise ValueError(
                "Aggregated prediction has no matching "
                f"ground-truth label for {key}."
            )

        final_true_labels.append(true_label_lookup[key])

        final_predicted_labels.append(prediction["predicted_class"])

    if not final_true_labels:
        raise ValueError("No timesteps remain after boundary trimming.")

    final_true_labels_array = np.asarray(
        final_true_labels,
        dtype=np.int64,
    )

    final_predicted_labels_array = np.asarray(
        final_predicted_labels,
        dtype=np.int64,
    )

    accuracy = float(np.mean(final_true_labels_array == final_predicted_labels_array))

    macro_f1 = macro_f1_score(
        final_true_labels_array,
        final_predicted_labels_array,
        n_classes=n_classes,
    )

    confusion_matrix = build_confusion_matrix(
        final_true_labels_array,
        final_predicted_labels_array,
        n_classes=n_classes,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion_matrix,
        "y_true": final_true_labels_array,
        "y_pred": final_predicted_labels_array,
        "predictions": aggregated_predictions,
        "n_predictions": len(aggregated_predictions),
    }
