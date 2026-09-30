"""Evaluation metrics for multiclass sequence classification."""

from __future__ import annotations

import numpy as np


def macro_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_classes: int,
) -> float:
    """
    Compute multiclass macro-F1 from integer class labels.

    Args:
        y_true:
            Ground-truth class labels.

            Can be any shape, for example:
                (n_samples,)
                (n_sequences, sequence_length)

        y_pred:
            Predicted integer class labels with the same shape
            as ``y_true``.

        n_classes:
            Total number of target classes.

    Returns:
        Macro-F1 score obtained by computing F1 independently
        for each class and then averaging across classes.

    Raises:
        ValueError:
            If inputs are incompatible or n_classes is invalid.
    """
    if n_classes <= 1:
        raise ValueError("n_classes must be greater than 1.")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    if np.any((y_true_flat < 0) | (y_true_flat >= n_classes)):
        raise ValueError("y_true contains class labels outside the valid range.")

    if np.any((y_pred_flat < 0) | (y_pred_flat >= n_classes)):
        raise ValueError("y_pred contains class labels outside the valid range.")

    class_f1_scores = []

    for class_index in range(n_classes):
        true_positive = np.sum(
            (y_true_flat == class_index) & (y_pred_flat == class_index)
        )

        false_positive = np.sum(
            (y_true_flat != class_index) & (y_pred_flat == class_index)
        )

        false_negative = np.sum(
            (y_true_flat == class_index) & (y_pred_flat != class_index)
        )

        precision_denominator = true_positive + false_positive

        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator if precision_denominator > 0 else 0.0
        )

        recall = true_positive / recall_denominator if recall_denominator > 0 else 0.0

        f1_denominator = precision + recall

        f1 = 2.0 * precision * recall / f1_denominator if f1_denominator > 0 else 0.0

        class_f1_scores.append(float(f1))

    return float(np.mean(class_f1_scores))
