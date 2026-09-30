"""Tests for multiclass sequence-classification metrics."""

from __future__ import annotations

import numpy as np
import pytest

from finance_ml.models.metrics import macro_f1_score


def test_macro_f1_is_one_for_perfect_predictions() -> None:
    """Return macro-F1 of 1 for perfect multiclass predictions."""
    y_true = np.array(
        [
            0,
            1,
            2,
            3,
        ]
    )

    y_pred = np.array(
        [
            0,
            1,
            2,
            3,
        ]
    )

    score = macro_f1_score(
        y_true,
        y_pred,
        n_classes=4,
    )

    assert score == pytest.approx(1.0)


def test_macro_f1_for_imperfect_multiclass_predictions() -> None:
    """Average class-specific F1 scores equally."""
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
            2,
            2,
        ]
    )

    y_pred = np.array(
        [
            0,
            1,
            1,
            1,
            2,
            0,
        ]
    )

    score = macro_f1_score(
        y_true,
        y_pred,
        n_classes=3,
    )

    expected_class_0_f1 = 0.5
    expected_class_1_f1 = 0.8
    expected_class_2_f1 = 2.0 / 3.0

    expected_macro_f1 = np.mean(
        [
            expected_class_0_f1,
            expected_class_1_f1,
            expected_class_2_f1,
        ]
    )

    assert score == pytest.approx(expected_macro_f1)


def test_macro_f1_accepts_seq2seq_arrays() -> None:
    """Flatten sequence dimensions when computing timestep macro-F1."""
    y_true = np.array(
        [
            [0, 1, 2],
            [2, 1, 0],
        ]
    )

    y_pred = np.array(
        [
            [0, 1, 2],
            [2, 1, 0],
        ]
    )

    score = macro_f1_score(
        y_true,
        y_pred,
        n_classes=3,
    )

    assert score == pytest.approx(1.0)


def test_macro_f1_assigns_zero_to_unpredicted_class() -> None:
    """Assign F1 of zero when a true class is never predicted."""
    y_true = np.array(
        [
            0,
            1,
            2,
        ]
    )

    y_pred = np.array(
        [
            0,
            1,
            1,
        ]
    )

    score = macro_f1_score(
        y_true,
        y_pred,
        n_classes=3,
    )

    expected_class_0_f1 = 1.0
    expected_class_1_f1 = 2.0 / 3.0
    expected_class_2_f1 = 0.0

    expected_macro_f1 = np.mean(
        [
            expected_class_0_f1,
            expected_class_1_f1,
            expected_class_2_f1,
        ]
    )

    assert score == pytest.approx(expected_macro_f1)


def test_macro_f1_rejects_invalid_inputs() -> None:
    """Reject incompatible shapes, empty arrays, and invalid labels."""
    with pytest.raises(ValueError):
        macro_f1_score(
            np.array([0, 1]),
            np.array([0]),
            n_classes=2,
        )

    with pytest.raises(ValueError):
        macro_f1_score(
            np.array([]),
            np.array([]),
            n_classes=2,
        )

    with pytest.raises(ValueError):
        macro_f1_score(
            np.array([0, 2]),
            np.array([0, 1]),
            n_classes=2,
        )

    with pytest.raises(ValueError):
        macro_f1_score(
            np.array([0, 1]),
            np.array([0, 2]),
            n_classes=2,
        )

    with pytest.raises(ValueError):
        macro_f1_score(
            np.array([0, 1]),
            np.array([0, 1]),
            n_classes=1,
        )
