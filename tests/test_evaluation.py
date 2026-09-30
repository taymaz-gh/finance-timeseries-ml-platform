"""Tests for sequence-model evaluation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from finance_ml.models.evaluation import (
    aggregate_true_labels,
    build_confusion_matrix,
    evaluate_sequence_predictions,
)


def test_aggregate_true_labels_trims_boundaries() -> None:
    """Keep one true label per retained original timestep."""
    y_true = np.array(
        [
            [0, 0, 1, 1, 2, 2, 3],
        ],
        dtype=np.int64,
    )

    metadata = [
        {
            "group": "A",
            "times": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ],
        }
    ]

    result = aggregate_true_labels(
        y_true,
        metadata,
        sequence_length=7,
        boundary_width=2,
    )

    assert result == {
        ("A", 3): 1,
        ("A", 4): 1,
        ("A", 5): 2,
    }


def test_build_confusion_matrix() -> None:
    """Count true classes by predicted classes."""
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
            2,
        ]
    )

    y_pred = np.array(
        [
            0,
            1,
            1,
            2,
            2,
        ]
    )

    result = build_confusion_matrix(
        y_true,
        y_pred,
        n_classes=3,
    )

    expected = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_evaluate_sequence_predictions_aggregates_overlap() -> None:
    """Evaluate unique retained timesteps after overlap aggregation."""
    probabilities = np.array(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.9, 0.1],
                [0.6, 0.4],
                [0.4, 0.6],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.1, 0.9],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
        ],
        dtype=float,
    )

    y_true = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.int64,
    )

    metadata = [
        {
            "group": "A",
            "times": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ],
        },
        {
            "group": "A",
            "times": [
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ],
        },
    ]

    result = evaluate_sequence_predictions(
        probabilities,
        y_true,
        metadata,
        sequence_length=7,
        n_classes=2,
        boundary_width=2,
    )

    assert result["n_predictions"] == 4

    np.testing.assert_array_equal(
        result["y_true"],
        np.array(
            [
                0,
                0,
                1,
                1,
            ]
        ),
    )

    np.testing.assert_array_equal(
        result["y_pred"],
        np.array(
            [
                0,
                0,
                1,
                1,
            ]
        ),
    )

    assert result["accuracy"] == pytest.approx(1.0)

    assert result["macro_f1"] == pytest.approx(1.0)

    np.testing.assert_array_equal(
        result["confusion_matrix"],
        np.array(
            [
                [2, 0],
                [0, 2],
            ]
        ),
    )


def test_evaluate_sequence_predictions_rejects_wrong_class_dimension() -> None:
    """Reject probability tensors with the wrong class dimension."""
    probabilities = np.zeros(
        shape=(1, 7, 3),
        dtype=float,
    )

    y_true = np.zeros(
        shape=(1, 7),
        dtype=np.int64,
    )

    metadata = [
        {
            "group": "A",
            "times": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ],
        }
    ]

    with pytest.raises(ValueError):
        evaluate_sequence_predictions(
            probabilities,
            y_true,
            metadata,
            sequence_length=7,
            n_classes=2,
            boundary_width=2,
        )


def test_evaluate_sequence_predictions_rejects_empty_retained_set() -> None:
    """Reject configurations that remove all positions."""
    probabilities = np.zeros(
        shape=(1, 4, 2),
        dtype=float,
    )

    y_true = np.zeros(
        shape=(1, 4),
        dtype=np.int64,
    )

    metadata = [
        {
            "group": "A",
            "times": [
                1,
                2,
                3,
                4,
            ],
        }
    ]

    with pytest.raises(ValueError):
        evaluate_sequence_predictions(
            probabilities,
            y_true,
            metadata,
            sequence_length=4,
            n_classes=2,
            boundary_width=2,
        )
