"""Tests for overlapping sequence prediction aggregation."""

from __future__ import annotations

import numpy as np

from finance_ml.models.prediction_aggregation import (
    aggregate_sequence_probabilities,
)


def test_discards_sequence_boundaries() -> None:
    """Discard predictions near both sequence boundaries."""
    probabilities = np.array(
        [
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.6, 0.4],
                [0.4, 0.6],
                [0.2, 0.8],
                [0.1, 0.9],
            ]
        ],
        dtype=float,
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

    result = aggregate_sequence_probabilities(
        probabilities,
        metadata,
        sequence_length=7,
        boundary_width=2,
    )

    assert [item["time"] for item in result] == [
        3,
        4,
        5,
    ]

    assert [item["predicted_class"] for item in result] == [
        0,
        0,
        1,
    ]

    assert all(item["n_contributions"] == 1 for item in result)


def test_averages_overlapping_predictions() -> None:
    """Average retained probabilities for repeated timesteps."""
    probabilities = np.array(
        [
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.8, 0.2],
                [0.6, 0.4],
                [0.4, 0.6],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.7, 0.3],
                [0.3, 0.7],
                [0.2, 0.8],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
        ],
        dtype=float,
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

    result = aggregate_sequence_probabilities(
        probabilities,
        metadata,
        sequence_length=7,
        boundary_width=2,
    )

    result_by_time = {item["time"]: item for item in result}

    np.testing.assert_allclose(
        result_by_time[4]["mean_probabilities"],
        np.array(
            [
                0.65,
                0.35,
            ]
        ),
    )

    np.testing.assert_allclose(
        result_by_time[5]["mean_probabilities"],
        np.array(
            [
                0.35,
                0.65,
            ]
        ),
    )

    assert result_by_time[4]["n_contributions"] == 2
    assert result_by_time[5]["n_contributions"] == 2

    assert result_by_time[4]["predicted_class"] == 0
    assert result_by_time[5]["predicted_class"] == 1
