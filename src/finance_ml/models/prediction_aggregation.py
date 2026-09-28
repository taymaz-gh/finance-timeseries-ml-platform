"""Utilities for aggregating overlapping sequence predictions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def aggregate_sequence_probabilities(
    probabilities: np.ndarray,
    metadata: list[dict],
    *,
    sequence_length: int,
    boundary_width: int = 2,
) -> list[dict]:
    """
    Aggregate overlapping seq2seq probability predictions by timestep.

    Predictions near sequence boundaries are discarded. Remaining
    probabilities for the same original timestep are averaged.

    Args:
        probabilities:
            Array with shape:
                (n_sequences, sequence_length, n_classes)

            Each timestep contains a class-probability vector.

        metadata:
            Sequence metadata produced alongside the sequence windows.

            Each item must contain:
                - "group"
                - "times"

            where "times" is an ordered list containing one original
            timestamp per timestep in the sequence.

        sequence_length:
            Number of timesteps in each sequence.

        boundary_width:
            Number of positions to discard from both the beginning and
            end of each sequence.

    Returns:
        List of dictionaries containing:
            - group
            - time
            - mean_probabilities
            - predicted_class
            - n_contributions

        Results are sorted by group and time.

    Raises:
        ValueError:
            If shapes or metadata are inconsistent, or boundary_width
            removes every sequence position.
    """
    if probabilities.ndim != 3:
        raise ValueError(
            "probabilities must have shape "
            "(n_sequences, sequence_length, n_classes)."
        )

    n_sequences, observed_sequence_length, _ = probabilities.shape

    if observed_sequence_length != sequence_length:
        raise ValueError("sequence_length does not match probabilities.shape[1].")

    if len(metadata) != n_sequences:
        raise ValueError("metadata length must equal the number of sequences.")

    if boundary_width < 0:
        raise ValueError("boundary_width must be non-negative.")

    first_retained_position = boundary_width
    last_retained_position = sequence_length - boundary_width

    if first_retained_position >= last_retained_position:
        raise ValueError("boundary_width removes all sequence positions.")

    probability_store = defaultdict(list)

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
            time_value = times[position]

            key = (
                group_value,
                time_value,
            )

            probability_store[key].append(
                probabilities[
                    sequence_index,
                    position,
                ]
            )

    aggregated_predictions = []

    for (
        group_value,
        time_value,
    ), probability_vectors in probability_store.items():

        stacked_probabilities = np.stack(
            probability_vectors,
            axis=0,
        )

        mean_probabilities = stacked_probabilities.mean(
            axis=0,
        )

        predicted_class = int(np.argmax(mean_probabilities))

        aggregated_predictions.append(
            {
                "group": group_value,
                "time": time_value,
                "mean_probabilities": mean_probabilities,
                "predicted_class": predicted_class,
                "n_contributions": len(probability_vectors),
            }
        )

    aggregated_predictions.sort(
        key=lambda item: (
            item["group"],
            item["time"],
        )
    )

    return aggregated_predictions
