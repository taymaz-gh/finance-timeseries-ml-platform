"""Performing seq2seq inference and timestep-level aggregation."""

from __future__ import annotations

import numpy as np

from finance_ml.models.prediction_aggregation import aggregate_sequence_probabilities


def predict_and_aggregate(model, X, sequence_metadata, *, sequence_length: int, boundary_width: int = 2, batch_size: int = 256):
    """Predict probabilities and average overlapping, boundary-trimmed windows."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1] != sequence_length or X.shape[0] == 0:
        raise ValueError("X must be a nonempty (n_sequences, sequence_length, features) array")
    probabilities = np.asarray(model.predict(X, batch_size=batch_size, verbose=0))
    if probabilities.ndim != 3 or probabilities.shape[:2] != X.shape[:2]:
        raise ValueError("Model must return (n_sequences, sequence_length, n_classes)")
    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned non-finite probabilities")
    if (probabilities < -1e-6).any() or not np.allclose(probabilities.sum(axis=-1), 1, atol=1e-3):
        raise ValueError("Model outputs are not valid class probabilities")
    return aggregate_sequence_probabilities(
        probabilities,
        sequence_metadata,
        sequence_length=sequence_length,
        boundary_width=boundary_width,
    )
