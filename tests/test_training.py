"""Tests for sequence-model training utilities."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from finance_ml.models.lstm import (
    build_lstm_sequence_classifier,
)
from finance_ml.models.training import (
    ValidationMacroF1,
    _aggregate_true_labels,
    train_sequence_model,
)


def test_aggregate_true_labels_trims_boundaries() -> None:
    """Aggregate one true label per retained timestep."""
    y_true = np.array(
        [
            [0, 0, 1, 1, 2, 2, 3],
        ]
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

    result = _aggregate_true_labels(
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


def test_aggregate_true_labels_rejects_inconsistent_overlap() -> None:
    """Reject conflicting labels for the same retained timestep."""
    y_true = np.array(
        [
            [0, 0, 1, 1, 2, 2, 3],
            [0, 0, 2, 2, 3, 3, 3],
        ]
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

    with pytest.raises(ValueError):
        _aggregate_true_labels(
            y_true,
            metadata,
            sequence_length=7,
            boundary_width=2,
        )


def test_validation_macro_f1_adds_metric_to_logs() -> None:
    """Expose aggregated validation macro-F1 to Keras logs."""
    x_validation = np.zeros(
        shape=(1, 7, 2),
        dtype=np.float32,
    )

    y_validation = np.array(
        [
            [0, 0, 1, 1, 2, 2, 3],
        ],
        dtype=np.int64,
    )

    metadata_validation = [
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

    callback = ValidationMacroF1(
        x_validation=x_validation,
        y_validation=y_validation,
        metadata_validation=metadata_validation,
        sequence_length=7,
        n_classes=4,
        boundary_width=2,
        batch_size=1,
        verbose=0,
    )

    model = build_lstm_sequence_classifier(
        sequence_length=7,
        n_features=2,
        n_classes=4,
        lstm_units=(4,),
        dropout_rate=0.0,
    )

    callback.set_model(model)

    logs: dict[str, float] = {}

    callback.on_epoch_end(
        epoch=0,
        logs=logs,
    )

    assert "val_macro_f1" in logs
    assert 0.0 <= logs["val_macro_f1"] <= 1.0


def test_train_sequence_model_runs() -> None:
    """Train a tiny model and return Keras history."""
    rng = np.random.default_rng(seed=42)

    x_train = rng.normal(size=(8, 7, 2)).astype(np.float32)

    y_train = rng.integers(
        low=0,
        high=3,
        size=(8, 7),
        dtype=np.int64,
    )

    x_validation = rng.normal(size=(2, 7, 2)).astype(np.float32)

    y_validation = np.array(
        [
            [0, 0, 1, 1, 2, 2, 2],
            [0, 1, 1, 2, 2, 0, 0],
        ],
        dtype=np.int64,
    )

    metadata_validation = [
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
            "group": "B",
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
    ]

    model = build_lstm_sequence_classifier(
        sequence_length=7,
        n_features=2,
        n_classes=3,
        lstm_units=(4,),
        dropout_rate=0.0,
    )

    history = train_sequence_model(
        model,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        metadata_validation=metadata_validation,
        sequence_length=7,
        n_classes=3,
        boundary_width=2,
        epochs=2,
        batch_size=4,
        patience=1,
        verbose=0,
    )

    assert isinstance(
        history,
        tf.keras.callbacks.History,
    )

    assert "loss" in history.history
    assert "val_loss" in history.history
    assert "val_macro_f1" in history.history


def test_train_sequence_model_rejects_invalid_training_parameters() -> None:
    """Reject invalid epoch, batch-size, and patience values."""
    model = build_lstm_sequence_classifier(
        sequence_length=7,
        n_features=2,
        n_classes=3,
        lstm_units=(4,),
    )

    x = np.zeros(
        shape=(1, 7, 2),
        dtype=np.float32,
    )

    y = np.zeros(
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
        train_sequence_model(
            model,
            x_train=x,
            y_train=y,
            x_validation=x,
            y_validation=y,
            metadata_validation=metadata,
            sequence_length=7,
            n_classes=3,
            epochs=0,
        )

    with pytest.raises(ValueError):
        train_sequence_model(
            model,
            x_train=x,
            y_train=y,
            x_validation=x,
            y_validation=y,
            metadata_validation=metadata,
            sequence_length=7,
            n_classes=3,
            batch_size=0,
        )

    with pytest.raises(ValueError):
        train_sequence_model(
            model,
            x_train=x,
            y_train=y,
            x_validation=x,
            y_validation=y,
            metadata_validation=metadata,
            sequence_length=7,
            n_classes=3,
            patience=-1,
        )
