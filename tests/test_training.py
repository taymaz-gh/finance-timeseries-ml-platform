"""Tests for sequence-model training utilities."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from finance_ml.models.evaluation import (
    aggregate_true_labels,
)
from finance_ml.models.lstm import (
    build_lstm_sequence_classifier,
)
from finance_ml.models.training import (
    AggregatedMacroF1,
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
            "times": [1, 2, 3, 4, 5, 6, 7],
        },
        {
            "group": "A",
            "times": [2, 3, 4, 5, 6, 7, 8],
        },
    ]

    with pytest.raises(ValueError):
        aggregate_true_labels(
            y_true,
            metadata,
            sequence_length=7,
            boundary_width=2,
        )


def test_aggregated_macro_f1_adds_metric_to_logs() -> None:
    """Expose evaluation-consistent macro-F1 to Keras logs."""
    x_data = np.zeros(
        shape=(1, 7, 2),
        dtype=np.float32,
    )

    y_data = np.array(
        [
            [0, 0, 1, 1, 2, 2, 3],
        ],
        dtype=np.int64,
    )

    metadata = [
        {
            "group": "A",
            "times": [1, 2, 3, 4, 5, 6, 7],
        }
    ]

    callback = AggregatedMacroF1(
        x_data=x_data,
        y_data=y_data,
        metadata=metadata,
        sequence_length=7,
        n_classes=4,
        boundary_width=2,
        batch_size=1,
        metric_name="macro_f1",
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

    assert "macro_f1" in logs
    assert 0.0 <= logs["macro_f1"] <= 1.0


def test_aggregated_macro_f1_supports_validation_metric_name() -> None:
    """Expose validation macro-F1 under the validation log name."""
    x_data = np.zeros(
        shape=(1, 7, 2),
        dtype=np.float32,
    )

    y_data = np.array(
        [
            [0, 0, 1, 1, 2, 2, 3],
        ],
        dtype=np.int64,
    )

    metadata = [
        {
            "group": "A",
            "times": [1, 2, 3, 4, 5, 6, 7],
        }
    ]

    callback = AggregatedMacroF1(
        x_data=x_data,
        y_data=y_data,
        metadata=metadata,
        sequence_length=7,
        n_classes=4,
        boundary_width=2,
        batch_size=1,
        metric_name="val_macro_f1",
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
    """Train a tiny model and return both macro-F1 histories."""
    rng = np.random.default_rng(seed=42)

    x_train = rng.normal(
        size=(8, 7, 2)
    ).astype(np.float32)

    y_train = rng.integers(
        low=0,
        high=3,
        size=(8, 7),
        dtype=np.int64,
    )

    metadata_train = [
        {
            "group": f"train_{index}",
            "times": [1, 2, 3, 4, 5, 6, 7],
        }
        for index in range(8)
    ]

    x_validation = rng.normal(
        size=(2, 7, 2)
    ).astype(np.float32)

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
            "times": [1, 2, 3, 4, 5, 6, 7],
        },
        {
            "group": "B",
            "times": [1, 2, 3, 4, 5, 6, 7],
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
        metadata_train=metadata_train,
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
    assert "macro_f1" in history.history
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
            "times": [1, 2, 3, 4, 5, 6, 7],
        }
    ]

    with pytest.raises(ValueError):
        train_sequence_model(
            model,
            x_train=x,
            y_train=y,
            metadata_train=metadata,
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
            metadata_train=metadata,
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
            metadata_train=metadata,
            x_validation=x,
            y_validation=y,
            metadata_validation=metadata,
            sequence_length=7,
            n_classes=3,
            patience=-1,
        )



def test_train_sequence_model_saves_best_model_checkpoint(
    tmp_path: Path,
) -> None:
    """Save the best complete Keras model when checkpointing is enabled."""
    rng = np.random.default_rng(seed=42)

    x_train = rng.normal(
        size=(8, 7, 2)
    ).astype(np.float32)

    y_train = rng.integers(
        low=0,
        high=3,
        size=(8, 7),
        dtype=np.int64,
    )

    metadata_train = [
        {
            "group": f"train_{index}",
            "times": [1, 2, 3, 4, 5, 6, 7],
        }
        for index in range(8)
    ]

    x_validation = rng.normal(
        size=(2, 7, 2)
    ).astype(np.float32)

    y_validation = rng.integers(
        low=0,
        high=3,
        size=(2, 7),
        dtype=np.int64,
    )

    metadata_validation = [
        {
            "group": f"validation_{index}",
            "times": [1, 2, 3, 4, 5, 6, 7],
        }
        for index in range(2)
    ]

    model = build_lstm_sequence_classifier(
        sequence_length=7,
        n_features=2,
        n_classes=3,
        lstm_units=(4,),
        dropout_rate=0.0,
    )

    checkpoint_path = tmp_path / "best_model.keras"

    train_sequence_model(
        model,
        x_train=x_train,
        y_train=y_train,
        metadata_train=metadata_train,
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
        reduce_lr_on_plateau=True,
        lr_factor=0.5,
        lr_patience=1,
        min_lr=1e-6,
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()
    assert checkpoint_path.is_file()

    loaded_model = tf.keras.models.load_model(
        checkpoint_path,
    )

    assert loaded_model.input_shape == (
        None,
        7,
        2,
    )

    assert loaded_model.output_shape == (
        None,
        7,
        3,
    )

    # Comparing the restored in-memory model with the saved checkpoint.
    original_predictions = model.predict(
        x_validation,
        verbose=0,
    )

    checkpoint_predictions = loaded_model.predict(
        x_validation,
        verbose=0,
    )

    np.testing.assert_allclose(
        original_predictions,
        checkpoint_predictions,
        rtol=1e-5,
        atol=1e-6,
    )



def test_train_sequence_model_rejects_invalid_reduce_lr_parameters() -> None:
    """Reject invalid ReduceLROnPlateau configuration."""
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
            "times": [1, 2, 3, 4, 5, 6, 7],
        }
    ]

    with pytest.raises(ValueError):
        train_sequence_model(
            model,
            x_train=x,
            y_train=y,
            metadata_train=metadata,
            x_validation=x,
            y_validation=y,
            metadata_validation=metadata,
            sequence_length=7,
            n_classes=3,
            epochs=1,
            reduce_lr_on_plateau=True,
            lr_factor=1.0,
        )

    with pytest.raises(ValueError):
        train_sequence_model(
            model,
            x_train=x,
            y_train=y,
            metadata_train=metadata,
            x_validation=x,
            y_validation=y,
            metadata_validation=metadata,
            sequence_length=7,
            n_classes=3,
            epochs=1,
            reduce_lr_on_plateau=True,
            lr_factor=0.5,
            min_lr=-1.0,
        )
