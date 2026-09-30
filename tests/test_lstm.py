"""Tests for configurable LSTM sequence classifier."""

from __future__ import annotations

import tensorflow as tf

from finance_ml.models.lstm import (
    build_lstm_sequence_classifier,
)


def test_build_single_layer_lstm() -> None:
    """Build a single-layer seq2seq LSTM classifier."""
    model = build_lstm_sequence_classifier(
        sequence_length=7,
        n_features=41,
        n_classes=4,
        lstm_units=(64,),
    )

    assert model.input_shape == (
        None,
        7,
        41,
    )

    assert model.output_shape == (
        None,
        7,
        4,
    )

    lstm_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)
    ]

    assert len(lstm_layers) == 1
    assert lstm_layers[0].units == 64
    assert lstm_layers[0].return_sequences is True


def test_build_stacked_lstm() -> None:
    """Build a two-layer stacked seq2seq LSTM classifier."""
    model = build_lstm_sequence_classifier(
        sequence_length=7,
        n_features=41,
        n_classes=4,
        lstm_units=(64, 32),
    )

    lstm_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)
    ]

    assert len(lstm_layers) == 2

    assert [layer.units for layer in lstm_layers] == [
        64,
        32,
    ]

    assert all(layer.return_sequences for layer in lstm_layers)

    assert model.output_shape == (
        None,
        7,
        4,
    )


def test_lstm_uses_sparse_multiclass_configuration() -> None:
    """Compile the classifier for sparse multiclass targets."""
    model = build_lstm_sequence_classifier(
        sequence_length=7,
        n_features=41,
        n_classes=4,
    )

    assert model.loss == "sparse_categorical_crossentropy"

    x = tf.zeros(
        shape=(
            2,
            7,
            41,
        ),
        dtype=tf.float32,
    )

    y = tf.zeros(
        shape=(
            2,
            7,
        ),
        dtype=tf.int32,
    )

    metric_values = model.train_on_batch(
        x,
        y,
        return_dict=True,
    )

    assert "loss" in metric_values
    assert "accuracy" in metric_values


def test_invalid_lstm_configuration_raises_error() -> None:
    """Reject invalid LSTM architecture parameters."""
    invalid_configurations = [
        {
            "sequence_length": 0,
            "n_features": 41,
            "n_classes": 4,
        },
        {
            "sequence_length": 7,
            "n_features": 0,
            "n_classes": 4,
        },
        {
            "sequence_length": 7,
            "n_features": 41,
            "n_classes": 1,
        },
        {
            "sequence_length": 7,
            "n_features": 41,
            "n_classes": 4,
            "lstm_units": (),
        },
        {
            "sequence_length": 7,
            "n_features": 41,
            "n_classes": 4,
            "lstm_units": (64, 0),
        },
        {
            "sequence_length": 7,
            "n_features": 41,
            "n_classes": 4,
            "dropout_rate": 1.0,
        },
        {
            "sequence_length": 7,
            "n_features": 41,
            "n_classes": 4,
            "learning_rate": 0.0,
        },
    ]

    for configuration in invalid_configurations:
        try:
            build_lstm_sequence_classifier(**configuration)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Expected ValueError for invalid configuration: " f"{configuration}"
            )
