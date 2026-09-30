"""Configurable LSTM sequence-classification model."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers, metrics, optimizers


def build_lstm_sequence_classifier(
    *,
    sequence_length: int,
    n_features: int,
    n_classes: int,
    lstm_units: Sequence[int] = (64,),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Build and compile a configurable seq2seq LSTM classifier.

    Args:
        sequence_length:
            Number of timesteps in each input sequence.

        n_features:
            Number of input features per timestep.

        n_classes:
            Number of target classes.

        lstm_units:
            Number of units in each LSTM layer.

            Examples:
                (64,)       -> one LSTM layer
                (64, 32)    -> two stacked LSTM layers
                (128, 64, 32) -> three stacked LSTM layers

        dropout_rate:
            Dropout rate applied after each LSTM layer.

        learning_rate:
            Adam optimizer learning rate.

    Returns:
        Compiled Keras model.

    Raises:
        ValueError:
            If any model configuration is invalid.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    if n_features <= 0:
        raise ValueError("n_features must be positive.")

    if n_classes <= 1:
        raise ValueError("n_classes must be greater than 1.")

    if not lstm_units:
        raise ValueError("lstm_units must contain at least one layer size.")

    if any(units <= 0 for units in lstm_units):
        raise ValueError("All values in lstm_units must be positive.")

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must satisfy 0 <= dropout_rate < 1.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    inputs = layers.Input(
        shape=(
            sequence_length,
            n_features,
        ),
        name="sequence_input",
    )

    x = inputs

    for layer_index, units in enumerate(
        lstm_units,
        start=1,
    ):
        x = layers.LSTM(
            units=units,
            return_sequences=True,
            name=f"lstm_{layer_index}",
        )(x)

        x = layers.Dropout(
            rate=dropout_rate,
            name=f"dropout_{layer_index}",
        )(x)

    outputs = layers.TimeDistributed(
        layers.Dense(
            units=n_classes,
            activation="softmax",
        ),
        name="class_probabilities",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="lstm_sequence_classifier",
    )

    optimizer = optimizers.Adam(
        learning_rate=learning_rate,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    return model
