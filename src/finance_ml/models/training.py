"""Training utilities for sequence-classification models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tensorflow as tf

from finance_ml.models.evaluation import (
    aggregate_true_labels,
)
from finance_ml.models.metrics import (
    macro_f1_score,
)
from finance_ml.models.prediction_aggregation import (
    aggregate_sequence_probabilities,
)


class AggregatedMacroF1(tf.keras.callbacks.Callback):
    """
    Compute evaluation-consistent macro-F1 after each training epoch.

    The callback performs model inference on a specified dataset at the
    end of each epoch, removes sequence-boundary predictions, aggregates
    overlapping probability vectors by original timestep, and computes
    macro-F1 on the resulting retained predictions.

    The same callback is used for both training and validation data.
    The metric is written to Keras logs under a configurable name such
    as ``macro_f1`` or ``val_macro_f1``.
    """

    def __init__(
        self,
        *,
        x_data: np.ndarray,
        y_data: np.ndarray,
        metadata: Sequence[dict],
        sequence_length: int,
        n_classes: int,
        boundary_width: int = 2,
        batch_size: int = 256,
        metric_name: str,
        verbose: int = 1,
    ) -> None:
        """
        Initialize aggregated macro-F1 computation.

        Args:
            x_data:
                Features with shape
                ``(n_sequences, sequence_length, n_features)``.

            y_data:
                Integer targets with shape
                ``(n_sequences, sequence_length)``.

            metadata:
                Metadata corresponding to the sequences.

            sequence_length:
                Number of timesteps in every sequence.

            n_classes:
                Number of target classes.

            boundary_width:
                Number of sequence positions discarded from each edge
                before overlapping predictions are aggregated.

            batch_size:
                Batch size used for model prediction.

            metric_name:
                Name under which the metric is written to Keras logs.

            verbose:
                Whether to print the metric after each epoch.
        """
        super().__init__()

        if n_classes <= 1:
            raise ValueError("n_classes must be greater than 1.")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if not metric_name:
            raise ValueError("metric_name must not be empty.")

        self.x_data = np.asarray(x_data)

        self.y_data = np.asarray(y_data)

        self.metadata = list(metadata)

        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.boundary_width = boundary_width
        self.batch_size = batch_size
        self.metric_name = metric_name
        self.verbose = verbose

        # Precomputing true labels avoids repeating this work after
        # every epoch.
        self.true_labels = aggregate_true_labels(
            self.y_data,
            self.metadata,
            sequence_length=self.sequence_length,
            boundary_width=self.boundary_width,
        )

    def on_epoch_end(
        self,
        epoch: int,
        logs: dict | None = None,
    ) -> None:
        """Compute and expose aggregated macro-F1."""
        if logs is None:
            logs = {}

        probabilities = self.model.predict(
            self.x_data,
            batch_size=self.batch_size,
            verbose=0,
        )

        aggregated_predictions = aggregate_sequence_probabilities(
            probabilities,
            self.metadata,
            sequence_length=self.sequence_length,
            boundary_width=self.boundary_width,
        )

        y_true_aggregated = []
        y_pred_aggregated = []

        for prediction in aggregated_predictions:
            key = (
                prediction["group"],
                prediction["time"],
            )

            if key not in self.true_labels:
                raise ValueError(
                    "Aggregated prediction has no matching "
                    f"ground-truth label for {key}."
                )

            y_true_aggregated.append(self.true_labels[key])

            y_pred_aggregated.append(prediction["predicted_class"])

        if not y_true_aggregated:
            raise ValueError(
                "No timesteps remain after boundary trimming."
            )

        aggregated_macro_f1 = macro_f1_score(
            np.asarray(y_true_aggregated),
            np.asarray(y_pred_aggregated),
            n_classes=self.n_classes,
        )

        logs[self.metric_name] = aggregated_macro_f1

        if self.verbose:
            print(
                f" - {self.metric_name}: "
                f"{aggregated_macro_f1:.4f}"
            )


def train_sequence_model(
    model: tf.keras.Model,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    metadata_train: Sequence[dict],
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    metadata_validation: Sequence[dict],
    sequence_length: int,
    n_classes: int,
    boundary_width: int = 2,
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 5,
    verbose: int = 1,
    reduce_lr_on_plateau: bool = False,
    lr_factor: float = 0.5,
    lr_patience: int = 2,
    min_lr: float = 1e-6,
    checkpoint_path: str | None = None,
) -> tf.keras.callbacks.History:
    """
    Train a sequence classifier using aggregated validation macro-F1.

    Training and validation macro-F1 are both computed using the same
    evaluation convention: boundary trimming followed by overlapping
    prediction aggregation at the original-timestep level.

    Early stopping, optional learning-rate reduction, and optional
    model checkpointing use ``val_macro_f1``.

    Args:
        model:
            Compiled Keras sequence-classification model.

        x_train:
            Training features.

        y_train:
            Integer training targets.

        metadata_train:
            Metadata corresponding to training sequences.

        x_validation:
            Validation features.

        y_validation:
            Integer validation targets.

        metadata_validation:
            Metadata corresponding to validation sequences.

        sequence_length:
            Number of timesteps in each sequence.

        n_classes:
            Number of target classes.

        boundary_width:
            Number of positions discarded from each sequence edge.

        epochs:
            Maximum number of training epochs.

        batch_size:
            Training and prediction batch size.

        patience:
            Number of epochs without improvement in validation
            macro-F1 before training stops.

        verbose:
            Keras training verbosity.

        reduce_lr_on_plateau:
            Whether to reduce the learning rate when validation
            macro-F1 stops improving.

        lr_factor:
            Multiplicative factor applied to the learning rate when
            a plateau is detected.

        lr_patience:
            Number of epochs without validation macro-F1 improvement
            before reducing the learning rate.

        min_lr:
            Lower bound for the learning rate.

        checkpoint_path:
            Optional path for saving the best model according to
            validation macro-F1. If ``None``, no model checkpoint
            is saved.

    Returns:
        Keras training history.

    Raises:
        ValueError:
            If training parameters are invalid.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if patience < 0:
        raise ValueError("patience must be non-negative.")

    if reduce_lr_on_plateau:
        if not 0.0 < lr_factor < 1.0:
            raise ValueError("lr_factor must be between 0 and 1.")

        if lr_patience < 0:
            raise ValueError("lr_patience must be non-negative.")

        if min_lr < 0:
            raise ValueError("min_lr must be non-negative.")

    training_macro_f1_callback = AggregatedMacroF1(
        x_data=x_train,
        y_data=y_train,
        metadata=metadata_train,
        sequence_length=sequence_length,
        n_classes=n_classes,
        boundary_width=boundary_width,
        batch_size=batch_size,
        metric_name="macro_f1",
        verbose=verbose,
    )

    validation_macro_f1_callback = AggregatedMacroF1(
        x_data=x_validation,
        y_data=y_validation,
        metadata=metadata_validation,
        sequence_length=sequence_length,
        n_classes=n_classes,
        boundary_width=boundary_width,
        batch_size=batch_size,
        metric_name="val_macro_f1",
        verbose=verbose,
    )

    callbacks = [
        training_macro_f1_callback,
        validation_macro_f1_callback,
    ]

    if reduce_lr_on_plateau:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1",
            mode="max",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
            verbose=verbose,
        )

        callbacks.append(reduce_lr)

    if checkpoint_path is not None:
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=verbose,
        )

        callbacks.append(model_checkpoint)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_macro_f1",
        mode="max",
        patience=patience,
        restore_best_weights=True,
        verbose=verbose,
    )

    callbacks.append(early_stopping)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(
            x_validation,
            y_validation,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
    )

    return history
