"""Tests for end-to-end LSTM experiment orchestration."""

from __future__ import annotations

import numpy as np

from finance_ml.models.experiment import (
    SequenceExperimentResult,
    run_lstm_experiment,
)


def _build_metadata(
    *,
    n_sequences: int,
    sequence_length: int,
    group_prefix: str,
) -> list[dict]:
    """Build simple non-overlapping metadata for toy experiment tests."""
    metadata = []

    for sequence_index in range(n_sequences):
        start_time = sequence_index * sequence_length

        metadata.append(
            {
                "group": f"{group_prefix}_{sequence_index}",
                "times": list(
                    range(
                        start_time,
                        start_time + sequence_length,
                    )
                ),
            }
        )

    return metadata


def test_run_lstm_experiment_returns_complete_result_with_test_evaluation() -> None:
    """Run a tiny end-to-end LSTM experiment -- evaluate_test=True"""
    rng = np.random.default_rng(seed=42)

    sequence_length = 7
    n_features = 3
    n_classes = 3

    x_train = rng.normal(
        size=(12, sequence_length, n_features)
    ).astype(np.float32)

    y_train = rng.integers(
        low=0,
        high=n_classes,
        size=(12, sequence_length),
        dtype=np.int64,
    )

    x_validation = rng.normal(
        size=(4, sequence_length, n_features)
    ).astype(np.float32)

    y_validation = rng.integers(
        low=0,
        high=n_classes,
        size=(4, sequence_length),
        dtype=np.int64,
    )

    x_test = rng.normal(
        size=(4, sequence_length, n_features)
    ).astype(np.float32)

    y_test = rng.integers(
        low=0,
        high=n_classes,
        size=(4, sequence_length),
        dtype=np.int64,
    )

    metadata_train = _build_metadata(
        n_sequences=12,
        sequence_length=sequence_length,
        group_prefix="train",
    )

    metadata_validation = _build_metadata(
        n_sequences=4,
        sequence_length=sequence_length,
        group_prefix="validation",
    )

    metadata_test = _build_metadata(
        n_sequences=4,
        sequence_length=sequence_length,
        group_prefix="test",
    )

    result = run_lstm_experiment(
        x_train=x_train,
        y_train=y_train,
        metadata_train=metadata_train,
        x_validation=x_validation,
        y_validation=y_validation,
        metadata_validation=metadata_validation,
        x_test=x_test,
        y_test=y_test,
        metadata_test=metadata_test,
        sequence_length=sequence_length,
        n_features=n_features,
        n_classes=n_classes,
        lstm_units=(8,),
        dropout_rate=0.0,
        learning_rate=1e-3,
        boundary_width=2,
        epochs=2,
        batch_size=4,
        patience=1,
        verbose=0,
        evaluate_test=True,
    )

    assert isinstance(
        result,
        SequenceExperimentResult,
    )

    assert result.model.input_shape == (
        None,
        sequence_length,
        n_features,
    )

    assert result.model.output_shape == (
        None,
        sequence_length,
        n_classes,
    )

    assert "loss" in result.history.history
    assert "val_loss" in result.history.history
    assert "val_macro_f1" in result.history.history

    assert result.validation_evaluation is not None
    assert result.test_evaluation is not None   # for:  evaluate_test=True

    assert 0.0 <= result.validation_evaluation["accuracy"] <= 1.0

    assert 0.0 <= result.validation_evaluation["macro_f1"] <= 1.0

    assert 0.0 <= result.test_evaluation["accuracy"] <= 1.0

    assert 0.0 <= result.test_evaluation["macro_f1"] <= 1.0

    assert result.validation_evaluation["confusion_matrix"].shape == (
        n_classes,
        n_classes,
    )

    assert result.test_evaluation["confusion_matrix"].shape == (
        n_classes,
        n_classes,
    )



def test_run_lstm_experiment_returns_complete_result_without_test_evaluation() -> None:
    """Run a tiny end-to-end LSTM experiment -- evaluate_test=False"""
    rng = np.random.default_rng(seed=42)

    sequence_length = 7
    n_features = 3
    n_classes = 3

    x_train = rng.normal(
        size=(12, sequence_length, n_features)
    ).astype(np.float32)

    y_train = rng.integers(
        low=0,
        high=n_classes,
        size=(12, sequence_length),
        dtype=np.int64,
    )

    x_validation = rng.normal(
        size=(4, sequence_length, n_features)
    ).astype(np.float32)

    y_validation = rng.integers(
        low=0,
        high=n_classes,
        size=(4, sequence_length),
        dtype=np.int64,
    )

    x_test = rng.normal(
        size=(4, sequence_length, n_features)
    ).astype(np.float32)

    y_test = rng.integers(
        low=0,
        high=n_classes,
        size=(4, sequence_length),
        dtype=np.int64,
    )

    metadata_train = _build_metadata(
        n_sequences=12,
        sequence_length=sequence_length,
        group_prefix="train",
    )

    metadata_validation = _build_metadata(
        n_sequences=4,
        sequence_length=sequence_length,
        group_prefix="validation",
    )

    metadata_test = _build_metadata(
        n_sequences=4,
        sequence_length=sequence_length,
        group_prefix="test",
    )

    result = run_lstm_experiment(
        x_train=x_train,
        y_train=y_train,
        metadata_train=metadata_train,
        x_validation=x_validation,
        y_validation=y_validation,
        metadata_validation=metadata_validation,
        x_test=x_test,
        y_test=y_test,
        metadata_test=metadata_test,
        sequence_length=sequence_length,
        n_features=n_features,
        n_classes=n_classes,
        lstm_units=(8,),
        dropout_rate=0.0,
        learning_rate=1e-3,
        boundary_width=2,
        epochs=2,
        batch_size=4,
        patience=1,
        verbose=0,
        evaluate_test=False,
    )

    assert isinstance(
        result,
        SequenceExperimentResult,
    )

    assert result.model.input_shape == (
        None,
        sequence_length,
        n_features,
    )

    assert result.model.output_shape == (
        None,
        sequence_length,
        n_classes,
    )

    assert "loss" in result.history.history
    assert "val_loss" in result.history.history
    assert "val_macro_f1" in result.history.history

    assert result.validation_evaluation is not None
    assert result.test_evaluation is None   # for:  evaluate_test=False

    assert 0.0 <= result.validation_evaluation["accuracy"] <= 1.0

    assert 0.0 <= result.validation_evaluation["macro_f1"] <= 1.0

    assert result.validation_evaluation["confusion_matrix"].shape == (
        n_classes,
        n_classes,
    )



def test_run_lstm_experiment_respects_boundary_trimming() -> None:
    """Evaluate only retained positions after boundary trimming."""
    rng = np.random.default_rng(seed=123)

    sequence_length = 7
    n_features = 2
    n_classes = 2

    x_train = rng.normal(size=(6, sequence_length, n_features)).astype(np.float32)

    y_train = rng.integers(
        low=0,
        high=n_classes,
        size=(6, sequence_length),
        dtype=np.int64,
    )

    x_validation = rng.normal(size=(2, sequence_length, n_features)).astype(np.float32)

    y_validation = rng.integers(
        low=0,
        high=n_classes,
        size=(2, sequence_length),
        dtype=np.int64,
    )

    x_test = rng.normal(size=(2, sequence_length, n_features)).astype(np.float32)

    y_test = rng.integers(
        low=0,
        high=n_classes,
        size=(2, sequence_length),
        dtype=np.int64,
    )

    metadata_train = _build_metadata(
        n_sequences=6,
        sequence_length=sequence_length,
        group_prefix="train",
    )

    metadata_validation = _build_metadata(
        n_sequences=2,
        sequence_length=sequence_length,
        group_prefix="validation",
    )

    metadata_test = _build_metadata(
        n_sequences=2,
        sequence_length=sequence_length,
        group_prefix="test",
    )

    result = run_lstm_experiment(
        x_train=x_train,
        y_train=y_train,
        metadata_train=metadata_train,
        x_validation=x_validation,
        y_validation=y_validation,
        metadata_validation=metadata_validation,
        x_test=x_test,
        y_test=y_test,
        metadata_test=metadata_test,
        sequence_length=sequence_length,
        n_features=n_features,
        n_classes=n_classes,
        lstm_units=(4,),
        dropout_rate=0.0,
        boundary_width=2,
        epochs=1,
        batch_size=2,
        patience=0,
        verbose=0,
    )

    retained_positions_per_sequence = sequence_length - 2 * 2

    expected_predictions = 2 * retained_positions_per_sequence

    assert result.validation_evaluation["n_predictions"] == expected_predictions

    assert result.test_evaluation["n_predictions"] == expected_predictions


def test_run_lstm_experiment_saves_best_model_checkpoint(
    tmp_path,
) -> None:
    """Save and report the best model checkpoint when requested."""
    rng = np.random.default_rng(seed=42)

    sequence_length = 7
    n_features = 2
    n_classes = 3

    x_train = rng.normal(
        size=(6, sequence_length, n_features)
    ).astype(np.float32)

    y_train = rng.integers(
        low=0,
        high=n_classes,
        size=(6, sequence_length),
        dtype=np.int64,
    )

    x_validation = rng.normal(
        size=(2, sequence_length, n_features)
    ).astype(np.float32)

    y_validation = rng.integers(
        low=0,
        high=n_classes,
        size=(2, sequence_length),
        dtype=np.int64,
    )

    x_test = rng.normal(
        size=(2, sequence_length, n_features)
    ).astype(np.float32)

    y_test = rng.integers(
        low=0,
        high=n_classes,
        size=(2, sequence_length),
        dtype=np.int64,
    )

    metadata_train = _build_metadata(
        n_sequences=6,
        sequence_length=sequence_length,
        group_prefix="train",
    )

    metadata_validation = _build_metadata(
        n_sequences=2,
        sequence_length=sequence_length,
        group_prefix="validation",
    )

    metadata_test = _build_metadata(
        n_sequences=2,
        sequence_length=sequence_length,
        group_prefix="test",
    )

    # Testing automatic creation of a nonexistent artifact directory.
    artifact_dir = tmp_path / "new" / "nested" / "artifacts"

    assert not artifact_dir.exists()

    result = run_lstm_experiment(
        x_train=x_train,
        y_train=y_train,
        metadata_train=metadata_train,
        x_validation=x_validation,
        y_validation=y_validation,
        metadata_validation=metadata_validation,
        x_test=x_test,
        y_test=y_test,
        metadata_test=metadata_test,
        sequence_length=sequence_length,
        n_features=n_features,
        n_classes=n_classes,
        lstm_units=(4,),
        dropout_rate=0.0,
        learning_rate=1e-3,
        boundary_width=2,
        epochs=2,
        batch_size=2,
        patience=1,
        verbose=0,
        reduce_lr_on_plateau=True,
        lr_factor=0.5,
        lr_patience=1,
        min_lr=1e-6,
        save_best_model=True,
        evaluate_test=False,
        seed=42,
        artifact_dir=artifact_dir,
    )

    assert result.artifact_paths is not None

    checkpoint_path = result.artifact_paths["best_model"]

    assert checkpoint_path.exists()
    assert checkpoint_path.is_file()
    assert checkpoint_path.name == "best_model.keras"
    assert artifact_dir.is_dir()
    assert checkpoint_path.parent == artifact_dir
