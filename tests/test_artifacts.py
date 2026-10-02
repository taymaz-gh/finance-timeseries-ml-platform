"""Tests for training-artifact utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from finance_ml.models.artifacts import (
    save_training_artifacts,
)


def test_save_training_artifacts(tmp_path: Path) -> None:
    """Save history and all learning curves."""
    history = tf.keras.callbacks.History()

    history.history = {
        "loss": [0.8, 0.5],
        "accuracy": [0.70, 0.82],
        "macro_f1": [0.68, 0.80],
        "val_loss": [0.7, 0.4],
        "val_accuracy": [0.75, 0.85],
        "val_macro_f1": [0.72, 0.83],
    }

    paths = save_training_artifacts(
        history,
        tmp_path,
    )

    expected_files = {
        "history",
        "loss",
        "accuracy",
        "macro_f1",
    }

    assert set(paths) == expected_files

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0

    history_df = np.genfromtxt(
        paths["history"],
        delimiter=",",
        names=True,
    )

    assert len(history_df) == 2
    assert "loss" in history_df.dtype.names
    assert "macro_f1" in history_df.dtype.names
    assert "val_macro_f1" in history_df.dtype.names