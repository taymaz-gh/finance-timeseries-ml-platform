"""Checking real Keras serialization and safe local reload without training."""

from __future__ import annotations

import json

import numpy as np

from finance_ml.models.local_bundle import (
    load_local_bundle,
    publish_local_bundle,
)


def test_real_keras_checkpoint_round_trip(tmp_path) -> None:
    """Saving, validating, and reloading a tiny untrained seq2seq model.

    Testing the artifact format without Spark, fitted models, network access,
    MLflow logging, or Unity Catalog registration.
    """
    import tensorflow as tf

    from finance_ml.models.lstm import build_lstm_sequence_classifier

    model = build_lstm_sequence_classifier(
        sequence_length=7,
        n_features=2,
        n_classes=4,
        lstm_units=(4,),
        dropout_rate=0.0,
        learning_rate=1e-3,
    )
    model.save(tmp_path / "best_model.keras")
    metadata = {
        "sequence_length": 7,
        "stride": 1,
        "boundary_width": 2,
        "n_classes": 4,
        "final_feature_columns": ["balance", "account_type_checking"],
        "model_columns": {
            "numeric_feature_columns": ["balance"],
            "categorical_feature_columns": ["account_type"],
        },
        "medians": {},
        "category_levels": {"account_type": ["checking"]},
        "scaler_parameters": {
            "balance": {"center": 0.0, "scale": 1.0, "method": "mad"}
        },
        "scaling_method": "mad",
        "train_end": "2026-02-01",
        "validation_end": "2026-02-10",
    }
    (tmp_path / "preprocessing_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    publish_local_bundle(tmp_path, model=model)

    inputs = np.zeros((1, 7, 2), dtype=np.float32)
    expected = model.predict(inputs, verbose=0)
    loaded = load_local_bundle(tmp_path)
    actual = loaded.model.predict(inputs, verbose=0)

    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert loaded.metadata["final_feature_columns"] == [
        "balance", "account_type_checking"
    ]
    tf.keras.backend.clear_session()
