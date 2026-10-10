"""Testing seq2seq inference and aggregation without TensorFlow training."""
import numpy as np
import pytest

from finance_ml.models.inference import predict_and_aggregate


class FakeModel:
    def predict(self, X, **kwargs):
        return np.tile(np.array([[[0.2, 0.8]]]), (len(X), X.shape[1], 1))


def test_predictions_aggregated_by_timestep():
    X = np.zeros((2, 7, 2), dtype=np.float32)
    meta = [
        {"group": "a", "times": list(range(7))},
        {"group": "a", "times": list(range(1, 8))},
    ]
    result = predict_and_aggregate(FakeModel(), X, meta, sequence_length=7)
    assert len(result) == 4
    assert [x["time"] for x in result] == [2, 3, 4, 5]
    assert result[1]["n_contributions"] == 2
    assert all(x["predicted_class"] == 1 for x in result)


def test_empty_sequences_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        predict_and_aggregate(FakeModel(), np.empty((0, 7, 2)), [], sequence_length=7)
