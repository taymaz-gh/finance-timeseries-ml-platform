"""Testing credential-free synthetic data preparation."""

from __future__ import annotations

import pandas as pd
import pytest

from finance_ml.data.synthetic_sources import (
    LABEL_COLUMNS,
    generate_in_memory_synthetic_data,
)


def test_synthetic_data_shapes() -> None:
    """Checking historical and future account coverage."""
    data = generate_in_memory_synthetic_data(
        n_existing_accounts=3,
        historical_days=10,
        future_days=4,
        n_new_accounts=2,
    )

    assert len(data["historical"]) == 30
    assert len(data["temporal_observations"]) == 12
    assert len(data["account_observations"]) == 28
    assert len(data["temporal_truth"]) == 12
    assert len(data["account_truth"]) == 28


def test_future_observations_exclude_labels() -> None:
    """Preventing ground-truth leakage into inference inputs."""
    data = generate_in_memory_synthetic_data()

    for name in ("temporal_observations", "account_observations"):
        assert not set(LABEL_COLUMNS).intersection(data[name].columns)

    for name in ("temporal_truth", "account_truth"):
        assert set(LABEL_COLUMNS).issubset(data[name].columns)


def test_future_truth_matches_observation_keys() -> None:
    """Preserving account and timestamp alignment."""
    data = generate_in_memory_synthetic_data()

    for kind in ("temporal", "account"):
        observations = data[f"{kind}_observations"]
        truth = data[f"{kind}_truth"]

        pd.testing.assert_frame_equal(
            observations[["account_id", "date"]].reset_index(drop=True),
            truth[["account_id", "date"]].reset_index(drop=True),
        )


def test_generation_is_reproducible() -> None:
    """Producing identical data for identical random seeds."""
    first = generate_in_memory_synthetic_data(random_seed=42)
    second = generate_in_memory_synthetic_data(random_seed=42)

    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_zero_new_accounts_is_supported() -> None:
    """Allowing demonstrations with existing accounts only."""
    data = generate_in_memory_synthetic_data(n_new_accounts=0)

    assert data["account_observations"].empty
    assert data["account_truth"].empty


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_existing_accounts": 0},
        {"historical_days": -1},
        {"future_days": 0},
        {"n_new_accounts": -1},
        {"random_seed": -5},
        {"n_existing_accounts": True},
        {"n_existing_accounts": 20_001},
    ],
)
def test_rejects_invalid_demo_parameters(overrides: dict) -> None:
    """Rejecting invalid or excessively large local simulations."""
    with pytest.raises(ValueError):
        generate_in_memory_synthetic_data(**overrides)
