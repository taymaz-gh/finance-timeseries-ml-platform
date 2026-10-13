"""Testing YAML configuration loading independently of ML dependencies."""

from __future__ import annotations

import pytest

from finance_ml.config.config_loader import load_yaml_config
from finance_ml.config.training_config import TRAINING_CONFIG


def test_load_valid_yaml(tmp_path) -> None:
    """Loading valid experiment-specific overrides."""
    path = tmp_path / "training.yaml"
    path.write_text(
        "epochs: 20\nbatch_size: 128\n",
        encoding="utf-8",
    )

    result = load_yaml_config(path, allowed_keys=TRAINING_CONFIG)

    assert result == {"epochs": 20, "batch_size": 128}


def test_load_empty_yaml(tmp_path) -> None:
    """Treating an empty YAML document as no overrides."""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    assert load_yaml_config(path, allowed_keys=TRAINING_CONFIG) == {}


def test_reject_missing_file(tmp_path) -> None:
    """Rejecting a configuration path that does not exist."""
    path = tmp_path / "missing.yaml"

    with pytest.raises(ValueError, match="not found"):
        load_yaml_config(path, allowed_keys=TRAINING_CONFIG)


def test_reject_malformed_yaml(tmp_path) -> None:
    """Rejecting invalid YAML syntax."""
    path = tmp_path / "invalid.yaml"
    path.write_text("epochs: [20\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML"):
        load_yaml_config(path, allowed_keys=TRAINING_CONFIG)


def test_reject_non_mapping_yaml(tmp_path) -> None:
    """Rejecting YAML documents that are not dictionaries."""
    path = tmp_path / "list.yaml"
    path.write_text("- epochs\n- batch_size\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        load_yaml_config(path, allowed_keys=TRAINING_CONFIG)


def test_reject_unknown_key(tmp_path) -> None:
    """Detecting misspelled configuration parameters."""
    path = tmp_path / "typo.yaml"
    path.write_text("epohcs: 20\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown configuration keys"):
        load_yaml_config(path, allowed_keys=TRAINING_CONFIG)


def test_reject_non_string_key(tmp_path) -> None:
    """Rejecting YAML keys that are not strings."""
    path = tmp_path / "numeric_key.yaml"
    path.write_text("123: value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="keys must be strings"):
        load_yaml_config(path, allowed_keys=TRAINING_CONFIG)


def test_loading_does_not_mutate_defaults(tmp_path) -> None:
    """Preserving the shared default configuration."""
    path = tmp_path / "training.yaml"
    path.write_text("epochs: 20\n", encoding="utf-8")

    original_epochs = TRAINING_CONFIG["epochs"]

    result = load_yaml_config(path, allowed_keys=TRAINING_CONFIG)

    assert result["epochs"] == 20
    assert TRAINING_CONFIG["epochs"] == original_epochs
