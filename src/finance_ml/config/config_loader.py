"""Loading external configuration files without ML dependencies.

Keeping file parsing separate from pipeline-specific defaults and validation.

Providing reusable configuration-loading utilities for CLI commands,
scheduled jobs, and other application entry points.

Currently supporting YAML; additional formats may be added later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml_config(
    path: str | Path,
    *,
    allowed_keys: Mapping[str, Any],
) -> dict[str, Any]:
    """Load YAML overrides and reject malformed or unknown settings."""
    config_path = Path(path)

    if not config_path.is_file():
        raise ValueError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {config_path}") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read configuration file: {config_path}") from exc

    # Treating an empty YAML document as no overrides.
    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            "YAML configuration must contain a mapping " "of parameter names to values."
        )

    if any(not isinstance(key, str) for key in data):
        raise ValueError("YAML configuration keys must be strings.")

    unknown_keys = sorted(set(data) - set(allowed_keys))
    if unknown_keys:
        raise ValueError(f"Unknown configuration keys: {', '.join(unknown_keys)}")

    return data
