"""Loading application configuration from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def load_databricks_config() -> dict[str, str]:
    """
    Loading Databricks connection settings from environment variables.

    Returns:
        Dictionary containing Databricks connection settings.

    Raises:
        ValueError:
            If a required setting is missing.
    """
    load_dotenv()

    config = {
        "server_hostname": os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        "http_path": os.getenv("DATABRICKS_HTTP_PATH"),
        "access_token": os.getenv("DATABRICKS_TOKEN"),
    }

    missing = [key for key, value in config.items() if not value]

    if missing:
        raise ValueError("Missing Databricks configuration: " + ", ".join(missing))

    return config
