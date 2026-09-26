import pytest

from finance_ml.utils.config import (
    load_databricks_config,
)


def test_missing_databricks_config_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checking that missing Databricks settings raise an error."""

    monkeypatch.delenv(
        "DATABRICKS_SERVER_HOSTNAME",
        raising=False,
    )
    monkeypatch.delenv(
        "DATABRICKS_HTTP_PATH",
        raising=False,
    )
    monkeypatch.delenv(
        "DATABRICKS_TOKEN",
        raising=False,
    )

    monkeypatch.setattr(
        "finance_ml.utils.config.load_dotenv",
        lambda: None,
    )

    with pytest.raises(ValueError):
        load_databricks_config()
