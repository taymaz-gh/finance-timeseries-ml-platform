"""Testing Databricks SQL query helper functions."""

from unittest.mock import MagicMock, patch

import pandas as pd

from finance_ml.data.query import (
    query_with_sql_connector,
    query_with_sqlalchemy,
)


MOCK_CONFIG = {
    "server_hostname": "test-host",
    "http_path": "/sql/test",
    "access_token": "test-token",
    "catalog": "test_catalog",
    "schema": "test_schema",
}


def test_query_with_sql_connector_returns_dataframe() -> None:
    """Checking that SQL connector query results are returned as a pandas DataFrame."""

    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [(120600,)]
    mock_cursor.description = [
        ("n_rows", "bigint", None, None, None, None, None),
    ]

    mock_connection = MagicMock()
    mock_connection.__enter__.return_value = mock_connection
    mock_connection.cursor.return_value.__enter__.return_value = mock_cursor

    with (
        patch(
            "finance_ml.data.query.load_databricks_config",
            return_value=MOCK_CONFIG,
        ),
        patch(
            "finance_ml.data.query.sql.connect",
            return_value=mock_connection,
        ),
    ):
        result = query_with_sql_connector(
            """
            SELECT COUNT(*) AS n_rows
            FROM finance_ml.bronze.account_events_raw
            """
        )

    expected = pd.DataFrame(
        [(120600,)],
        columns=["n_rows"],
    )

    pd.testing.assert_frame_equal(result, expected)


def test_query_with_sqlalchemy_returns_dataframe() -> None:
    """Checking that SQLAlchemy query results are returned as a pandas DataFrame."""

    expected = pd.DataFrame(
        [(120600,)],
        columns=["n_rows"],
    )

    mock_engine = MagicMock()
    mock_connection = MagicMock()

    mock_engine.connect.return_value.__enter__.return_value = mock_connection

    with (
        patch(
            "finance_ml.data.query.load_databricks_config",
            return_value=MOCK_CONFIG,
        ),
        patch(
            "finance_ml.data.query.create_engine",
            return_value=mock_engine,
        ),
        patch(
            "finance_ml.data.query.pd.read_sql",
            return_value=expected,
        ),
    ):
        result = query_with_sqlalchemy(
            """
            SELECT COUNT(*) AS n_rows
            FROM finance_ml.bronze.account_events_raw
            """
        )

    pd.testing.assert_frame_equal(result, expected)