"""Querying Databricks through the SQL Connector."""

from __future__ import annotations

import os

import pandas as pd
from databricks import sql
from dotenv import load_dotenv


def query_with_sql_connector(
    query: str,
) -> pd.DataFrame:
    """
    Executing a SQL query against Databricks.

    Args:
        query:
            SQL statement to execute.

    Returns:
        Query result as a pandas DataFrame.
    """
    load_dotenv()

    server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    access_token = os.getenv("DATABRICKS_TOKEN")

    with sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=access_token,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)

            rows = cursor.fetchall()

            columns = [column[0] for column in cursor.description]

    return pd.DataFrame(
        rows,
        columns=columns,
    )
