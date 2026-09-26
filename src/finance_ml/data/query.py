"""Querying Databricks through the SQL Connector."""

from __future__ import annotations


import pandas as pd
from databricks import sql

from finance_ml.utils.config import load_databricks_config


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
    config = load_databricks_config()

    with sql.connect(
        server_hostname=config["server_hostname"],
        http_path=config["http_path"],
        access_token=config["access_token"],
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)

            rows = cursor.fetchall()

            columns = [column[0] for column in cursor.description]

    return pd.DataFrame(
        rows,
        columns=columns,
    )
