"""Querying Databricks through the SQL Connector."""

from __future__ import annotations

import pandas as pd

from databricks import sql

from finance_ml.utils.config import load_databricks_config

from sqlalchemy import create_engine
from sqlalchemy.engine import URL


def query_with_spark(
    spark,
    query: str,
):
    """
    Executing a SQL query through an existing Spark session.

    Args:
        spark:
            Active SparkSession, such as the one provided by Databricks.

        query:
            SQL statement to execute.

    Returns:
        Spark DataFrame containing the query result.
    """
    return spark.sql(query)


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


def query_with_sqlalchemy(
    query: str,
) -> pd.DataFrame:
    """
    Executing a SQL query against Databricks through SQLAlchemy.

    Args:
        query:
            SQL statement to execute.

    Returns:
        Query result as a pandas DataFrame.
    """
    config = load_databricks_config()

    connection_url = URL.create(
        "databricks",
        username="token",
        password=config["access_token"],
        host=config["server_hostname"],
        query={
            "http_path": config["http_path"],
            "catalog": config["catalog"],
            "schema": config["schema"],
        },
    )

    engine = create_engine(connection_url)

    with engine.connect() as connection:
        return pd.read_sql(
            query,
            connection,
        )
