from finance_ml.data.query import (
    query_with_sql_connector,
)


query = """
SELECT COUNT(*) AS n_rows
FROM finance_ml.bronze.account_events_raw
"""

df = query_with_sql_connector(query)

print(df)
