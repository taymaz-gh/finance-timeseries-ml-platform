from finance_ml.data.query import (
    query_with_sqlalchemy,
)


query = """
SELECT COUNT(*) AS n_rows
FROM finance_ml.bronze.account_events_raw
"""

df = query_with_sqlalchemy(query)

print(df)
