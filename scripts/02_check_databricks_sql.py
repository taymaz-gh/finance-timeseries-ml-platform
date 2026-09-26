from dotenv import load_dotenv
import os

from databricks import sql

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
        cursor.execute(
            """
            SELECT COUNT(*) AS n_rows
            FROM finance_ml.bronze.account_events_raw
            """
        )

        rows = cursor.fetchall()

        print(rows)
        print(cursor.description)

        columns = [column[0] for column in cursor.description]
        print(columns)

        import pandas as pd
        df = pd.DataFrame(rows, columns=columns,)
        print(df)