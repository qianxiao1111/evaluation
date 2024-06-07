import sqlite3
import pandas as pd
from typing import Any


def executor_on_df(query: str, df: pd.DataFrame, table_name: str = "test") -> Any:
    path_to_db = "data/tmp.db"
    conn = sqlite3.connect(path_to_db)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    cursor = conn.cursor()
    try:
        resp = cursor.execute(query).fetchall()
        cursor.close()
        conn.close()
        return resp
    except Exception as e:
        cursor.close()
        conn.close()
        return e


def executor_on_db(query: str, path_to_db: str):
    conn = sqlite3.connect(path_to_db)
    cursor = conn.cursor()
    try:
        resp = cursor.execute(query).fetchall()
        cursor.close()
        conn.close()
        return resp
    except Exception as e:
        cursor.close()
        conn.close()
        return e


def read_table_from_db(query: str, path_to_db: str) -> Any:
    conn = sqlite3.connect(path_to_db)
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        return e
