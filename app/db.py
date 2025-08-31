import os
import psycopg2
from psycopg2.extras import RealDictCursor

def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "mydb"),
        user=os.getenv("DB_USER", "myuser"),
        password=os.getenv("DB_PASS", "mypass"),
        options=f"-c search_path={os.getenv('DB_SCHEMA','public')}"
    )

def dict_cur(conn):
    return conn.cursor(cursor_factory=RealDictCursor)

def get_schema() -> str:
    return os.getenv("DB_SCHEMA", "public")
