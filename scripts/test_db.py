import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.db_connection import get_connection

conn = get_connection()

print("Connected to PostgreSQL!")

cur = conn.cursor()

cur.execute("SELECT version();")

version = cur.fetchone()

print("PostgreSQL version:")
print(version)

cur.close()
conn.close()
