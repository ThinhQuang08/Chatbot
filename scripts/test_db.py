import psycopg2

conn = psycopg2.connect(
    dbname="chatbot",
    user="chatbot_user",
    password="supersecret",
    host="localhost",
    port="5432"
)

print("Connected to PostgreSQL!")

cur = conn.cursor()

cur.execute("SELECT version();")

version = cur.fetchone()

print("PostgreSQL version:")
print(version)

cur.close()
conn.close()
