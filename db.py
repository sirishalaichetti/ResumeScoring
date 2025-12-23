import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="resume_parser",
    user="postgres",
    password="root",
    port=5432
)
