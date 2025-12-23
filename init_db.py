import sqlite3

conn = sqlite3.connect("users.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    phone TEXT,
    upi TEXT UNIQUE,
    password TEXT
)
""")

conn.commit()
conn.close()

print("Database created successfully")
