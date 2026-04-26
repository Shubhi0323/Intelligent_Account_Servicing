import sqlite3
try:
    conn = sqlite3.connect("iasw.db")
    conn.execute("DROP TABLE IF EXISTS requests")
    conn.execute("DROP TABLE IF EXISTS users")
    conn.commit()
    conn.close()
    print("Tables dropped.")
except Exception as e:
    print(f"Failed to drop tables: {e}")
