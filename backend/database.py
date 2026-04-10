import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "app.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_verified BOOLEAN DEFAULT 0,
        temp_code TEXT
    )
    """)
    # Cache table for offline average temp/location
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS location_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        city TEXT NOT NULL,
        avg_temp REAL NOT NULL,
        avg_humidity REAL NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)
    # Predictions History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        N REAL, P REAL, K REAL,
        ph REAL, temperature REAL, humidity REAL, rainfall REAL,
        crops TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)
    conn.commit()
    
    # Backward Compatibility DB Migration logic
    # Will fail silently if the columns already exist, acting perfectly as an automatic migration.
    migration_columns = ["ph REAL", "temperature REAL", "humidity REAL", "rainfall REAL"]
    for col in migration_columns:
        try:
            cursor.execute(f"ALTER TABLE predictions_history ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass # Column inherently already migrated

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
