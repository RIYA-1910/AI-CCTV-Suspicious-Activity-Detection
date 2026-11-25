# ---------- Event Database (SQLite) ----------
import sqlite3
from datetime import datetime
from config import DB_PATH


def _connect():
    # Opens (or creates) the database file
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create the incidents table if it doesn't exist yet."""
    with _connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,      -- real-world time the alert fired
                event_type TEXT,      -- e.g. 'Restricted zone entry'
                track_id INTEGER,     -- which tracked person triggered it
                video_time REAL,      -- seconds into the video
                image_path TEXT       -- evidence screenshot location
            )
        """)


def add_incident(event_type, track_id, video_time, image_path):
    """Save one incident record."""
    with _connect() as con:
        con.execute(
            "INSERT INTO incidents (created_at, event_type, track_id, video_time, image_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             event_type, int(track_id), float(video_time), image_path),
        )


def get_incidents(limit=50):
    """Return the newest incidents first, as a list of dictionaries."""
    with _connect() as con:
        con.row_factory = sqlite3.Row   # lets us access columns by name
        rows = con.execute(
            "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def clear_incidents():
    """Delete all incident records (used by the dashboard's clear button)."""
    with _connect() as con:
        con.execute("DELETE FROM incidents")