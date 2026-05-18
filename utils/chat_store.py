"""
chat_store.py
-------------
SQLite-backed chat session store.

Location  : utils/chat_store.py
DB file   : data_cache/chat_history.db

Rules:
- Max 7 sessions total (FIFO — oldest deleted when limit exceeded)
- Each session stores ordered messages (role + content)
- Sessions are ordered by last_updated so the most recent appears first
- Sessions are resumed by loading their full message history
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── path ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "data_cache" / "chat_history.db"

MAX_SESSIONS = 7


# ── bootstrap ─────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    """Return a connection with row_factory so rows behave like dicts."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent FastAPI workers
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Call once at app startup."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id           TEXT PRIMARY KEY,
                title        TEXT NOT NULL DEFAULT 'New Chat',
                created_at   TEXT NOT NULL,
                last_updated TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role       TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);

            CREATE INDEX IF NOT EXISTS idx_sessions_updated
                ON sessions(last_updated DESC);
        """)


# ── internal helpers ───────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]


def _oldest_session_id(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute(
        "SELECT id FROM sessions ORDER BY last_updated ASC LIMIT 1"
    ).fetchone()
    return row["id"] if row else None


def _delete_session(conn: sqlite3.Connection, session_id: str) -> None:
    # CASCADE deletes messages automatically
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def _enforce_fifo(conn: sqlite3.Connection) -> None:
    """Delete the oldest session(s) until we are under MAX_SESSIONS."""
    while _session_count(conn) >= MAX_SESSIONS:
        oldest = _oldest_session_id(conn)
        if oldest:
            _delete_session(conn, oldest)


# ── public API ─────────────────────────────────────────────────────────────────

def create_session() -> dict:
    """
    Create a new session (enforcing the 7-session FIFO limit).
    Returns the new session dict + refreshed session list.
    """
    session_id = str(uuid.uuid4())
    now        = _now()

    with _get_conn() as conn:
        _enforce_fifo(conn)
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, last_updated) VALUES (?, ?, ?, ?)",
            (session_id, "New Chat", now, now),
        )

    return {
        "session_id": session_id,
        "sessions":   list_sessions(),
    }


def list_sessions() -> list[dict]:
    """
    Return all sessions ordered by last_updated DESC (most recent first).
    Each entry: {id, title, last_updated}
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, last_updated FROM sessions ORDER BY last_updated DESC"
        ).fetchall()

    return [dict(r) for r in rows]


def get_session_messages(session_id: str) -> list[dict]:
    """
    Return all messages for a session in chronological order.
    Each entry: {role, content}
    """
    with _get_conn() as conn:
        # Confirm session exists
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()

        if not exists:
            return []

        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()

    return [dict(r) for r in rows]


def append_message(session_id: str, role: str, content: str) -> None:
    """
    Append a message to a session and bump last_updated.
    Also sets the session title from the first user message (trimmed to 60 chars).
    """
    now = _now()

    with _get_conn() as conn:
        # Upsert-guard: create session if it somehow doesn't exist
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, title, created_at, last_updated) VALUES (?, 'New Chat', ?, ?)",
            (session_id, now, now),
        )

        # Auto-title: use first user message
        if role == "user":
            current_title = conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()["title"]

            if current_title == "New Chat":
                title = content.strip()[:60]
                conn.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?",
                    (title, session_id),
                )

        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )

        conn.execute(
            "UPDATE sessions SET last_updated = ? WHERE id = ?",
            (now, session_id),
        )


def delete_session(session_id: str) -> bool:
    """
    Delete a session and all its messages.
    Returns True if the session existed, False otherwise.
    """
    with _get_conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()

        if not exists:
            return False

        _delete_session(conn, session_id)

    return True


# ── initialise on import ───────────────────────────────────────────────────────
init_db()
