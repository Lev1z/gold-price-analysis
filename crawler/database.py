"""SQLite helpers for gold price and news data."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Mapping

from crawler.config import DEFAULT_DB_PATH


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection and ensure the parent directory exists."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create project tables if they do not already exist."""

    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gold_prices (
                date TEXT PRIMARY KEY,
                open REAL,
                close REAL NOT NULL,
                high REAL,
                low REAL,
                volume REAL,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gold_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                publish_time TEXT,
                content TEXT,
                source TEXT,
                url TEXT UNIQUE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def upsert_price_rows(
    conn: sqlite3.Connection, rows: Iterable[Mapping[str, object]]
) -> int:
    """Insert or update gold price rows by date."""

    count = 0
    for row in rows:
        conn.execute(
            """
            INSERT INTO gold_prices (date, open, close, high, low, volume, source)
            VALUES (:date, :open, :close, :high, :low, :volume, :source)
            ON CONFLICT(date) DO UPDATE SET
                open = excluded.open,
                close = excluded.close,
                high = excluded.high,
                low = excluded.low,
                volume = excluded.volume,
                source = excluded.source
            """,
            {
                "date": row["date"],
                "open": row.get("open"),
                "close": row["close"],
                "high": row.get("high"),
                "low": row.get("low"),
                "volume": row.get("volume"),
                "source": row.get("source"),
            },
        )
        count += 1
    conn.commit()
    return count


def upsert_news_rows(
    conn: sqlite3.Connection, rows: Iterable[Mapping[str, object]]
) -> int:
    """Insert or update news rows by URL."""

    count = 0
    for row in rows:
        conn.execute(
            """
            INSERT INTO gold_news (title, publish_time, content, source, url)
            VALUES (:title, :publish_time, :content, :source, :url)
            ON CONFLICT(url) DO UPDATE SET
                title = excluded.title,
                publish_time = excluded.publish_time,
                content = excluded.content,
                source = excluded.source
            """,
            {
                "title": row["title"],
                "publish_time": row.get("publish_time"),
                "content": row.get("content"),
                "source": row.get("source"),
                "url": row.get("url"),
            },
        )
        count += 1
    conn.commit()
    return count


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DEFAULT_DB_PATH}")
