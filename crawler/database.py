"""SQLite 数据库辅助函数。

爬虫只负责把“干净的字典列表”交给这里；后续分析模块只需要读数据库。
这种拆法能减少互相影响：换爬虫数据源时，不必改分析代码。
"""

from __future__ import annotations

import sqlite3
import re
from pathlib import Path
from typing import Iterable, Mapping

from crawler.config import DEFAULT_DB_PATH


def _normalize_news_title_key(title: object) -> str:
    """生成新闻标题去重键，供数据库层兜底使用。"""

    text = str(title or "")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\s\u3000]+", "", text)
    text = re.sub(r"(\.{3,}|…+)$", "", text)
    return text.casefold()


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """确保数据库文件夹存在，打开 SQLite，并返回连接对象。"""

    # 转为path对象，方便拿到父目录
    path = Path(db_path)
    # 若父目录（data/）文件夹不存在，则自动创建；若已经存在，不要报错
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """创建项目需要的表；如果表已经存在，不会重复创建。"""

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
    """按日期写入金价数据；同一天的数据再次写入时会覆盖旧值。"""

    count = 0
    for row in rows:
        if not row.get("date") or row.get("close") is None:
            raise ValueError(f"金价数据缺少必要字段 date/close: {row}")

        conn.execute(
            # excluded 可以理解为“这次新传入的值”。
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
                # get 方法取不到字段时返回 None，适合处理成交量等可能缺失的数据。
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
    """写入新闻数据。

    URL 相同时直接更新；URL 不同但“标题+发布时间”相同，也视作同一篇新闻。
    这是对 Bing RSS 跳转链接不稳定的兜底处理。
    """

    count = 0
    for row in rows:
        if not row.get("title"):
            raise ValueError(f"新闻数据缺少必要字段 title: {row}")

        publish_time = row.get("publish_time")
        title_key = _normalize_news_title_key(row.get("title"))
        existing_id = None
        if publish_time:
            existing_rows = conn.execute(
                "SELECT id, title FROM gold_news WHERE publish_time = ?",
                (publish_time,),
            ).fetchall()
            for existing in existing_rows:
                if _normalize_news_title_key(existing["title"]) == title_key:
                    existing_id = existing["id"]
                    break

        if existing_id is not None:
            url = row.get("url")
            if url:
                url_owner = conn.execute(
                    "SELECT id FROM gold_news WHERE url = ?",
                    (url,),
                ).fetchone()
                if url_owner is not None and url_owner["id"] != existing_id:
                    current = conn.execute(
                        "SELECT url FROM gold_news WHERE id = ?",
                        (existing_id,),
                    ).fetchone()
                    url = current["url"] if current else url

            conn.execute(
                """
                UPDATE gold_news
                SET title = :title,
                    publish_time = :publish_time,
                    content = :content,
                    source = :source,
                    url = :url
                WHERE id = :id
                """,
                {
                    "id": existing_id,
                    "title": row["title"],
                    "publish_time": publish_time,
                    "content": row.get("content"),
                    "source": row.get("source"),
                    "url": url,
                },
            )
            count += 1
            continue

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


def cleanup_duplicate_news(conn: sqlite3.Connection) -> int:
    """清理历史重复新闻，保留每组“规范化标题+发布时间”的第一条。"""

    rows = conn.execute(
        "SELECT id, title, publish_time FROM gold_news ORDER BY id"
    ).fetchall()
    seen: set[tuple[str, str]] = set()
    duplicate_ids: list[int] = []
    for row in rows:
        key = (_normalize_news_title_key(row["title"]), str(row["publish_time"] or ""))
        if key in seen:
            duplicate_ids.append(int(row["id"]))
            continue
        seen.add(key)

    if not duplicate_ids:
        return 0

    conn.executemany(
        "DELETE FROM gold_news WHERE id = ?",
        [(row_id,) for row_id in duplicate_ids],
    )
    conn.commit()
    return len(duplicate_ids)


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DEFAULT_DB_PATH}")
