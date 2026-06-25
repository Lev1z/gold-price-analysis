import sqlite3

from crawler.database import (
    cleanup_duplicate_news,
    get_connection,
    init_db,
    upsert_news_rows,
    upsert_price_rows,
)


def test_init_db_creates_expected_tables(tmp_path):
    db_path = tmp_path / "gold.db"

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()

    table_names = {row[0] for row in rows}
    assert "gold_prices" in table_names
    assert "gold_news" in table_names


def test_upsert_price_rows_is_idempotent(tmp_path):
    db_path = tmp_path / "gold.db"
    init_db(db_path)
    row = {
        "date": "2026-05-26",
        "open": 3300.0,
        "close": 3310.0,
        "high": 3320.0,
        "low": 3290.0,
        "volume": 1000,
        "source": "test",
    }

    with get_connection(db_path) as conn:
        assert upsert_price_rows(conn, [row]) == 1
        row["close"] = 3315.0
        assert upsert_price_rows(conn, [row]) == 1
        rows = conn.execute("SELECT date, close FROM gold_prices").fetchall()

    assert len(rows) == 1
    assert rows[0]["date"] == "2026-05-26"
    assert rows[0]["close"] == 3315.0


def test_cleanup_duplicate_news_keeps_one_title_time_pair(tmp_path):
    db_path = tmp_path / "gold.db"
    init_db(db_path)

    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO gold_news (title, publish_time, content, source, url)
            VALUES (:title, :publish_time, :content, :source, :url)
            """,
            [
                {
                    "title": "黄金价格大跌",
                    "publish_time": "2026-05-29 08:00:00",
                    "content": "a",
                    "source": "Bing News RSS",
                    "url": "https://example.com/a",
                },
                {
                    "title": " 黄金 价格 大跌 ",
                    "publish_time": "2026-05-29 08:00:00",
                    "content": "b",
                    "source": "Bing News RSS",
                    "url": "https://example.com/b",
                },
                {
                    "title": "黄金价格反弹",
                    "publish_time": "2026-05-29 09:00:00",
                    "content": "c",
                    "source": "Bing News RSS",
                    "url": "https://example.com/c",
                },
            ],
        )
        conn.commit()

        deleted = cleanup_duplicate_news(conn)
        rows = conn.execute("SELECT title, publish_time FROM gold_news ORDER BY id").fetchall()

    assert deleted == 1
    assert [(row["title"], row["publish_time"]) for row in rows] == [
        ("黄金价格大跌", "2026-05-29 08:00:00"),
        ("黄金价格反弹", "2026-05-29 09:00:00"),
    ]


def test_upsert_news_rows_updates_existing_title_time_even_when_url_changes(tmp_path):
    db_path = tmp_path / "gold.db"
    init_db(db_path)

    first = {
        "title": "黄金价格大跌",
        "publish_time": "2026-05-29 08:00:00",
        "content": "old",
        "source": "Bing News RSS",
        "url": "https://example.com/a",
    }
    second = {
        "title": " 黄金 价格 大跌 ",
        "publish_time": "2026-05-29 08:00:00",
        "content": "new",
        "source": "Bing News RSS",
        "url": "https://example.com/b",
    }

    with get_connection(db_path) as conn:
        upsert_news_rows(conn, [first])
        upsert_news_rows(conn, [second])
        rows = conn.execute("SELECT title, content, url FROM gold_news").fetchall()

    assert len(rows) == 1
    assert rows[0]["content"] == "new"
    assert rows[0]["url"] == "https://example.com/b"


def test_upsert_news_rows_keeps_existing_url_when_new_url_belongs_to_another_row(tmp_path):
    db_path = tmp_path / "gold.db"
    init_db(db_path)

    with get_connection(db_path) as conn:
        upsert_news_rows(
            conn,
            [
                {
                    "title": "黄金价格大跌",
                    "publish_time": "2026-05-29 08:00:00",
                    "content": "old",
                    "source": "Bing News RSS",
                    "url": "https://example.com/a",
                },
                {
                    "title": "黄金价格反弹",
                    "publish_time": "2026-05-29 09:00:00",
                    "content": "other",
                    "source": "Bing News RSS",
                    "url": "https://example.com/b",
                },
            ],
        )

        upsert_news_rows(
            conn,
            [
                {
                    "title": " 黄金 价格 大跌 ",
                    "publish_time": "2026-05-29 08:00:00",
                    "content": "new",
                    "source": "东方财富黄金频道",
                    "url": "https://example.com/b",
                }
            ],
        )
        rows = conn.execute(
            "SELECT title, content, source, url FROM gold_news ORDER BY id"
        ).fetchall()

    assert len(rows) == 2
    assert rows[0]["content"] == "new"
    assert rows[0]["source"] == "东方财富黄金频道"
    assert rows[0]["url"] == "https://example.com/a"
    assert rows[1]["url"] == "https://example.com/b"
