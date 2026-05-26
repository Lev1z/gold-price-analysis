import sqlite3

from crawler.database import get_connection, init_db, upsert_price_rows


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
