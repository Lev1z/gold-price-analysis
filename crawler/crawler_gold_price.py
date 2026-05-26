"""Gold price crawler entry point.

This file is intentionally a skeleton. The team can fill in the real API URL and
JSON/HTML parsing after confirming the target website in browser DevTools.
"""

from __future__ import annotations

from crawler.config import DEFAULT_DB_PATH
from crawler.database import get_connection, init_db, upsert_price_rows


def fetch_gold_price_rows() -> list[dict[str, object]]:
    """Fetch gold price rows from a public source.

    Return rows shaped like:
    {
        "date": "2026-05-26",
        "open": 3300.0,
        "close": 3312.5,
        "high": 3320.0,
        "low": 3288.0,
        "volume": 10000,
        "source": "example",
    }
    """

    return []


def main() -> None:
    init_db(DEFAULT_DB_PATH)
    rows = fetch_gold_price_rows()
    with get_connection(DEFAULT_DB_PATH) as conn:
        inserted = upsert_price_rows(conn, rows)
    print(f"Gold price crawler finished: {inserted} rows saved.")


if __name__ == "__main__":
    main()
