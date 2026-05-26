"""Gold news crawler entry point."""

from __future__ import annotations

from crawler.config import DEFAULT_DB_PATH
from crawler.database import get_connection, init_db, upsert_news_rows


def fetch_gold_news_rows() -> list[dict[str, object]]:
    """Fetch gold-related news rows from a public source.

    Return rows shaped like:
    {
        "title": "Gold price update",
        "publish_time": "2026-05-26 10:00:00",
        "content": "Full article text...",
        "source": "example",
        "url": "https://example.com/news/1",
    }
    """

    return []


def main() -> None:
    init_db(DEFAULT_DB_PATH)
    rows = fetch_gold_news_rows()
    with get_connection(DEFAULT_DB_PATH) as conn:
        inserted = upsert_news_rows(conn, rows)
    print(f"Gold news crawler finished: {inserted} rows saved.")


if __name__ == "__main__":
    main()
