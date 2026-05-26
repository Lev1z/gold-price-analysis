"""Load project data from SQLite into pandas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crawler.config import DEFAULT_DB_PATH
from crawler.database import get_connection


def load_price_data(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Load gold price rows sorted by date."""

    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM gold_prices ORDER BY date",
            conn,
            parse_dates=["date"],
        )


def load_news_data(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Load gold news rows sorted by publish time."""

    with get_connection(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM gold_news ORDER BY publish_time",
            conn,
            parse_dates=["publish_time"],
        )
