"""东方财富金价 K 线爬虫。

数据源：东方财富公开 K 线接口，当前使用 COMEX 黄金连续合约（GC00Y）的日线数据。
运行示例：

    python -m crawler.crawler_gold_price --limit 1000
"""

from __future__ import annotations

import argparse
from typing import Any

from crawler.config import (
    DEFAULT_DB_PATH,
    EASTMONEY_GOLD_NAME,
    EASTMONEY_GOLD_SECID,
    EASTMONEY_KLINE_URL,
)
from crawler.database import get_connection, init_db, upsert_price_rows
from crawler.http_client import get_response


def _to_float(value: str) -> float | None:
    """把接口返回的字符串转成 float；空值或 '-' 统一转成 None。"""

    if value in {"", "-", "None", "null"}:
        return None
    return float(value)


def parse_eastmoney_klines(payload: dict[str, Any]) -> list[dict[str, object]]:
    """解析东方财富 K 线 JSON，转换成数据库需要的字段。

    东方财富的单条 K 线是逗号分隔字符串，常见格式为：
    日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
    """

    data = payload.get("data") or {}
    klines = data.get("klines") or []
    rows: list[dict[str, object]] = []

    for item in klines:
        parts = item.split(",")
        if len(parts) < 6:
            # 返回格式异常时跳过该行，避免一条脏数据中断整个爬虫。
            continue

        rows.append(
            {
                "date": parts[0],
                "open": _to_float(parts[1]),
                "close": _to_float(parts[2]),
                "high": _to_float(parts[3]),
                "low": _to_float(parts[4]),
                "volume": _to_float(parts[5]),
                "source": EASTMONEY_GOLD_NAME,
            }
        )

    return rows


def fetch_gold_price_rows(limit: int = 1000) -> list[dict[str, object]]:
    """请求东方财富接口，返回金价日线数据。

    limit 表示最多拉取多少条日线；1000 条大约覆盖 4 年交易日，足够课程分析使用。
    """

    params = {
        "secid": EASTMONEY_GOLD_SECID,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "0",
        "end": "20500101",
        "lmt": str(limit),
    }
    response = get_response(
        EASTMONEY_KLINE_URL,
        params=params,
    )
    payload = response.json()
    if payload.get("rc") != 0:
        raise RuntimeError(f"东方财富接口返回异常: {payload}")

    return parse_eastmoney_klines(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="爬取东方财富 COMEX 黄金日线数据")
    parser.add_argument("--limit", type=int, default=1000, help="最多拉取多少条日线")
    args = parser.parse_args()

    init_db(DEFAULT_DB_PATH)
    rows = fetch_gold_price_rows(limit=args.limit)
    with get_connection(DEFAULT_DB_PATH) as conn:
        inserted = upsert_price_rows(conn, rows)
    print(f"Gold price crawler finished: {inserted} rows saved.")


if __name__ == "__main__":
    main()
