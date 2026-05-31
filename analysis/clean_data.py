"""数据清洗辅助函数。

爬虫拿到的数据可能有重复、空值、日期类型不统一等问题。
分析模块先统一清洗，再做统计和画图，后面的结果会更稳定。
"""

from __future__ import annotations

import pandas as pd


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗金价数据：日期标准化、数值列转换、去重、排序。"""

    if df.empty:
        return df.copy()

    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    for column in ["open", "close", "high", "low", "volume"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=["date", "close"])
    cleaned = cleaned.drop_duplicates(subset=["date"], keep="last")
    cleaned = cleaned.sort_values("date").reset_index(drop=True)
    return cleaned


def clean_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗新闻数据：去掉空标题、统一发布时间、按 URL 去重。"""

    if df.empty:
        return df.copy()

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["title"])
    if "publish_time" in cleaned.columns:
        cleaned["publish_time"] = pd.to_datetime(cleaned["publish_time"], errors="coerce")
    if "url" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["url"], keep="last")
    if "publish_time" in cleaned.columns:
        cleaned = cleaned.sort_values("publish_time", na_position="last")
    return cleaned.reset_index(drop=True)
