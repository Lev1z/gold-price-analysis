"""黄金相关新闻爬虫。

数据源：Bing News RSS。RSS 的好处是结构固定，适合作为课程项目的保底新闻源。
运行示例：

    python -m crawler.crawler_news --limit 30
"""

from __future__ import annotations

import argparse
import html
import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Iterable
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup

from crawler.config import (
    BING_NEWS_QUERIES,
    BING_NEWS_RSS_URL,
    DEFAULT_DB_PATH,
    REQUEST_DELAY_SECONDS,
)
from crawler.database import cleanup_duplicate_news, get_connection, init_db, upsert_news_rows
from crawler.http_client import get_response


TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "from",
    "spm",
    "ved",
}


def clean_text(text: str | None) -> str:
    """清理 RSS 摘要或网页正文中的 HTML 标签和多余空白。"""

    if not text:
        return ""

    unescaped = html.unescape(text)
    soup = BeautifulSoup(unescaped, "lxml")
    return " ".join(soup.get_text(" ", strip=True).split())


def normalize_publish_time(value: str | None) -> str | None:
    """把 RSS 的 pubDate 转成 SQLite 中更容易排序的字符串。"""

    if not value:
        return None

    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return value

    return dt.strftime("%Y-%m-%d %H:%M:%S")


def normalize_news_title(title: str | None) -> str:
    """生成新闻标题去重键，减少空格、省略号差异造成的重复。"""

    text = clean_text(title)
    text = re.sub(r"[\s\u3000]+", "", text)
    text = re.sub(r"(\.{3,}|…+)$", "", text)
    return text.casefold()


def canonicalize_news_url(url: str | None) -> str:
    """规范化新闻链接。

    Bing RSS 经常返回 apiclick.aspx 跳转链接，同一篇文章每次的 tid/c 参数可能不同。
    这里优先还原 query 里的真实 url，并去掉常见跟踪参数。
    """

    if not url:
        return ""

    parsed = urlparse(url.strip())
    query = parse_qs(parsed.query)
    if parsed.netloc.endswith("bing.com") and "url" in query and query["url"]:
        parsed = urlparse(query["url"][0])
        query = parse_qs(parsed.query)

    kept_query = {
        key: values
        for key, values in query.items()
        if key.lower() not in TRACKING_QUERY_KEYS
    }
    normalized_query = urlencode(kept_query, doseq=True)
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            "",
            normalized_query,
            "",
        )
    )


def parse_rss_items(xml_text: str) -> list[dict[str, object]]:
    """解析 RSS XML，提取新闻标题、链接、发布时间和摘要。"""

    root = ET.fromstring(xml_text)
    rows: list[dict[str, object]] = []

    for item in root.findall(".//item"):
        title = clean_text(item.findtext("title"))
        url = canonicalize_news_url(item.findtext("link"))
        description = clean_text(item.findtext("description"))
        publish_time = normalize_publish_time(item.findtext("pubDate"))

        if not title or not url:
            continue

        rows.append(
            {
                "title": title,
                "publish_time": publish_time,
                "content": description,
                "source": "Bing News RSS",
                "url": url,
            }
        )

    return rows


def fetch_article_text(url: str, max_paragraphs: int = 12) -> str:
    """尝试进入新闻原文页抓正文。

    很多新闻站点会反爬或跳转，所以这里失败时直接返回空字符串，
    主流程会自动使用 RSS 摘要作为兜底内容。
    """

    try:
        response = get_response(
            url,
            allow_redirects=True,
        )
    except RuntimeError:
        return ""

    soup = BeautifulSoup(response.text, "lxml")
    paragraphs = []
    for paragraph in soup.find_all("p"):
        text = clean_text(paragraph.get_text(" ", strip=True))
        if len(text) >= 20:
            paragraphs.append(text)
        if len(paragraphs) >= max_paragraphs:
            break

    return "\n".join(paragraphs)


def fetch_rss_rows(query: str) -> list[dict[str, object]]:
    """按关键词请求 Bing News RSS。"""

    response = get_response(
        BING_NEWS_RSS_URL,
        params={"q": query, "format": "rss"},
    )
    return parse_rss_items(response.text)


def deduplicate_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """按规范化 URL 或“标题+发布时间”去重。"""

    seen_urls: set[str] = set()
    seen_title_times: set[tuple[str, str]] = set()
    result: list[dict[str, object]] = []
    for row in rows:
        normalized_row = dict(row)
        url = canonicalize_news_url(str(normalized_row.get("url") or ""))
        title_key = normalize_news_title(str(normalized_row.get("title") or ""))
        publish_time = str(normalized_row.get("publish_time") or "")
        title_time_key = (title_key, publish_time)

        if (url and url in seen_urls) or title_time_key in seen_title_times:
            continue
        if url:
            seen_urls.add(url)
            normalized_row["url"] = url
        seen_title_times.add(title_time_key)
        result.append(normalized_row)
    return result


def fetch_gold_news_rows(
    limit: int = 30,
    enrich_articles: bool = True,
    queries: Iterable[str] = BING_NEWS_QUERIES,
) -> list[dict[str, object]]:
    """抓取黄金相关新闻，并尽量补充正文内容。"""

    rows: list[dict[str, object]] = []
    for query in queries:
        try:
            rows.extend(fetch_rss_rows(query))
        except RuntimeError as exc:
            # RSS 搜索属于外部网络请求，偶发失败不应该让整个项目流程中断。
            # 这里保留提示，方便汇报时说明“反爬/网络波动”的处理方式。
            print(f"Warning: RSS query failed, skipped query={query!r}: {exc}")
        time.sleep(REQUEST_DELAY_SECONDS)

    rows = deduplicate_rows(rows)[:limit]
    if not rows:
        print("Warning: no new RSS rows fetched. Existing database rows will be kept.")
        return []

    if enrich_articles:
        for row in rows:
            url = str(row.get("url") or "")
            article_text = fetch_article_text(url)
            if article_text:
                row["content"] = article_text

            # 记录一下域名，后续分析新闻来源时也能用。
            domain = urlparse(url).netloc
            if domain:
                row["source"] = f"Bing News RSS / {domain}"
            time.sleep(REQUEST_DELAY_SECONDS)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="爬取黄金相关新闻 RSS")
    parser.add_argument("--limit", type=int, default=30, help="最多保存多少条新闻")
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="只保存 RSS 摘要，不进入原文页抓正文",
    )
    args = parser.parse_args()

    init_db(DEFAULT_DB_PATH)
    rows = fetch_gold_news_rows(limit=args.limit, enrich_articles=not args.no_enrich)
    with get_connection(DEFAULT_DB_PATH) as conn:
        inserted = upsert_news_rows(conn, rows)
        deleted = cleanup_duplicate_news(conn)
    print(f"Gold news crawler finished: {inserted} rows saved, {deleted} duplicates removed.")


if __name__ == "__main__":
    main()
