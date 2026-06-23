"""Streamlit 仪表盘的数据处理与图表工具。

把这些逻辑从页面文件中拆出来，方便测试，也方便后续替换预测模型。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import re

import pandas as pd
import plotly.graph_objects as go

from analysis.clean_data import clean_news_data, clean_price_data
from analysis.load_data import load_news_data, load_price_data
from analysis.news_quality import is_low_quality_news, score_news_relevance
from analysis.statistics import add_price_indicators
from crawler.config import DEFAULT_DB_PATH


BULLISH_KEYWORDS = ("上涨", "走高", "反弹", "飙升", "突破", "新高", "避险", "买入", "降息")
BEARISH_KEYWORDS = ("下跌", "大跌", "暴跌", "承压", "回落", "跳水", "跌破", "抛售", "加息")
EVENT_NEWS_KEYWORDS = (
    "美联储",
    "美元",
    "美债",
    "CPI",
    "PCE",
    "非农",
    "地缘",
    "中东",
    "关税",
    "避险",
    "通胀",
    "降息",
    "加息",
    "收益率",
    "央行",
)
TIME_RANGE_DAYS = {
    "1M": 31,
    "3M": 93,
    "6M": 186,
    "1Y": 366,
}


def load_dashboard_data(db_path: str | Path = DEFAULT_DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取并清洗网页展示需要的数据。"""

    prices = clean_price_data(load_price_data(db_path))
    news = clean_news_data(load_news_data(db_path))
    return prices, news


def compute_dashboard_metrics(
    prices: pd.DataFrame, news: pd.DataFrame
) -> dict[str, str | float | int]:
    """计算顶部指标卡需要展示的数据。"""

    if prices.empty:
        return {
            "start_date": "-",
            "end_date": "-",
            "latest_close": 0.0,
            "total_return": 0.0,
            "volatility_20": 0.0,
            "news_count": int(len(news)),
        }

    working = add_price_indicators(prices, windows=(20,))
    first_close = float(working["close"].iloc[0])
    latest_close = float(working["close"].iloc[-1])
    volatility = working["volatility_20"].dropna()

    return {
        "start_date": working["date"].iloc[0].strftime("%Y-%m-%d"),
        "end_date": working["date"].iloc[-1].strftime("%Y-%m-%d"),
        "latest_close": latest_close,
        "total_return": latest_close / first_close - 1 if first_close else 0.0,
        "volatility_20": float(volatility.iloc[-1]) if not volatility.empty else 0.0,
        "news_count": int(len(news)),
    }


def _row_to_news_item(row: pd.Series) -> dict[str, Any]:
    """把 DataFrame 行转成页面上更好用的新闻字典。"""

    publish_time = row.get("publish_time")
    if pd.notna(publish_time):
        publish_time_text = pd.to_datetime(publish_time).strftime("%Y-%m-%d %H:%M")
    else:
        publish_time_text = "-"

    content = str(row.get("content") or "")
    return {
        "title": str(row.get("title") or ""),
        "publish_time": publish_time_text,
        "source": str(row.get("source") or ""),
        "url": str(row.get("url") or ""),
        "content": content,
        "summary": content[:160] + ("..." if len(content) > 160 else ""),
    }


def normalize_news_title(title: str) -> str:
    """标题归一化，用于去重相似新闻。"""

    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(title))


def _news_event_score(title: str, content: str = "") -> int:
    """按宏观事件关键词给新闻排序，事件性强的排在前面。"""

    text = f"{title} {content}"
    keyword_score = sum(1 for keyword in EVENT_NEWS_KEYWORDS if keyword in text)
    return keyword_score + score_news_relevance(title, content)


def _count_keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    """统计方向关键词命中数。"""

    return sum(1 for keyword in keywords if keyword in text)


def classify_news_direction(title: str, content: str = "") -> str:
    """判断新闻方向。

    标题通常浓缩了行情方向，所以标题命中权重高于正文；正文只作为辅助。
    """

    title_text = str(title or "")
    content_text = str(content or "")
    bullish_score = _count_keyword_hits(title_text, BULLISH_KEYWORDS) * 3
    bearish_score = _count_keyword_hits(title_text, BEARISH_KEYWORDS) * 3
    bullish_score += _count_keyword_hits(content_text, BULLISH_KEYWORDS)
    bearish_score += _count_keyword_hits(content_text, BEARISH_KEYWORDS)

    if bullish_score > bearish_score:
        return "bullish"
    if bearish_score > bullish_score:
        return "bearish"
    return "related"


def _publish_sort_value(row: pd.Series) -> pd.Timestamp:
    """返回新闻排序用的发布时间，缺失值排到最后。"""

    value = pd.to_datetime(row.get("publish_time"), errors="coerce")
    if pd.isna(value):
        return pd.Timestamp.min
    return value


def classify_news_items(news: pd.DataFrame, limit: int | None = 6) -> dict[str, list[dict[str, Any]]]:
    """过滤、去重，并粗略区分利多、利空和普通相关新闻。"""

    groups: dict[str, list[dict[str, Any]]] = {
        "bullish": [],
        "bearish": [],
        "related": [],
    }
    if news.empty:
        return groups

    working = news.copy()
    if "publish_time" in working.columns:
        working = working.sort_values("publish_time", ascending=False, na_position="last")

    ranked_rows: list[tuple[int, pd.Series]] = []
    seen_titles: set[str] = set()
    for _, row in working.iterrows():
        title = str(row.get("title") or "")
        content = str(row.get("content") or "")
        normalized_title = normalize_news_title(title)
        if not normalized_title or normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)
        if is_low_quality_news(title, content):
            continue
        ranked_rows.append((_news_event_score(title, content), row))

    ranked_rows.sort(key=lambda pair: (_publish_sort_value(pair[1]), pair[0]), reverse=True)

    for _, row in ranked_rows:
        item = _row_to_news_item(row)
        bucket = classify_news_direction(
            str(row.get("title") or ""),
            str(row.get("content") or ""),
        )

        if limit is None or len(groups[bucket]) < limit:
            groups[bucket].append(item)

    return groups


def filter_price_range(
    prices: pd.DataFrame,
    time_range: str = "1Y",
    start_date: object | None = None,
    end_date: object | None = None,
) -> pd.DataFrame:
    """按时间范围筛选价格数据，避免默认全量图过度压缩。"""

    if prices.empty:
        return prices.copy()

    working = clean_price_data(prices)

    if time_range == "CUSTOM":
        if start_date is not None:
            working = working[working["date"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            working = working[working["date"] <= pd.to_datetime(end_date)]
        return working.reset_index(drop=True)

    if time_range == "ALL":
        return working

    days = TIME_RANGE_DAYS.get(time_range)
    if days is None:
        return working

    end_date = pd.to_datetime(working["date"].max())
    start_date = end_date - pd.Timedelta(days=days)
    return working[working["date"] >= start_date].reset_index(drop=True)


def calculate_visible_y_range(
    prices: pd.DataFrame,
    *,
    include_ma5: bool = True,
    include_ma20: bool = True,
    include_ma60: bool = False,
    chart_type: str = "line",
) -> list[float] | None:
    """按当前可见数据计算 Y 轴范围，避免长周期图压扁早期价格波动。"""

    if prices.empty:
        return None

    columns = ["close"]
    if chart_type == "candlestick":
        columns.extend(column for column in ["high", "low"] if column in prices.columns)

    ma_columns = [
        (include_ma5, "ma_5"),
        (include_ma20, "ma_20"),
        (include_ma60, "ma_60"),
    ]
    columns.extend(column for enabled, column in ma_columns if enabled and column in prices.columns)
    existing_columns = [column for column in columns if column in prices.columns]
    if not existing_columns:
        return None

    values = prices[existing_columns].apply(pd.to_numeric, errors="coerce").stack().dropna()
    if values.empty:
        return None

    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        padding = abs(min_value) * 0.05 or 1.0
    else:
        padding = (max_value - min_value) * 0.08
    return [max(0.0, min_value - padding), max_value + padding]


def create_naive_forecast(
    prices: pd.DataFrame, days: int = 10, lookback: int = 20
) -> pd.DataFrame:
    """生成一个非常简单的趋势延伸预测。

    这不是正式模型，只用于网页先把“预测区间如何展示”搭出来。
    后续 ARIMA/LSTM 完成后，可以直接替换这个函数的数据来源。
    """

    if prices.empty:
        return pd.DataFrame(columns=["date", "predicted_close"])

    working = clean_price_data(prices)
    recent = working.tail(max(2, lookback))
    if len(recent) < 2:
        step = 0.0
    else:
        step = (float(recent["close"].iloc[-1]) - float(recent["close"].iloc[0])) / (
            len(recent) - 1
        )

    last_date = pd.to_datetime(working["date"].iloc[-1])
    last_close = float(working["close"].iloc[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)
    predicted = [last_close + step * (index + 1) for index in range(days)]
    return pd.DataFrame({"date": future_dates, "predicted_close": predicted})


def create_price_figure(
    prices: pd.DataFrame,
    *,
    show_ma5: bool = True,
    show_ma20: bool = True,
    show_ma60: bool = False,
    show_forecast: bool = False,
    forecast_days: int = 10,
    time_range: str = "1Y",
    start_date: object | None = None,
    end_date: object | None = None,
    chart_type: str = "line",
) -> go.Figure:
    """创建 Plotly 金价走势图。"""

    ranged_prices = filter_price_range(prices, time_range, start_date, end_date)
    working = add_price_indicators(ranged_prices, windows=(5, 20, 60))
    fig = go.Figure()

    if chart_type == "candlestick" and {"open", "high", "low", "close"}.issubset(working.columns):
        fig.add_trace(
            go.Candlestick(
                x=working["date"],
                open=working["open"],
                high=working["high"],
                low=working["low"],
                close=working["close"],
                name="OHLC",
                increasing_line_color="#dc2626",
                decreasing_line_color="#16a34a",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=working["date"],
                y=working["close"],
                mode="lines",
                name="Close",
                line={"color": "#c77800", "width": 2.4},
                hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
            )
        )

    ma_options = [(show_ma5, 5), (show_ma20, 20), (show_ma60, 60)]
    colors = {5: "#2563eb", 20: "#059669", 60: "#7c3aed"}
    for enabled, window in ma_options:
        if enabled:
            fig.add_trace(
                go.Scatter(
                    x=working["date"],
                    y=working[f"ma_{window}"],
                    mode="lines",
                    name=f"MA{window}",
                    line={"color": colors[window], "width": 1.3},
                    hovertemplate="%{x|%Y-%m-%d}<br>MA: %{y:.2f}<extra></extra>",
                )
            )

    if show_forecast:
        forecast = create_naive_forecast(working, days=forecast_days)
        if not forecast.empty:
            forecast_start = forecast["date"].iloc[0]
            forecast_end = forecast["date"].iloc[-1]
            fig.add_vrect(
                x0=forecast_start,
                x1=forecast_end,
                fillcolor="#e5e7eb",
                opacity=0.45,
                line_width=0,
                layer="below",
                annotation_text="Forecast",
                annotation_position="top left",
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast["date"],
                    y=forecast["predicted_close"],
                    mode="lines",
                    name="Naive Forecast",
                    line={"color": "#64748b", "width": 2, "dash": "dash"},
                    hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        height=540,
        margin={"l": 48, "r": 24, "t": 46, "b": 48},
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="white",
        title="Gold Price Trend",
        xaxis_title="Date",
        yaxis_title="Price",
        font={"color": "#111827", "size": 13},
        title_font={"color": "#111827", "size": 18},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        tickfont={"color": "#334155"},
        title_font={"color": "#334155"},
    )
    y_range = calculate_visible_y_range(
        working,
        include_ma5=show_ma5,
        include_ma20=show_ma20,
        include_ma60=show_ma60,
        chart_type=chart_type,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        tickfont={"color": "#334155"},
        title_font={"color": "#334155"},
        autorange=y_range is None,
        range=y_range,
        fixedrange=False,
    )
    return fig


def current_timestamp() -> str:
    """返回页面更新完成时显示的时间戳。"""

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def escape_html_text(value: object) -> str:
    """转义准备放入 HTML 的文本，防止新闻内容破坏页面结构。"""

    import html

    return html.escape(str(value or ""), quote=True)
