import pandas as pd

from app.dashboard import (
    classify_news_items,
    compute_dashboard_metrics,
    create_naive_forecast,
    create_price_figure,
    escape_html_text,
    filter_price_range,
    normalize_news_title,
)


def test_compute_dashboard_metrics():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-01", periods=25),
            "close": [100.0 + index for index in range(25)],
        }
    )
    news = pd.DataFrame({"title": ["a", "b", "c"]})

    metrics = compute_dashboard_metrics(prices, news)

    assert metrics["start_date"] == "2026-05-01"
    assert metrics["end_date"] == "2026-05-25"
    assert metrics["latest_close"] == 124.0
    assert round(metrics["total_return"], 6) == 0.24
    assert metrics["news_count"] == 3


def test_classify_news_items():
    news = pd.DataFrame(
        {
            "title": ["避险需求推动黄金上涨", "美债收益率上行金价承压", "黄金市场观察"],
            "content": ["", "", ""],
            "publish_time": pd.to_datetime(
                ["2026-05-01 10:00:00", "2026-05-02 10:00:00", "2026-05-03 10:00:00"]
            ),
            "source": ["s1", "s2", "s3"],
            "url": ["u1", "u2", "u3"],
        }
    )

    groups = classify_news_items(news)

    assert groups["bullish"][0]["title"] == "避险需求推动黄金上涨"
    assert groups["bearish"][0]["title"] == "美债收益率上行金价承压"
    assert groups["related"][0]["title"] == "黄金市场观察"


def test_classify_news_items_filters_retail_and_deduplicates_titles():
    news = pd.DataFrame(
        {
            "title": [
                "今日金价5月29日多少钱一克",
                "美联储降息预期升温，黄金避险需求上升",
                "美联储降息预期升温 黄金避险需求上升",
                "美债收益率上行，金价承压",
            ],
            "content": ["", "", "", ""],
            "publish_time": pd.to_datetime(
                [
                    "2026-05-01 10:00:00",
                    "2026-05-02 10:00:00",
                    "2026-05-02 11:00:00",
                    "2026-05-03 10:00:00",
                ]
            ),
            "source": ["s1", "s2", "s3", "s4"],
            "url": ["u1", "u2", "u3", "u4"],
        }
    )

    groups = classify_news_items(news, limit=10)

    all_titles = [item["title"] for group in groups.values() for item in group]
    assert "今日金价5月29日多少钱一克" not in all_titles
    normalized_titles = [normalize_news_title(title) for title in all_titles]
    assert normalized_titles.count("美联储降息预期升温黄金避险需求上升") == 1
    assert "美联储降息预期升温黄金避险需求上升" in normalized_titles
    assert groups["bearish"][0]["title"] == "美债收益率上行，金价承压"


def test_normalize_news_title_removes_punctuation():
    assert normalize_news_title("【黄金收评】美伊停火！黄金“大变脸”") == "黄金收评美伊停火黄金大变脸"


def test_create_naive_forecast_extends_after_last_date():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-01", periods=10),
            "close": [100.0 + index for index in range(10)],
        }
    )

    forecast = create_naive_forecast(prices, days=3, lookback=5)

    assert len(forecast) == 3
    assert forecast.iloc[0]["date"].strftime("%Y-%m-%d") == "2026-05-11"
    assert forecast.iloc[-1]["predicted_close"] > forecast.iloc[0]["predicted_close"]


def test_filter_price_range_keeps_recent_year():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=800),
            "close": [100.0 + index for index in range(800)],
        }
    )

    filtered = filter_price_range(prices, "1Y")

    assert filtered["date"].min() > pd.Timestamp("2024-12-01")
    assert filtered["date"].max() == prices["date"].max()


def test_create_price_figure_can_render_candlestick():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-01", periods=5),
            "open": [100, 101, 102, 103, 104],
            "high": [103, 104, 105, 106, 107],
            "low": [99, 100, 101, 102, 103],
            "close": [102, 103, 104, 105, 106],
        }
    )

    fig = create_price_figure(prices, chart_type="candlestick", time_range="ALL")

    assert fig.data[0].type == "candlestick"


def test_escape_html_text_prevents_markup_injection():
    escaped = escape_html_text('<a href="bad">新闻</a>')

    assert escaped == "&lt;a href=&quot;bad&quot;&gt;新闻&lt;/a&gt;"
