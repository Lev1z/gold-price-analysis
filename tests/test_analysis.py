import pandas as pd

from analysis.clean_data import clean_news_data, clean_price_data
from analysis.seasonality import monthly_average_close, monthly_return_summary
from analysis.statistics import (
    add_price_indicators,
    calculate_price_summary,
    calculate_return_statistics,
    find_key_price_moves,
)
from analysis.visualize_news import match_news_to_price_moves
from analysis.visualize_prices import (
    plot_close_with_moving_average,
    plot_return_histogram,
)


def test_calculate_return_statistics():
    df = pd.DataFrame(
        {
            "date": ["2026-05-24", "2026-05-25", "2026-05-26"],
            "close": [100.0, 110.0, 121.0],
        }
    )

    stats = calculate_return_statistics(df)

    assert stats["count"] == 2.0
    assert round(stats["mean_return"], 6) == 0.1
    assert round(stats["std_return"], 6) == 0.0
    assert round(stats["max_return"], 6) == 0.1
    assert round(stats["min_return"], 6) == 0.1


def test_clean_price_data_sorts_and_deduplicates():
    df = pd.DataFrame(
        {
            "date": ["2026-05-26", "2026-05-25", "2026-05-26", None],
            "close": [121.0, 100.0, 122.0, 130.0],
        }
    )

    cleaned = clean_price_data(df)

    assert cleaned["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-05-25",
        "2026-05-26",
    ]
    assert cleaned["close"].tolist() == [100.0, 122.0]


def test_add_price_indicators_creates_returns_and_moving_averages():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-01", periods=5),
            "close": [100.0, 102.0, 101.0, 105.0, 110.0],
        }
    )

    result = add_price_indicators(df, windows=(2, 3))

    assert "daily_return" in result.columns
    assert "ma_2" in result.columns
    assert "ma_3" in result.columns
    assert round(result.loc[1, "daily_return"], 6) == 0.02
    assert round(result.loc[2, "ma_2"], 6) == 101.5


def test_calculate_price_summary():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-01", periods=3),
            "close": [100.0, 110.0, 105.0],
            "high": [101.0, 112.0, 106.0],
            "low": [99.0, 108.0, 103.0],
        }
    )

    summary = calculate_price_summary(df)

    assert summary["row_count"] == 3.0
    assert summary["start_close"] == 100.0
    assert summary["end_close"] == 105.0
    assert summary["max_high"] == 112.0
    assert summary["min_low"] == 99.0


def test_find_key_price_moves_returns_largest_absolute_moves():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-01", periods=5),
            "close": [100.0, 110.0, 108.0, 90.0, 99.0],
        }
    )

    moves = find_key_price_moves(df, top_n=2)

    assert len(moves) == 2
    assert moves.iloc[0]["date"].strftime("%Y-%m-%d") == "2026-05-04"


def test_monthly_return_summary():
    df = pd.DataFrame(
        {
            "date": ["2026-05-01", "2026-05-31", "2026-06-01", "2026-06-30"],
            "close": [100.0, 110.0, 120.0, 114.0],
        }
    )

    summary = monthly_return_summary(df)

    assert summary["month"].dt.strftime("%Y-%m").tolist() == ["2026-05", "2026-06"]
    assert round(summary.loc[0, "monthly_return"], 6) == 0.1
    assert round(summary.loc[1, "monthly_return"], 6) == -0.05


def test_monthly_average_close():
    df = pd.DataFrame(
        {
            "date": ["2026-05-01", "2026-05-31", "2026-06-01"],
            "close": [100.0, 110.0, 120.0],
        }
    )

    monthly = monthly_average_close(df)

    assert monthly["close"].tolist() == [105.0, 120.0]


def test_match_news_to_price_moves():
    moves = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-26"]),
            "daily_return": [0.05],
        }
    )
    news = pd.DataFrame(
        {
            "title": ["黄金价格上涨"],
            "publish_time": pd.to_datetime(["2026-05-26 10:00:00"]),
            "source": ["test"],
            "url": ["https://example.com"],
        }
    )

    matched = match_news_to_price_moves(moves, news)

    assert matched[0]["title"] == "黄金价格上涨"


def test_plot_functions_create_files(tmp_path):
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-01", periods=5),
            "close": [100.0, 102.0, 101.0, 105.0, 110.0],
        }
    )
    df = add_price_indicators(df, windows=(2,))

    trend_path = plot_close_with_moving_average(df, tmp_path / "trend.png", windows=(2,))
    hist_path = plot_return_histogram(df, tmp_path / "hist.png")

    assert trend_path.exists()
    assert hist_path.exists()
