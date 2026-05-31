from crawler.database import get_connection, init_db, upsert_news_rows, upsert_price_rows
from analysis.run_analysis import generate_analysis_outputs


def test_generate_analysis_outputs(tmp_path):
    db_path = tmp_path / "gold.db"
    output_dir = tmp_path / "output"
    init_db(db_path)

    price_rows = [
        {
            "date": f"2026-05-{day:02d}",
            "open": 100.0 + day,
            "close": 100.0 + day * 2,
            "high": 102.0 + day * 2,
            "low": 98.0 + day,
            "volume": 1000 + day,
            "source": "test",
        }
        for day in range(1, 31)
    ]
    news_rows = [
        {
            "title": "黄金价格出现明显波动",
            "publish_time": "2026-05-15 10:00:00",
            "content": "测试新闻内容",
            "source": "test",
            "url": "https://example.com/news",
        }
    ]

    with get_connection(db_path) as conn:
        upsert_price_rows(conn, price_rows)
        upsert_news_rows(conn, news_rows)

    outputs = generate_analysis_outputs(db_path=db_path, output_dir=output_dir)

    assert outputs["price_rows"] == 30
    assert (output_dir / "price_summary.csv").exists()
    assert (output_dir / "return_statistics.csv").exists()
    assert (output_dir / "close_ma.png").exists()
    assert (output_dir / "return_histogram.png").exists()
    assert (output_dir / "rolling_volatility.png").exists()
