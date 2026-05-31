"""一键运行数据分析与可视化。

运行示例：

    python -m analysis.run_analysis

输出文件会放到 analysis/output/，可直接用于 PPT。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.clean_data import clean_news_data, clean_price_data
from analysis.load_data import load_news_data, load_price_data
from analysis.seasonality import monthly_average_close, monthly_return_summary
from analysis.statistics import (
    add_price_indicators,
    calculate_price_summary,
    calculate_return_statistics,
    find_key_price_moves,
)
from analysis.visualize_news import match_news_to_price_moves, plot_price_events
from analysis.visualize_prices import (
    plot_close_line,
    plot_close_with_moving_average,
    plot_monthly_average,
    plot_return_histogram,
    plot_rolling_volatility,
)
from crawler.config import DEFAULT_DB_PATH, PROJECT_ROOT


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "output"


def _save_dict_as_csv(data: dict[str, float], output_path: Path) -> None:
    """把指标字典保存成两列表格，便于打开查看。"""

    df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in data.items()]
    )
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def _save_events(events: list[dict[str, object]], output_path: Path) -> None:
    """保存关键事件匹配结果。"""

    pd.DataFrame(events).to_csv(output_path, index=False, encoding="utf-8-sig")


def generate_analysis_outputs(
    db_path: str | Path = DEFAULT_DB_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """读取数据库，生成统计结果和图表文件。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    price_df = clean_price_data(load_price_data(db_path))
    news_df = clean_news_data(load_news_data(db_path))
    if price_df.empty:
        raise ValueError("数据库中没有可分析的金价数据，请先运行 crawler。")

    price_with_indicators = add_price_indicators(price_df)
    price_summary = calculate_price_summary(price_df)
    return_stats = calculate_return_statistics(price_df)
    monthly_average = monthly_average_close(price_df)
    monthly_returns = monthly_return_summary(price_df)
    key_moves = find_key_price_moves(price_df, top_n=10)
    events = match_news_to_price_moves(key_moves, news_df)

    # 表格类输出：后续 PPT 或报告可以直接引用这些 CSV。
    _save_dict_as_csv(price_summary, output / "price_summary.csv")
    _save_dict_as_csv(return_stats, output / "return_statistics.csv")
    price_with_indicators.to_csv(output / "price_with_indicators.csv", index=False)
    monthly_average.to_csv(output / "monthly_average_close.csv", index=False)
    monthly_returns.to_csv(output / "monthly_return_summary.csv", index=False)
    key_moves.to_csv(output / "key_price_moves.csv", index=False)
    _save_events(events, output / "event_matches.csv")

    # 图表类输出：图名保持英文，避免 PPT 或不同电脑上出现乱码。
    plot_close_line(price_df, output / "close_line.png")
    plot_close_with_moving_average(price_df, output / "close_ma.png")
    plot_return_histogram(price_df, output / "return_histogram.png")
    plot_monthly_average(price_df, output / "monthly_average.png")
    plot_rolling_volatility(price_df, output / "rolling_volatility.png")
    plot_price_events(price_df, events, output / "price_events.png")

    return {
        "price_rows": int(len(price_df)),
        "news_rows": int(len(news_df)),
        "event_matches": int(len(events)),
        "output_dir": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成黄金价格分析图表和统计结果")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite 数据库路径")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="分析结果输出目录",
    )
    args = parser.parse_args()

    outputs = generate_analysis_outputs(
        db_path=Path(args.db_path),
        output_dir=Path(args.output_dir),
    )
    print("Analysis finished.")
    print(f"Price rows: {outputs['price_rows']}")
    print(f"News rows: {outputs['news_rows']}")
    print(f"Event matches: {outputs['event_matches']}")
    print(f"Output dir: {outputs['output_dir']}")


if __name__ == "__main__":
    main()
