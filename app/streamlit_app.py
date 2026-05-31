"""黄金价格智能分析平台 Streamlit 页面。"""

from __future__ import annotations

import time

import streamlit as st

from ai.rag.build_index import build_all_indexes
from ai.rag.query import answer_question
from analysis.run_analysis import generate_analysis_outputs
from app.dashboard import (
    classify_news_items,
    compute_dashboard_metrics,
    create_price_figure,
    current_timestamp,
    escape_html_text,
    load_dashboard_data,
)
from crawler.config import DEFAULT_DB_PATH
from crawler.crawler_gold_price import fetch_gold_price_rows
from crawler.crawler_news import fetch_gold_news_rows
from crawler.database import (
    cleanup_duplicate_news,
    get_connection,
    init_db,
    upsert_news_rows,
    upsert_price_rows,
)


st.set_page_config(page_title="黄金价格智能分析平台", layout="wide")


def inject_styles() -> None:
    """为 Streamlit 页面补一点轻量 CSS，让展示更像一个成品页面。"""

    st.markdown(
        """
        <style>
        .stApp {
            background: #f8fafc;
            color: #111827;
        }
        .block-container {
            padding-top: 3.4rem;
            padding-bottom: 2.5rem;
            max-width: 1240px;
        }
        .app-title {
            font-size: 2rem;
            font-weight: 760;
            margin-bottom: 0.25rem;
            color: #111827;
        }
        .app-subtitle {
            color: #6b7280;
            margin-bottom: 1.25rem;
        }
        .metric-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            background: #ffffff;
        }
        .metric-label {
            color: #6b7280;
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-size: 1.15rem;
            font-weight: 720;
            color: #111827;
        }
        .timeline {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }
        .news-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.42rem 0.6rem;
            background: #ffffff;
            min-height: 0;
            display: grid;
            grid-template-columns: 7.8rem minmax(0, 1fr) 10rem;
            gap: 0.6rem;
            align-items: center;
        }
        .news-time {
            color: #6b7280;
            font-size: 0.72rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .news-title {
            font-weight: 680;
            color: #111827;
            font-size: 0.84rem;
            line-height: 1.25;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .news-source {
            color: #64748b;
            font-size: 0.72rem;
            text-align: right;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 720;
            margin-top: 1.2rem;
            margin-bottom: 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_dashboard_data() -> tuple:
    """缓存数据库读取结果；点击更新按钮后会清空缓存。"""

    return load_dashboard_data(DEFAULT_DB_PATH)


def run_manual_update(price_limit: int, news_limit: int, enrich_news: bool) -> str:
    """手动更新数据库、分析输出和 RAG 索引。"""

    init_db(DEFAULT_DB_PATH)
    price_rows = fetch_gold_price_rows(limit=price_limit)
    news_rows = fetch_gold_news_rows(limit=news_limit, enrich_articles=enrich_news)

    with get_connection(DEFAULT_DB_PATH) as conn:
        price_count = upsert_price_rows(conn, price_rows)
        news_count = upsert_news_rows(conn, news_rows)
        duplicate_count = cleanup_duplicate_news(conn)

    generate_analysis_outputs(DEFAULT_DB_PATH)
    build_all_indexes(DEFAULT_DB_PATH)
    cached_dashboard_data.clear()

    return (
        f"更新完成：金价 {price_count} 条，新闻 {news_count} 条，清理重复新闻 {duplicate_count} 条。"
        f" 更新时间：{current_timestamp()}"
    )


def format_percent(value: float) -> str:
    """格式化百分比。"""

    return f"{value * 100:.2f}%"


def render_metric_card(label: str, value: str) -> None:
    """渲染顶部指标卡。"""

    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_news_timeline(
    title: str,
    items: list[dict],
    empty_text: str,
    visible_count: int = 4,
) -> None:
    """渲染新闻时间轴卡片。"""

    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if not items:
        st.info(empty_text)
        return

    visible_items = items[:visible_count]
    hidden_items = items[visible_count:]
    _render_news_list(visible_items)
    if hidden_items:
        with st.expander(f"查看更多（{len(hidden_items)} 条）", expanded=False):
            _render_news_list(hidden_items)


def _render_news_list(items: list[dict]) -> None:
    """渲染单行新闻列表。"""

    cards = ['<div class="timeline">']
    for item in items:
        url = escape_html_text(item.get("url", ""))
        source = escape_html_text(item.get("source", ""))
        title_text = escape_html_text(item.get("title", ""))
        publish_time = escape_html_text(str(item.get("publish_time", "-"))[:10])
        if url:
            title_html = f'<a href="{url}" target="_blank">{title_text}</a>'
        else:
            title_html = title_text
        cards.append(
            f'<div class="news-card">'
            f'<div class="news-time">{publish_time}</div>'
            f'<div class="news-title">{title_html}</div>'
            f'<div class="news-source">{source}</div>'
            f'</div>'
        )
    cards.append("</div>")
    st.markdown("\n".join(cards), unsafe_allow_html=True)


def main() -> None:
    inject_styles()

    st.markdown('<div class="app-title">黄金价格智能分析平台</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">爬虫采集 · 数据分析 · 新闻时间轴 · RAG 问答 · 预测展示</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("数据更新")
        price_limit = st.number_input("金价日线数量", min_value=100, max_value=5000, value=1000, step=100)
        news_limit = st.number_input("新闻数量", min_value=10, max_value=300, value=100, step=10)
        enrich_news = st.checkbox("抓取新闻原文（较慢）", value=False)
        if st.button("更新数据到今天", type="primary"):
            with st.spinner("正在更新数据、生成分析结果并重建 RAG 索引..."):
                try:
                    message = run_manual_update(
                        price_limit=int(price_limit),
                        news_limit=int(news_limit),
                        enrich_news=bool(enrich_news),
                    )
                except Exception as exc:
                    st.error(f"更新失败：{exc}")
                else:
                    st.success(message)
                    time.sleep(0.5)
                    st.rerun()

        st.divider()
        st.header("图表设置")
        time_range = st.radio(
            "时间范围",
            options=["1M", "3M", "6M", "1Y", "ALL"],
            index=3,
            horizontal=True,
        )
        chart_type_label = st.radio(
            "图表类型",
            options=["收盘价走势", "K线图"],
            horizontal=True,
        )
        show_ma5 = st.checkbox("显示 MA5", value=True)
        show_ma20 = st.checkbox("显示 MA20", value=True)
        show_ma60 = st.checkbox("显示 MA60", value=False)
        show_forecast = st.toggle("显示预测区间", value=False)
        forecast_days = st.slider("预测交易日数量", min_value=5, max_value=30, value=10)
        st.caption("当前预测为简易趋势延伸，用于展示预测区间；后续可替换为 ARIMA/LSTM。")

    prices, news = cached_dashboard_data()
    metrics = compute_dashboard_metrics(prices, news)
    news_groups = classify_news_items(news)

    metric_columns = st.columns(5)
    with metric_columns[0]:
        render_metric_card("数据范围", f"{metrics['start_date']} ~ {metrics['end_date']}")
    with metric_columns[1]:
        render_metric_card("最新收盘价", f"{metrics['latest_close']:.2f}")
    with metric_columns[2]:
        render_metric_card("总涨跌幅", format_percent(float(metrics["total_return"])))
    with metric_columns[3]:
        render_metric_card("20日波动率", format_percent(float(metrics["volatility_20"])))
    with metric_columns[4]:
        render_metric_card("新闻数量", str(metrics["news_count"]))

    st.markdown('<div class="section-title">金价走势</div>', unsafe_allow_html=True)
    if prices.empty:
        st.warning("数据库中暂无金价数据，请先点击侧边栏的“更新数据到今天”。")
    else:
        fig = create_price_figure(
            prices,
            show_ma5=show_ma5,
            show_ma20=show_ma20,
            show_ma60=show_ma60,
            show_forecast=show_forecast,
            forecast_days=int(forecast_days),
            time_range=str(time_range),
            chart_type="candlestick" if chart_type_label == "K线图" else "line",
        )
        st.plotly_chart(fig, use_container_width=True)

    render_news_timeline(
        "利多新闻 / 可能推动上涨的事件",
        news_groups["bullish"],
        "暂未匹配到明显利多新闻。",
    )

    render_news_timeline(
        "利空新闻 / 可能推动下跌的事件",
        news_groups["bearish"],
        "暂未匹配到明显利空新闻。",
    )

    with st.expander("其他相关新闻", expanded=False):
        render_news_timeline("相关新闻", news_groups["related"], "暂无其他相关新闻。")

    st.markdown('<div class="section-title">RAG 问答</div>', unsafe_allow_html=True)
    question = st.text_input(
        "输入你想问的问题",
        placeholder="例如：最近黄金价格为什么波动？",
    )
    top_k = st.slider("检索新闻数量", min_value=1, max_value=8, value=4)
    use_chroma = st.checkbox("使用 Chroma 向量检索（可能较慢）", value=False)
    if st.button("提问", disabled=not question.strip()):
        with st.spinner("正在检索相关新闻并生成回答..."):
            st.write(answer_question(question, top_k=top_k, use_chroma=use_chroma))


if __name__ == "__main__":
    main()
