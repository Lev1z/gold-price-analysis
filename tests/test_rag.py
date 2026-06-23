from crawler.database import get_connection, init_db, upsert_news_rows
from ai.rag.build_index import build_keyword_index, load_news_documents, split_text
from ai.rag.query import (
    build_price_context,
    build_prompt,
    detect_question_intent,
    load_env_file,
    retrieve_contexts,
)
from crawler.database import upsert_price_rows


def test_load_news_documents_reads_clean_text(tmp_path):
    db_path = tmp_path / "gold.db"
    init_db(db_path)
    with get_connection(db_path) as conn:
        upsert_news_rows(
            conn,
            [
                {
                    "title": "黄金价格上涨",
                    "publish_time": "2026-05-29 10:00:00",
                    "content": "避险需求上升，黄金价格走高。",
                    "source": "test",
                    "url": "https://example.com/a",
                }
            ],
        )

    docs = load_news_documents(db_path)

    assert len(docs) == 1
    assert docs[0]["title"] == "黄金价格上涨"
    assert "避险需求" in docs[0]["text"]


def test_load_news_documents_filters_low_quality_jewelry_cpi_news(tmp_path):
    db_path = tmp_path / "gold.db"
    init_db(db_path)
    with get_connection(db_path) as conn:
        upsert_news_rows(
            conn,
            [
                {
                    "title": "5月份江门CPI同比上涨1.4%黄金饰品价格上涨34.6%",
                    "publish_time": "2026-06-19 10:00:00",
                    "content": "居民消费价格指数上涨，黄金饰品价格涨幅较大。",
                    "source": "test",
                    "url": "https://example.com/local-cpi",
                },
                {
                    "title": "美国CPI低于预期，现货黄金走高",
                    "publish_time": "2026-06-20 10:00:00",
                    "content": "美元指数回落，美债收益率下行，国际金价受到支撑。",
                    "source": "test",
                    "url": "https://example.com/macro-gold",
                },
            ],
        )

    docs = load_news_documents(db_path)

    assert [doc["title"] for doc in docs] == ["美国CPI低于预期，现货黄金走高"]


def test_split_text_keeps_short_text_and_splits_long_text():
    short_chunks = split_text("短文本", chunk_size=20, overlap=5)
    long_chunks = split_text("黄金价格上涨。" * 20, chunk_size=30, overlap=5)

    assert short_chunks == ["短文本"]
    assert len(long_chunks) > 1
    assert all(chunk for chunk in long_chunks)


def test_build_keyword_index_and_retrieve_contexts():
    docs = [
        {
            "id": 1,
            "title": "黄金上涨",
            "publish_time": "2026-05-29",
            "source": "test",
            "url": "https://example.com/a",
            "text": "避险需求推动黄金价格上涨",
        },
        {
            "id": 2,
            "title": "原油新闻",
            "publish_time": "2026-05-29",
            "source": "test",
            "url": "https://example.com/b",
            "text": "原油价格变化",
        },
    ]
    index = build_keyword_index(docs)

    contexts = retrieve_contexts("黄金为什么上涨", index=index, top_k=1)

    assert len(contexts) == 1
    assert contexts[0]["title"] == "黄金上涨"


def test_build_prompt_contains_question_and_sources():
    contexts = [
        {
            "title": "黄金上涨",
            "publish_time": "2026-05-29",
            "source": "test",
            "url": "https://example.com/a",
            "text": "避险需求推动黄金价格上涨",
            "score": 3.0,
        }
    ]

    prompt = build_prompt("黄金为什么上涨？", contexts, price_context="最新收盘价：4500")

    assert "黄金为什么上涨？" in prompt
    assert "避险需求推动黄金价格上涨" in prompt
    assert "黄金上涨" in prompt
    assert "最新收盘价：4500" in prompt


def test_detect_question_intent_for_decision_and_event():
    assert detect_question_intent("最近值得入手吗") == "decision"
    assert detect_question_intent("2026年1月30日为什么暴跌") == "date_event"
    assert detect_question_intent("未来走势如何") == "forecast"


def test_build_price_context_contains_market_metrics(tmp_path):
    db_path = tmp_path / "gold.db"
    init_db(db_path)
    rows = []
    for index, day in enumerate(range(1, 41), start=1):
        close = 100.0 + index
        rows.append(
            {
                "date": f"2026-05-{day:02d}" if day <= 31 else f"2026-06-{day - 31:02d}",
                "open": close - 1,
                "close": close,
                "high": close + 1,
                "low": close - 2,
                "volume": 1000,
                "source": "test",
            }
        )
    with get_connection(db_path) as conn:
        upsert_price_rows(conn, rows)

    context = build_price_context("最近值得入手吗", db_path=db_path)

    assert "最新收盘价" in context
    assert "近7个交易日涨跌幅" in context
    assert "MA20" in context
    assert "问题类型：decision" in context


def test_retrieve_contexts_deduplicates_same_title_and_time():
    docs = [
        {
            "id": 1,
            "title": "黄金上涨",
            "publish_time": "2026-05-29",
            "source": "test",
            "url": "https://example.com/a",
            "text": "避险需求推动黄金价格上涨",
        },
        {
            "id": 2,
            "title": "黄金上涨",
            "publish_time": "2026-05-29",
            "source": "test",
            "url": "https://example.com/b",
            "text": "避险需求推动黄金价格上涨",
        },
    ]
    index = build_keyword_index(docs)

    contexts = retrieve_contexts("黄金上涨", index=index, top_k=4)

    assert len(contexts) == 1


def test_load_env_file_reads_key_value_pairs(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("DEEPSEEK_API_KEY=test-key\n# comment\nEMPTY=\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    load_env_file(env_path)

    assert __import__("os").getenv("DEEPSEEK_API_KEY") == "test-key"
