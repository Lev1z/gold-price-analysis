"""黄金新闻 RAG 查询模块。

支持两种模式：
1. 配置 DEEPSEEK_API_KEY 后，检索相关新闻并调用 DeepSeek 生成回答。
2. 未配置 API Key 时，返回检索到的新闻摘要，保证演示不完全依赖外部 API。
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.clean_data import clean_price_data
from analysis.load_data import load_price_data
from analysis.statistics import add_price_indicators
from ai.rag.build_index import (
    CHROMA_DIR,
    EMBEDDING_MODEL_NAME,
    build_all_indexes,
    load_keyword_index,
    tokenize,
)
from crawler.config import DEFAULT_DB_PATH, PROJECT_ROOT


DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def detect_question_intent(question: str) -> str:
    """识别问题类型，让 prompt 采用更合适的回答方式。"""

    text = question.strip()
    if re.search(r"\d{4}年\d{1,2}月\d{1,2}日|为什么.*(暴跌|暴涨|下跌|上涨)", text):
        return "date_event"
    if any(word in text for word in ["值得", "入手", "买吗", "买入", "配置", "投资"]):
        return "decision"
    if any(word in text for word in ["未来", "预测", "走势", "会涨", "会跌"]):
        return "forecast"
    return "general"


def _period_return(prices, days: int) -> float | None:
    """计算近 N 个交易日涨跌幅。"""

    if len(prices) <= days:
        return None
    start = float(prices["close"].iloc[-days - 1])
    end = float(prices["close"].iloc[-1])
    if start == 0:
        return None
    return end / start - 1


def _format_percent(value: float | None) -> str:
    """格式化百分比，缺失时显示 N/A。"""

    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def build_price_context(
    question: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    """构建价格分析上下文，避免回答只依赖新闻。"""

    prices = clean_price_data(load_price_data(db_path))
    intent = detect_question_intent(question)
    if prices.empty:
        return f"问题类型：{intent}\n当前数据库中没有可用金价数据。"

    working = add_price_indicators(prices, windows=(20, 60))
    latest = working.iloc[-1]
    latest_close = float(latest["close"])
    latest_date = latest["date"].strftime("%Y-%m-%d")
    ma20 = float(latest["ma_20"]) if "ma_20" in working.columns else 0.0
    ma60 = float(latest["ma_60"]) if "ma_60" in working.columns else 0.0
    volatility_series = working["volatility_20"].dropna()
    volatility_20 = float(volatility_series.iloc[-1]) if not volatility_series.empty else None

    # 如果问题中出现具体日期，把该日期前后的交易数据也放进去。
    date_lines: list[str] = []
    date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", question)
    if date_match:
        year, month, day = (int(part) for part in date_match.groups())
        target = f"{year:04d}-{month:02d}-{day:02d}"
        target_date = pd.Timestamp(year=year, month=month, day=day)
        nearby = working[
            (working["date"] >= target_date - pd.Timedelta(days=5))
            & (working["date"] <= target_date + pd.Timedelta(days=5))
        ]
        date_lines.append(f"用户关注日期：{target}")
        if not nearby.empty:
            date_lines.append("关注日期附近交易数据：")
            for _, row in nearby.iterrows():
                daily_return = row.get("daily_return")
                daily_return_text = _format_percent(float(daily_return)) if daily_return == daily_return else "N/A"
                date_lines.append(
                    f"- {row['date'].strftime('%Y-%m-%d')}: open={row.get('open', 'N/A')}, "
                    f"close={row.get('close', 'N/A')}, high={row.get('high', 'N/A')}, "
                    f"low={row.get('low', 'N/A')}, 日涨跌幅={daily_return_text}"
                )

    trend_position = []
    trend_position.append("高于MA20" if latest_close >= ma20 else "低于MA20")
    trend_position.append("高于MA60" if latest_close >= ma60 else "低于MA60")

    lines = [
        f"问题类型：{intent}",
        f"金价数据范围：{working['date'].iloc[0].strftime('%Y-%m-%d')} 至 {latest_date}",
        f"最新收盘价：{latest_close:.2f}",
        f"近7个交易日涨跌幅：{_format_percent(_period_return(working, 7))}",
        f"近30个交易日涨跌幅：{_format_percent(_period_return(working, 30))}",
        f"近90个交易日涨跌幅：{_format_percent(_period_return(working, 90))}",
        f"MA20：{ma20:.2f}",
        f"MA60：{ma60:.2f}",
        f"当前趋势位置：{'，'.join(trend_position)}",
        f"20日波动率：{_format_percent(volatility_20)}",
    ]
    if date_lines:
        lines.extend(date_lines)
    return "\n".join(lines)


def load_env_file(path: str | Path = PROJECT_ROOT / ".env") -> None:
    """读取简单 .env 文件，避免额外依赖 python-dotenv。

    已经存在于系统环境变量中的值不会被覆盖。
    """

    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _deduplicate_contexts(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按标题和发布时间去重，减少同一篇新闻重复展示。"""

    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for item in contexts:
        key = (str(item.get("title", "")), str(item.get("publish_time", "")))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _score_keyword_result(question: str, item: dict[str, Any]) -> float:
    """用简单词重叠计算相关性分数。"""

    query_tokens = tokenize(question)
    if not query_tokens:
        return 0.0

    item_tokens = set(item.get("tokens", []))
    score = 0.0
    for token in query_tokens:
        if token in item_tokens:
            score += 2.0
        elif token in str(item.get("text", "")) or token in str(item.get("title", "")):
            score += 1.0
    return score


def retrieve_keyword_contexts(
    question: str,
    index: list[dict[str, Any]] | None = None,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """从关键词索引中检索相关文本片段。"""

    keyword_index = index if index is not None else load_keyword_index()
    scored: list[dict[str, Any]] = []
    for item in keyword_index:
        score = _score_keyword_result(question, item)
        if score <= 0:
            continue
        result = dict(item)
        result["score"] = score
        result.pop("tokens", None)
        scored.append(result)

    scored.sort(key=lambda row: row["score"], reverse=True)
    return _deduplicate_contexts(scored)[:top_k]


def retrieve_chroma_contexts(question: str, top_k: int = 4) -> list[dict[str, Any]]:
    """尝试从 Chroma 向量库检索相关文本片段。"""

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return []

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection("gold_news")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embedding = model.encode([question], normalize_embeddings=True).tolist()[0]
        result = collection.query(query_embeddings=[embedding], n_results=top_k)
    except Exception:
        return []

    contexts: list[dict[str, Any]] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for doc, metadata, distance in zip(documents, metadatas, distances):
        contexts.append(
            {
                "title": metadata.get("title", ""),
                "publish_time": metadata.get("publish_time", ""),
                "source": metadata.get("source", ""),
                "url": metadata.get("url", ""),
                "text": doc,
                "score": float(1.0 - distance) if distance is not None else 0.0,
            }
        )
    return _deduplicate_contexts(contexts)


def retrieve_contexts(
    question: str,
    index: list[dict[str, Any]] | None = None,
    top_k: int = 4,
    use_chroma: bool = False,
) -> list[dict[str, Any]]:
    """检索上下文。

    默认使用关键词索引，速度快且不依赖模型下载。
    如果显式启用 use_chroma，则优先尝试向量库，失败再降级关键词检索。
    """

    if use_chroma and index is None:
        chroma_contexts = retrieve_chroma_contexts(question, top_k=top_k)
        if chroma_contexts:
            return chroma_contexts

    return retrieve_keyword_contexts(question, index=index, top_k=top_k)


def build_prompt(
    question: str,
    contexts: list[dict[str, Any]],
    price_context: str = "",
) -> str:
    """把检索到的新闻片段拼成 DeepSeek 提示词。"""

    context_text = []
    for index, item in enumerate(contexts, start=1):
        context_text.append(
            "\n".join(
                [
                    f"[资料{index}]",
                    f"标题：{item.get('title', '')}",
                    f"时间：{item.get('publish_time', '')}",
                    f"来源：{item.get('source', '')}",
                    f"链接：{item.get('url', '')}",
                    f"内容：{item.get('text', '')}",
                ]
            )
        )

    joined_context = "\n\n".join(context_text) if context_text else "未检索到相关新闻。"
    intent = detect_question_intent(question)
    price_context_text = price_context or "未提供价格分析数据。"
    return f"""你是一个谨慎、客观的黄金价格分析助手。请基于“价格数据分析”和“新闻资料”回答问题，不要编造资料外的事实。
如果新闻资料不足，可以依据价格数据说明走势和风险，但必须明确“新闻原因无法完全确认”。

用户问题：
{question}

价格数据分析：
{price_context_text}

新闻资料：
{joined_context}

回答要求：
1. 先直接回答用户问题。
2. 明确区分“价格数据能说明什么”和“新闻资料可能说明什么”。
3. 如果是“是否值得买/入手”这类问题，只能给风险提示和分批/观望等非绝对建议，不能承诺收益。
4. 如果是具体日期涨跌原因，先引用该日前后的价格变化，再结合附近新闻解释；新闻不足时要说明不足。
5. 如果涉及价格涨跌，只能说“可能相关”，不要说成确定因果。
当前问题类型：{intent}
"""


def call_deepseek(prompt: str) -> str | None:
    """调用 DeepSeek API；未配置 Key 时返回 None。"""

    load_env_file()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "你是谨慎、客观的金融新闻分析助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def format_fallback_answer(
    question: str,
    contexts: list[dict[str, Any]],
    price_context: str = "",
) -> str:
    """没有 API Key 时，直接输出检索结果，仍然可以用于演示 RAG 检索。"""

    if not contexts:
        return f"问题：{question}\n\n没有检索到相关新闻。可以先运行 python -m ai.rag.build_index 重建索引。"

    lines = [
        f"问题：{question}",
        "",
        "未检测到 DEEPSEEK_API_KEY，以下是本地价格分析和相关新闻：",
        "",
        "价格分析：",
        price_context or "未提供价格分析数据。",
        "",
        "相关新闻：",
    ]
    for index, item in enumerate(contexts, start=1):
        lines.extend(
            [
                "",
                f"{index}. {item.get('title', '')}",
                f"   时间：{item.get('publish_time', '')}",
                f"   来源：{item.get('source', '')}",
                f"   摘要：{str(item.get('text', ''))[:180]}",
                f"   链接：{item.get('url', '')}",
            ]
        )
    return "\n".join(lines)


def answer_question(
    question: str,
    top_k: int = 4,
    auto_build: bool = True,
    use_chroma: bool = False,
) -> str:
    """检索相关新闻并回答问题。"""

    if auto_build and not (PROJECT_ROOT / "data" / "rag" / "keyword_index.json").exists():
        build_all_indexes(DEFAULT_DB_PATH)

    contexts = retrieve_contexts(question, top_k=top_k, use_chroma=use_chroma)
    price_context = build_price_context(question)
    prompt = build_prompt(question, contexts, price_context=price_context)
    answer = call_deepseek(prompt)
    if answer:
        return answer
    return format_fallback_answer(question, contexts, price_context=price_context)


def main() -> None:
    parser = argparse.ArgumentParser(description="黄金新闻 RAG 问答")
    parser.add_argument("question", nargs="*", help="要提问的问题")
    parser.add_argument("--top-k", type=int, default=4, help="检索多少条上下文")
    parser.add_argument(
        "--use-chroma",
        action="store_true",
        help="优先使用 Chroma 向量库检索；默认使用更稳的关键词索引",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="查询前先重建索引",
    )
    args = parser.parse_args()

    if args.rebuild:
        build_all_indexes(DEFAULT_DB_PATH)

    question = " ".join(args.question).strip()
    if not question:
        question = input("Question: ").strip()

    print(answer_question(question, top_k=args.top_k, use_chroma=args.use_chroma))


if __name__ == "__main__":
    main()
