"""构建黄金新闻 RAG 索引。

默认会先生成一个轻量的关键词索引文件，保证没有额外模型也能检索。
如果本机安装了 chromadb 和 sentence-transformers，则额外构建 Chroma 向量库。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from analysis.clean_data import clean_news_data
from analysis.load_data import load_news_data
from analysis.news_quality import is_low_quality_news
from crawler.config import DEFAULT_DB_PATH, PROJECT_ROOT


RAG_DIR = PROJECT_ROOT / "data" / "rag"
KEYWORD_INDEX_PATH = RAG_DIR / "keyword_index.json"
CHROMA_DIR = RAG_DIR / "chroma"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_news_documents(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """从 SQLite 读取新闻，并组装成 RAG 文档。

    每条文档保留 title/source/url/publish_time，回答时可以展示引用来源。
    """

    news_df = clean_news_data(load_news_data(db_path))
    documents: list[dict[str, Any]] = []
    for _, row in news_df.iterrows():
        title = str(row.get("title") or "").strip()
        content = str(row.get("content") or "").strip()
        if not title and not content:
            continue
        if is_low_quality_news(title, content):
            continue

        text = "\n".join(part for part in [title, content] if part)
        documents.append(
            {
                "id": int(row.get("id")) if row.get("id") is not None else len(documents) + 1,
                "title": title,
                "publish_time": str(row.get("publish_time") or ""),
                "source": str(row.get("source") or ""),
                "url": str(row.get("url") or ""),
                "text": text,
            }
        )

    return documents


def split_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """把长文本切成重叠片段，避免单条新闻过长影响检索。"""

    normalized = " ".join(str(text).split())
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = start + chunk_size
        chunks.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def tokenize(text: str) -> list[str]:
    """简单分词：中文按连续汉字片段，英文/数字按单词。

    这是保底检索方案，不追求 NLP 精细度，但足够让 RAG 先跑通。
    """

    raw_tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}", text.lower())
    tokens: list[str] = []
    for token in raw_tokens:
        tokens.append(token)
        # 中文连续片段通常没有空格，额外加入 2 字滑动词能提升保底检索命中率。
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
    return tokens


def build_keyword_index(
    documents: list[dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[dict[str, Any]]:
    """构建关键词检索索引。"""

    index: list[dict[str, Any]] = []
    for doc in documents:
        for chunk_id, chunk in enumerate(split_text(str(doc["text"]), chunk_size, overlap)):
            tokens = tokenize(chunk + " " + str(doc.get("title", "")))
            index.append(
                {
                    "id": f"{doc['id']}-{chunk_id}",
                    "doc_id": doc["id"],
                    "title": doc.get("title", ""),
                    "publish_time": doc.get("publish_time", ""),
                    "source": doc.get("source", ""),
                    "url": doc.get("url", ""),
                    "text": chunk,
                    "tokens": tokens,
                }
            )
    return index


def save_keyword_index(index: list[dict[str, Any]], path: str | Path = KEYWORD_INDEX_PATH) -> Path:
    """把关键词索引保存为 JSON 文件。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def load_keyword_index(path: str | Path = KEYWORD_INDEX_PATH) -> list[dict[str, Any]]:
    """读取关键词索引；不存在时返回空列表。"""

    index_path = Path(path)
    if not index_path.exists():
        return []
    return json.loads(index_path.read_text(encoding="utf-8"))


def build_chroma_index(
    documents: list[dict[str, Any]],
    persist_dir: str | Path = CHROMA_DIR,
    collection_name: str = "gold_news",
) -> bool:
    """尝试构建 Chroma 向量索引。

    返回 True 表示成功；如果依赖没装或模型下载失败，返回 False，
    程序仍可使用关键词索引作为保底检索。
    """

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return False

    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        client = chromadb.PersistentClient(path=str(persist_dir))
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = client.get_or_create_collection(collection_name)

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, str]] = []
        for item in build_keyword_index(documents):
            ids.append(str(item["id"]))
            texts.append(str(item["text"]))
            metadatas.append(
                {
                    "title": str(item.get("title", "")),
                    "publish_time": str(item.get("publish_time", "")),
                    "source": str(item.get("source", "")),
                    "url": str(item.get("url", "")),
                }
            )

        if not texts:
            return False

        embeddings = model.encode(texts, normalize_embeddings=True).tolist()
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        return True
    except Exception as exc:
        print(f"Warning: Chroma index build failed, fallback to keyword index: {exc}")
        return False


def build_all_indexes(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """构建所有可用索引，并返回摘要信息。"""

    documents = load_news_documents(db_path)
    keyword_index = build_keyword_index(documents)
    keyword_path = save_keyword_index(keyword_index)
    chroma_ok = build_chroma_index(documents)
    return {
        "documents": len(documents),
        "chunks": len(keyword_index),
        "keyword_index": str(keyword_path),
        "chroma_enabled": chroma_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构建黄金新闻 RAG 索引")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite 数据库路径")
    args = parser.parse_args()

    result = build_all_indexes(Path(args.db_path))
    print("RAG index built.")
    print(f"Documents: {result['documents']}")
    print(f"Chunks: {result['chunks']}")
    print(f"Keyword index: {result['keyword_index']}")
    print(f"Chroma enabled: {result['chroma_enabled']}")


if __name__ == "__main__":
    main()
