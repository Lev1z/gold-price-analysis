"""Query the RAG system."""

from __future__ import annotations


def answer_question(question: str) -> str:
    """Return a placeholder answer until the vector index is implemented."""

    return (
        "RAG query placeholder. After indexing news, this function will retrieve "
        f"relevant articles and ask DeepSeek to answer: {question}"
    )


def main() -> None:
    question = input("Question: ").strip()
    print(answer_question(question))


if __name__ == "__main__":
    main()
