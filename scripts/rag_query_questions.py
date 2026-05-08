"""
Поиск похожих вопросов в локальной RAG-базе.

Запуск:
    uv run python scripts/rag_query_questions.py --query "python asyncio event loop" --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import LocalRAGRetriever


def _run_raw(retriever: LocalRAGRetriever, query: str, top_k: int) -> list[dict]:
    """Чистый векторный поиск Chroma без гибридного rerank."""
    docs_scored = retriever.vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k=top_k,
        filter={"doc_type": "question"},
    )
    items: list[dict] = []
    for doc, score in docs_scored:
        items.append(
            {
                "relevance_score": score,
                "rank_score": None,
                "question_id": doc.metadata.get("question_id"),
                "question_title": doc.metadata.get("question_title", ""),
                "tags": doc.metadata.get("tags", ""),
            }
        )
    return items


def main():
    parser = argparse.ArgumentParser(description="Query similar questions from RAG")
    parser.add_argument("--query", required=True, help="Текст запроса")
    parser.add_argument("--top-k", type=int, default=5, help="Сколько результатов вернуть")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Сырой результат вектор-БД без гибридного rerank",
    )
    args = parser.parse_args()

    retriever = LocalRAGRetriever()
    if args.raw:
        results = _run_raw(retriever, args.query, args.top_k)
    else:
        results = retriever.retrieve_similar_questions(args.query, top_k=args.top_k)

    if not results:
        print("Ничего не найдено.")
        return

    mode = "raw vector" if args.raw else "reranked"
    print(f"Found {len(results)} results ({mode}):\n")
    for i, item in enumerate(results, 1):
        relevance = item.get("relevance_score")
        rank = item.get("rank_score")
        relevance_str = f"{relevance:.4f}" if isinstance(relevance, (int, float)) else "n/a"
        rank_str = f"{rank:.4f}" if isinstance(rank, (int, float)) else "n/a"
        print(
            f"{i}. relevance={relevance_str} rank={rank_str} "
            f"qid={item.get('question_id')} "
            f"title={item.get('question_title')!r} "
            f"tags={item.get('tags')}"
        )


if __name__ == "__main__":
    main()

