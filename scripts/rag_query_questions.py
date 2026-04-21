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


def main():
    parser = argparse.ArgumentParser(description="Query similar questions from RAG")
    parser.add_argument("--query", required=True, help="Текст запроса")
    parser.add_argument("--top-k", type=int, default=5, help="Сколько результатов вернуть")
    args = parser.parse_args()

    retriever = LocalRAGRetriever()
    results = retriever.retrieve_similar_questions(args.query, top_k=args.top_k)

    if not results:
        print("Ничего не найдено.")
        return

    print(f"Found {len(results)} results:\n")
    for i, item in enumerate(results, 1):
        print(
            f"{i}. qid={item.get('question_id')} "
            f"title={item.get('question_title')!r} "
            f"tags={item.get('tags')}"
        )


if __name__ == "__main__":
    main()

