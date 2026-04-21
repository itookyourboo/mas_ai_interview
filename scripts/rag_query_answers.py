"""
Поиск эталонных ответов в локальной RAG-базе.

Запуск:
    uv run python scripts/rag_query_answers.py \
      --question "Что такое декоратор в Python?" \
      --tags "Python,Декораторы" \
      --top-k 3
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
    parser = argparse.ArgumentParser(description="Query reference answers from RAG")
    parser.add_argument("--question", required=True, help="Текст вопроса")
    parser.add_argument(
        "--tags",
        default="",
        help="Теги через запятую (пример: Python,Декораторы)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Сколько результатов вернуть")
    parser.add_argument("--snippet", type=int, default=300, help="Длина предпросмотра ответа")
    args = parser.parse_args()

    tags = [x.strip() for x in args.tags.split(",") if x.strip()]
    retriever = LocalRAGRetriever()
    results = retriever.retrieve_reference_answers(
        question_text=args.question,
        tags=tags,
        top_k=args.top_k,
    )

    if not results:
        print("Эталонные ответы не найдены.")
        return

    print(f"Found {len(results)} reference answers:\n")
    for i, item in enumerate(results, 1):
        snippet = item.get("content", "").replace("\n", " ")[: args.snippet]
        print(
            f"{i}. answer_id={item.get('answer_id')} "
            f"source={item.get('question_title')!r} "
            f"tags={item.get('tags')} "
            f"is_best={item.get('is_best')} "
            f"favorites={item.get('favorites_count')}"
        )
        print(f"   {snippet}...\n")


if __name__ == "__main__":
    main()

