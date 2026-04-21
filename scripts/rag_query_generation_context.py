"""
Проверка generation-контекста из локальной RAG-базы.

Запуск:
    uv run python scripts/rag_query_generation_context.py \
      --position "Backend Python Developer" \
      --tech-stack "Python, FastAPI, PostgreSQL" \
      --level "Junior" \
      --topics "API,Асинхронность" \
      --question-type "код" \
      --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retriever import LocalRAGRetriever


def _parse_topics(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Build generation context from RAG")
    parser.add_argument("--position", required=True, help="Целевая позиция кандидата")
    parser.add_argument("--tech-stack", required=True, help="Стек через запятую")
    parser.add_argument("--level", required=True, help="Уровень кандидата")
    parser.add_argument("--topics", default="", help="Темы через запятую")
    parser.add_argument("--question-type", required=True, help="Тип вопроса (теория/код/дизайн)")
    parser.add_argument("--top-k", type=int, default=5, help="Сколько примеров использовать")
    args = parser.parse_args()

    retriever = LocalRAGRetriever()
    topics = _parse_topics(args.topics)

    # Технический вывод кандидатов и их generation-rank.
    candidates = retriever._retrieve_generation_questions(  # noqa: SLF001
        position=args.position,
        tech_stack=args.tech_stack,
        level=args.level,
        topics=topics,
        question_type=args.question_type,
        top_k=args.top_k,
    )
    if candidates:
        print(f"Found {len(candidates)} generation candidates:\n")
        for i, item in enumerate(candidates, 1):
            print(
                f"{i}. qid={item.get('question_id')} "
                f"rank={item.get('generation_rank_score')} "
                f"type_match={item.get('generation_type_match')} "
                f"title={item.get('question_title')!r} "
                f"tags={item.get('tags')}"
            )
        print()
    else:
        print("Generation candidates not found.\n")

    context = retriever.build_context_for_generation(
        position=args.position,
        tech_stack=args.tech_stack,
        level=args.level,
        topics=topics,
        question_type=args.question_type,
        top_k=args.top_k,
    )
    print("=== Generation Context ===")
    print(context)


if __name__ == "__main__":
    main()
