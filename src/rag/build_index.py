"""
CLI для построения локального RAG-индекса.
"""

import argparse

from .indexer import build_vector_index


def main():
    parser = argparse.ArgumentParser(description="Build local RAG vector index")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Индексировать только первые N записей (для smoke-test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Размер батча при индексации",
    )
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None
    stats = build_vector_index(limit=limit, batch_size=args.batch_size)
    print("RAG-индекс построен.")
    print(f"Всего записей: {stats.total_items}")
    print(f"Технических записей: {stats.technical_items}")
    print(f"Документов вопросов: {stats.question_docs}")
    print(f"Документов ответов: {stats.answer_docs}")


if __name__ == "__main__":
    main()

