"""
Показать статистику локальной RAG-базы (Chroma).

Запуск:
    uv run python scripts/rag_db_stats.py
    uv run python scripts/rag_db_stats.py --sample 10
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import settings


def main():
    parser = argparse.ArgumentParser(description="RAG DB stats")
    parser.add_argument("--sample", type=int, default=5, help="Количество примеров документов")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=settings.RAG_DB_PATH)
    collection = client.get_collection(settings.RAG_COLLECTION_NAME)

    total = collection.count()
    print(f"DB path: {settings.RAG_DB_PATH}")
    print(f"Collection: {settings.RAG_COLLECTION_NAME}")
    print(f"Total vectors: {total}")

    limit = min(max(args.sample, 1), total) if total > 0 else 0
    if limit == 0:
        print("Collection is empty.")
        return

    rows = collection.get(include=["metadatas", "documents"], limit=limit)
    metadatas = rows.get("metadatas", []) or []
    docs = rows.get("documents", []) or []

    doc_type_counter = Counter((m or {}).get("doc_type", "unknown") for m in metadatas)
    print("\nSample doc_type distribution:")
    for k, v in doc_type_counter.items():
        print(f"  - {k}: {v}")

    print("\nSample documents:")
    for i, (m, d) in enumerate(zip(metadatas, docs), 1):
        md = m or {}
        snippet = (d or "").replace("\n", " ")[:180]
        print(
            f"{i}. doc_type={md.get('doc_type')} "
            f"qid={md.get('question_id')} "
            f"title={md.get('question_title')!r}"
        )
        print(f"   snippet: {snippet}...")


if __name__ == "__main__":
    main()

