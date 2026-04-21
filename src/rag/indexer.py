"""
Индексатор локальной векторной базы для RAG.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_chroma import Chroma
except ModuleNotFoundError:  # pragma: no cover - fallback
    from langchain_community.vectorstores import Chroma

try:
    import settings
    from llm_provider import get_embeddings
except ModuleNotFoundError:  # pragma: no cover - fallback для запуска как src.*
    from src import settings
    from src.llm_provider import get_embeddings


NON_TECH_TAGS = {
    "личные",
    "опыт",
    "управление проектами",
    "карьера",
    "hr",
    "поведенческие",
    "soft skills",
    "софт скиллы",
}


@dataclass
class IndexStats:
    total_items: int
    technical_items: int
    question_docs: int
    answer_docs: int


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _normalize_tags(tags: list[str] | None) -> list[str]:
    clean: list[str] = []
    for tag in tags or []:
        value = _normalize_text(str(tag))
        if value:
            clean.append(value)
    return clean


def _normalize_tag_for_match(tag: str) -> str:
    return _normalize_text(tag).lower()


def is_technical_question(question: dict[str, Any]) -> bool:
    """
    Определить, является ли вопрос техническим.
    """
    tags = _normalize_tags(question.get("tags", []) or [])
    normalized_tags = {_normalize_tag_for_match(tag) for tag in tags}
    title = (question.get("title") or "").lower()
    slug = (question.get("slug") or "").lower()

    # Строгий режим: если есть хотя бы один non-tech тег, вопрос исключается.
    if normalized_tags & NON_TECH_TAGS:
        return False

    non_tech_keywords = (
        "зарплат",
        "о себе",
        "почему вы",
        "сильные стороны",
        "слабые стороны",
        "конфликт",
        "команд",
    )
    if any(k in title or k in slug for k in non_tech_keywords):
        return False

    # Если non-tech сигналов не найдено, считаем вопрос техническим.
    return True


def _answer_quality_score(answer: dict[str, Any]) -> tuple[int, int, int]:
    """
    Ключ сортировки ответов:
    1) is_best
    2) favorites_count
    3) длина текста
    """
    text = _normalize_text(answer.get("text", ""))
    return (
        1 if answer.get("is_best") else 0,
        int(answer.get("favorites_count", 0) or 0),
        len(text),
    )


def _is_low_quality_answer(text: str) -> bool:
    text_norm = _normalize_text(text).lower()
    if len(text_norm) < settings.RAG_MIN_ANSWER_CHARS:
        return True
    if text_norm in {"test", "test 11122331", "ok", "1111"}:
        return True
    return False


def _build_documents(dataset: list[dict[str, Any]]) -> tuple[list[Document], IndexStats]:
    docs: list[Document] = []
    technical_items = 0
    question_docs = 0
    answer_docs = 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
    )

    for item in dataset:
        question = item.get("question", {}) or {}
        if not is_technical_question(question):
            continue
        technical_items += 1

        qid = question.get("id")
        title = _normalize_text(question.get("title", ""))
        tags = _normalize_tags(question.get("tags", []) or [])
        tags_meta = tags if tags else ["unknown"]
        tags_lc = [_normalize_tag_for_match(tag) for tag in tags_meta]
        frequency = float(question.get("frequency", 0) or 0)
        tags_str = ", ".join(tags_meta)

        if not title:
            continue

        # Документ вопроса (для генерации похожих вопросов)
        q_content = f"Вопрос: {title}\nТеги: {tags_str}"
        docs.append(
            Document(
                page_content=q_content,
                metadata={
                    "doc_type": "question",
                    "question_id": qid,
                    "question_title": title,
                    "tags": tags_meta,
                    "tags_lc": tags_lc,
                    "frequency": frequency,
                },
            )
        )
        question_docs += 1

        # Документы ответов (эталоны для оценки)
        answers = item.get("answers", []) or []
        answers_sorted = sorted(answers, key=_answer_quality_score, reverse=True)
        top_answers = answers_sorted[: settings.RAG_MAX_ANSWERS_PER_QUESTION]

        for answer in top_answers:
            if answer.get("moderation_status") != "approved":
                continue
            answer_text = _normalize_text(answer.get("text", ""))
            if _is_low_quality_answer(answer_text):
                continue

            base_doc = Document(
                page_content=(
                    f"Вопрос-источник: {title}\n"
                    f"Теги: {tags_str}\n"
                    f"Ответ: {answer_text}"
                ),
                metadata={
                    "doc_type": "answer",
                    "question_id": qid,
                    "question_title": title,
                    "answer_id": answer.get("id"),
                    "tags": tags_meta,
                    "tags_lc": tags_lc,
                    "is_best": bool(answer.get("is_best", False)),
                    "favorites_count": int(answer.get("favorites_count", 0) or 0),
                },
            )
            split_docs = splitter.split_documents([base_doc])
            docs.extend(split_docs)
            answer_docs += len(split_docs)

    stats = IndexStats(
        total_items=len(dataset),
        technical_items=technical_items,
        question_docs=question_docs,
        answer_docs=answer_docs,
    )
    return docs, stats


def _batched(seq: list[Document], size: int):
    for i in range(0, len(seq), size):
        yield i // size + 1, seq[i : i + size]


def build_vector_index(
    json_path: str | Path = settings.RAG_SOURCE_JSON,
    persist_dir: str | Path = settings.RAG_DB_PATH,
    collection_name: str = settings.RAG_COLLECTION_NAME,
    limit: int | None = None,
    batch_size: int = 64,
) -> IndexStats:
    """
    Построить (или перестроить) локальный Chroma-индекс.
    """
    json_path = Path(json_path)
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open(encoding="utf-8") as f:
        dataset = json.load(f)

    if limit is not None and limit > 0:
        dataset = dataset[:limit]

    print(f"[RAG] Загружено записей: {len(dataset)}", flush=True)
    print("[RAG] Фильтрация и подготовка документов...", flush=True)
    docs, stats = _build_documents(dataset)
    if not docs:
        raise ValueError("Нет документов для индексации после фильтрации.")
    print(
        "[RAG] Подготовлено документов: "
        f"{len(docs)} (questions={stats.question_docs}, answers={stats.answer_docs})",
        flush=True,
    )

    print("[RAG] Инициализация embedding-модели...", flush=True)
    embeddings = get_embeddings()
    print("[RAG] Подключение к Chroma...", flush=True)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    try:
        vectorstore.delete_collection()
    except Exception:
        # Первая инициализация: коллекции может не быть.
        pass

    batch_size = max(1, int(batch_size))
    total_batches = (len(docs) + batch_size - 1) // batch_size
    print(
        f"[RAG] Индексация батчами: {total_batches} (batch_size={batch_size})",
        flush=True,
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    for batch_no, batch in _batched(docs, batch_size):
        vectorstore.add_documents(batch)
        print(f"[RAG] Batch {batch_no}/{total_batches} indexed", flush=True)

    # Для langchain_chroma персист выполняется автоматически.
    # Оставляем fallback для старых реализаций.
    persist_fn = getattr(vectorstore, "persist", None)
    if callable(persist_fn):
        persist_fn()
    print("[RAG] Индексация завершена.", flush=True)
    return stats

