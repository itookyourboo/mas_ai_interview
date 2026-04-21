"""
Retriever для локального RAG.
"""

from __future__ import annotations

import re
from typing import Any

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

try:
    from debug_utils import debug_log
except ModuleNotFoundError:  # pragma: no cover - fallback для запуска как src.*
    from src.debug_utils import debug_log


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tags_to_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


def _safe_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    # Часто бэкенды возвращают distance (>1 или >=0), а не similarity [0..1].
    if score > 1:
        return 1.0 / (1.0 + score)
    # Если это косинус в диапазоне [-1..1], переводим в [0..1].
    if score < 0:
        return max(0.0, min(1.0, (score + 1.0) / 2.0))
    return max(0.0, min(1.0, score))


def _lexical_overlap_ratio(a: str, b: str) -> float:
    tokens_a = {t for t in re.findall(r"\w+", _normalize(a)) if len(t) > 2}
    tokens_b = {t for t in re.findall(r"\w+", _normalize(b)) if len(t) > 2}
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def _tag_overlap_ratio(query_tags: list[str], doc_tags: list[str]) -> float:
    q = {_normalize(t) for t in query_tags if _normalize(t)}
    d = {_normalize(t) for t in doc_tags if _normalize(t)}
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


def _question_type_match_score(expected_type: str, title: str, tags: list[str]) -> float:
    """
    Эвристика совпадения типа вопроса для generation-rerank.
    """
    et = _normalize(expected_type)
    haystack = f"{_normalize(title)} {' '.join(_normalize(t) for t in tags)}"
    if "код" in et:
        markers = ("код", "sql", "запрос", "функц", "реализ", "алгоритм")
    elif "дизайн" in et:
        markers = ("дизайн", "архитект", "спроект", "endpoint", "сервис", "system")
    elif "отлад" in et:
        markers = ("ошиб", "debug", "почему не работает", "исправ", "отлад")
    else:
        markers = ("что такое", "разница", "объясн", "зачем", "когда")
    if any(m in haystack for m in markers):
        return 1.0
    return 0.0


def _stack_tokens(tech_stack: str) -> list[str]:
    tokens: list[str] = []
    for token in re.split(r"[,/|;]", tech_stack or ""):
        value = _normalize(token)
        if len(value) >= 2:
            tokens.append(value)
    return tokens


def _validated_top_k(value: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = 1
    return max(1, v)


class LocalRAGRetriever:
    """Обертка над Chroma для retrieval-вызовов проекта."""

    def __init__(
        self,
        persist_directory: str = settings.RAG_DB_PATH,
        collection_name: str = settings.RAG_COLLECTION_NAME,
    ):
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=persist_directory,
        )

    def _exact_question_matches(self, question_text: str) -> list[dict[str, Any]]:
        """
        Пытаемся получить точные совпадения вопроса по тексту из question-документов.
        """
        normalized_query = _normalize(question_text)
        matches: list[dict[str, Any]] = []

        collection = getattr(self.vectorstore, "_collection", None)
        if collection is None:
            return matches

        try:
            payload = collection.get(
                where={"doc_type": "question"},
                where_document={"$contains": question_text},
                include=["metadatas", "documents"],
            )
        except Exception:
            return matches

        metadatas = payload.get("metadatas", []) or []
        documents = payload.get("documents", []) or []
        for metadata, content in zip(metadatas, documents):
            title = (metadata or {}).get("question_title", "")
            if _normalize(title) != normalized_query:
                continue
            matches.append(
                {
                    "question_title": title,
                    "tags": _tags_to_list((metadata or {}).get("tags", "")),
                    "question_id": (metadata or {}).get("question_id"),
                    "content": content or "",
                    "relevance_score": 1.0,
                    "lexical_exact": True,
                }
            )
        return matches

    def _retrieve_candidate_questions(
        self,
        query: str,
        tags: list[str],
        top_k: int,
        *,
        use_lexical_signal: bool = True,
    ) -> list[dict[str, Any]]:
        top_k = _validated_top_k(top_k)
        query_norm = _normalize(query)
        requested_k = top_k * 4
        docs_with_scores: list[tuple[Any, float | None]] = []
        try:
            docs_scored = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=requested_k,
                filter={"doc_type": "question"},
            )
            docs_with_scores = [(doc, _safe_score(score)) for doc, score in docs_scored]
        except Exception:
            docs = self.vectorstore.similarity_search(
                query=query,
                k=requested_k,
                filter={"doc_type": "question"},
            )
            docs_with_scores = [(doc, None) for doc in docs]

        ranked: list[dict[str, Any]] = []
        for doc, relevance_score in docs_with_scores:
            title = doc.metadata.get("question_title", "")
            title_norm = _normalize(title)
            doc_tags = _tags_to_list(doc.metadata.get("tags", ""))
            lexical_exact = title_norm == query_norm
            lexical_soft = query_norm in title_norm or title_norm in query_norm
            lexical_overlap = _lexical_overlap_ratio(query, title) if settings.RAG_HYBRID_LEXICAL_BOOST else 0.0
            base = relevance_score if relevance_score is not None else 0.45
            lexical_signal = (
                (1.0 if lexical_exact else 0.6 if lexical_soft else 0.0)
                if settings.RAG_HYBRID_LEXICAL_BOOST and use_lexical_signal
                else 0.0
            )
            if settings.RAG_HYBRID_LEXICAL_BOOST:
                tag_overlap = _tag_overlap_ratio(tags, doc_tags)
                if use_lexical_signal:
                    rank_score = (
                        0.55 * base
                        + 0.25 * lexical_signal
                        + 0.10 * lexical_overlap
                        + 0.10 * tag_overlap
                    )
                else:
                    # Для generation-пути без lexical_signal оставляем диапазон [0..1].
                    rank_score = 0.80 * base + 0.12 * lexical_overlap + 0.08 * tag_overlap
            else:
                rank_score = 0.85 * base + 0.15 * _tag_overlap_ratio(tags, doc_tags)
            ranked.append(
                {
                    "question_title": title,
                    "tags": doc_tags,
                    "question_id": doc.metadata.get("question_id"),
                    "content": doc.page_content,
                    "relevance_score": relevance_score,
                    "rank_score": rank_score,
                    "lexical_exact": lexical_exact,
                    "lexical_overlap": lexical_overlap,
                    "source": "semantic",
                }
            )

        # Добавляем и усиливаем точные лексические совпадения из raw-коллекции.
        if settings.RAG_HYBRID_LEXICAL_BOOST and use_lexical_signal:
            for exact_match in self._exact_question_matches(query):
                exact_match["rank_score"] = 1.0
                ranked.append(exact_match)

        dedup: dict[str, dict[str, Any]] = {}
        for item in ranked:
            key = str(item.get("question_id") or item.get("question_title") or "")
            if not key:
                continue
            prev = dedup.get(key)
            if prev is None or float(item.get("rank_score", 0.0)) > float(prev.get("rank_score", 0.0)):
                dedup[key] = item

        ordered = sorted(
            dedup.values(),
            key=lambda x: float(x.get("rank_score", 0.0)),
            reverse=True,
        )
        return ordered[:top_k]

    def _answers_by_question_ids(self, question_boosts: dict[Any, float]) -> list[dict[str, Any]]:
        """
        Добор answer-доков напрямую по question_id (без векторного поиска).
        """
        collection = getattr(self.vectorstore, "_collection", None)
        if collection is None or not question_boosts:
            return []

        results: list[dict[str, Any]] = []
        for qid, boost in question_boosts.items():
            try:
                payload = collection.get(
                    where={"$and": [{"doc_type": "answer"}, {"question_id": qid}]},
                    include=["metadatas", "documents"],
                )
            except Exception:
                continue

            metadatas = payload.get("metadatas", []) or []
            documents = payload.get("documents", []) or []
            for metadata, content in zip(metadatas, documents):
                if not metadata:
                    continue
                doc_tags = _tags_to_list(metadata.get("tags", ""))
                answer_text = content or ""
                if "Ответ:" in answer_text:
                    answer_text = answer_text.split("Ответ:", 1)[1].strip()
                results.append(
                    {
                        "question_title": metadata.get("question_title", ""),
                        "question_id": metadata.get("question_id"),
                        "content": answer_text,
                        "tags": doc_tags,
                        "answer_id": metadata.get("answer_id"),
                        "is_best": bool(metadata.get("is_best", False)),
                        "favorites_count": int(metadata.get("favorites_count", 0) or 0),
                        "relevance_score": 1.0,
                        "rank_score": round(0.80 + 0.20 * max(0.0, min(1.0, float(boost))), 4),
                        "source_qid_match": True,
                    }
                )
        return results

    def retrieve_similar_questions(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        debug_log(
            enabled=settings.GENERATION_DEBUG_LOGS,
            stage='rag_query',
            message=f"retrieve_similar_questions query={query}",
            console_limit=settings.DEBUG_CONSOLE_PREVIEW_LIMIT,
            file_limit=None,
            file_path=settings.DEBUG_LOG_FILE if settings.DEBUG_LOG_TO_FILE else None,
        )
        results = self._retrieve_candidate_questions(query=query, tags=[], top_k=top_k)
        debug_log(
            enabled=settings.GENERATION_DEBUG_LOGS,
            stage='rag_result',
            message=(
                f"retrieve_similar_questions found={len(results)} "
                f"top={[r.get('question_title') for r in results[:3]]}"
            ),
            console_limit=settings.DEBUG_CONSOLE_PREVIEW_LIMIT,
            file_limit=None,
            file_path=settings.DEBUG_LOG_FILE if settings.DEBUG_LOG_TO_FILE else None,
        )
        return results

    def _retrieve_generation_questions(
        self,
        *,
        position: str,
        tech_stack: str,
        level: str,
        topics: list[str],
        question_type: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Retrieval, специализированный под генерацию вопросов.
        Использует semantic-query по контентным полям + rerank по типу/темам/технологиям.
        """
        top_k = _validated_top_k(top_k)
        topics = [t.strip() for t in topics if t.strip()]
        # semantic_query должен опираться только на поля, реально присутствующие в индексе вопроса:
        # title/tags/content. Поля position/level не хранятся в doc и дают шум.
        semantic_query = (
            f"Темы: {', '.join(topics)}\n"
            f"Технологии: {tech_stack}\n"
            f"Тип вопроса: {question_type}"
        )
        semantic_base = self._retrieve_candidate_questions(
            query=semantic_query,
            tags=topics,
            top_k=top_k * 4,
            use_lexical_signal=False,
        )
        metadata_base = self._metadata_candidates_for_generation(
            topics=topics,
            tech_stack=tech_stack,
            top_k=top_k * 4,
        )
        stack_tokens = _stack_tokens(tech_stack)

        merged_by_key: dict[str, dict[str, dict[str, Any]]] = {}
        for item in semantic_base:
            key = str(item.get("question_id") or item.get("question_title") or "")
            if not key:
                continue
            merged_by_key.setdefault(key, {})["semantic"] = item
        for item in metadata_base:
            key = str(item.get("question_id") or item.get("question_title") or "")
            if not key:
                continue
            merged_by_key.setdefault(key, {})["metadata"] = item

        base: list[dict[str, Any]] = []
        for _, parts in merged_by_key.items():
            semantic_item = parts.get("semantic")
            metadata_item = parts.get("metadata")
            if semantic_item and metadata_item:
                sem_rank = float(semantic_item.get("rank_score", 0.0))
                meta_rank = float(metadata_item.get("rank_score", 0.0))
                combined = dict(semantic_item)
                combined["rank_score"] = round(0.65 * sem_rank + 0.35 * meta_rank, 4)
                combined["source"] = "hybrid"
                base.append(combined)
            elif semantic_item:
                base.append(semantic_item)
            elif metadata_item:
                base.append(metadata_item)

        reranked: list[dict[str, Any]] = []
        for item in base:
            title = str(item.get("question_title", ""))
            tags = item.get("tags", []) or []
            base_rank = float(item.get("rank_score", 0.0))
            type_match = _question_type_match_score(question_type, title, tags)
            topic_match = _tag_overlap_ratio(topics, tags)
            stack_match = 0.0
            haystack = f"{_normalize(title)} {' '.join(_normalize(t) for t in tags)}"
            if stack_tokens:
                stack_match = sum(1 for tok in stack_tokens if tok in haystack) / len(stack_tokens)
            final_rank = (
                0.55 * base_rank
                + 0.25 * type_match
                + 0.15 * topic_match
                + 0.05 * stack_match
            )
            # Штрафуем явно не тот тип.
            if type_match == 0.0 and "код" in _normalize(question_type):
                final_rank *= 0.88
            candidate = dict(item)
            candidate["generation_rank_score"] = round(final_rank, 4)
            candidate["generation_type_match"] = type_match
            reranked.append(candidate)

        reranked.sort(key=lambda x: float(x.get("generation_rank_score", 0.0)), reverse=True)
        debug_log(
            enabled=settings.GENERATION_DEBUG_LOGS,
            stage='rag_result',
            message=(
                f"_retrieve_generation_questions semantic={len(semantic_base)} "
                f"metadata={len(metadata_base)} merged={len(base)} "
                f"position={position} level={level}"
            ),
            console_limit=settings.DEBUG_CONSOLE_PREVIEW_LIMIT,
            file_limit=None,
            file_path=settings.DEBUG_LOG_FILE if settings.DEBUG_LOG_TO_FILE else None,
        )
        return reranked[:top_k]

    def _metadata_candidates_for_generation(
        self,
        *,
        topics: list[str],
        tech_stack: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Metadata-first кандидаты: прямой поиск в коллекции по метаданным и документу.
        Это помогает, когда конкретный стек (например, FastAPI) нужно жестко учесть.
        """
        top_k = _validated_top_k(top_k)
        collection = getattr(self.vectorstore, "_collection", None)
        if collection is None:
            return []
        topics = [t.strip() for t in topics if t.strip()]
        stack_tokens = _stack_tokens(tech_stack)
        topic_tokens = [_normalize(t) for t in topics if _normalize(t)]
        query_tokens = sorted(set([*topic_tokens, *stack_tokens]))

        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []
        per_token_limit = top_k * 4
        for token in query_tokens:
            try:
                payload = collection.get(
                    where={
                        "$and": [
                            {"doc_type": "question"},
                            {"tags_lc": {"$contains": token}},
                        ]
                    },
                    include=["metadatas", "documents"],
                    limit=per_token_limit,
                )
            except Exception:
                continue
            metadatas.extend(payload.get("metadatas", []) or [])
            documents.extend(payload.get("documents", []) or [])

        # Strict mode: без full-scan fallback, только metadata prefilter.
        if not metadatas:
            return []

        ranked: list[dict[str, Any]] = []
        for metadata, content in zip(metadatas, documents):
            metadata = metadata or {}
            title = str(metadata.get("question_title", ""))
            doc_tags = _tags_to_list(metadata.get("tags", ""))
            haystack = f"{_normalize(title)} {_normalize(content or '')} {' '.join(_normalize(t) for t in doc_tags)}"
            topic_match = _tag_overlap_ratio(topics, doc_tags)
            stack_match = (
                sum(1 for tok in stack_tokens if tok in haystack) / len(stack_tokens)
                if stack_tokens
                else 0.0
            )

            if topic_match == 0.0 and stack_match == 0.0:
                continue

            base_rank = 0.20 + 0.80 * (0.65 * stack_match + 0.35 * topic_match)
            ranked.append(
                {
                    "question_title": title,
                    "tags": doc_tags,
                    "question_id": metadata.get("question_id"),
                    "content": content or "",
                    "relevance_score": None,
                    "rank_score": round(base_rank, 4),
                    "lexical_exact": False,
                    "lexical_overlap": 0.0,
                    "source": "metadata",
                }
            )
        dedup: dict[str, dict[str, Any]] = {}
        for item in ranked:
            key = str(item.get("question_id") or item.get("question_title") or "")
            if not key:
                continue
            prev = dedup.get(key)
            if prev is None or float(item.get("rank_score", 0.0)) > float(prev.get("rank_score", 0.0)):
                dedup[key] = item

        ordered = sorted(
            dedup.values(),
            key=lambda x: float(x.get("rank_score", 0.0)),
            reverse=True,
        )
        return ordered[:top_k]

    def retrieve_reference_answers(
        self,
        question_text: str,
        tags: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        tags = tags or []
        query = f"{question_text}\nТеги: {', '.join(tags)}"
        debug_log(
            enabled=settings.ASSESSMENT_DEBUG_LOGS,
            stage='rag_query',
            message=f"retrieve_reference_answers query={query}",
            console_limit=settings.DEBUG_CONSOLE_PREVIEW_LIMIT,
            file_limit=None,
            file_path=settings.DEBUG_LOG_FILE if settings.DEBUG_LOG_TO_FILE else None,
        )
        candidate_questions = self._retrieve_candidate_questions(
            query=question_text,
            tags=tags,
            top_k=max(top_k * 2, top_k),
        )
        candidate_qids = {
            q.get("question_id")
            for q in candidate_questions
            if q.get("question_id") is not None
        }
        candidate_qid_boosts: dict[Any, float] = {}
        exact_qid_boosts: dict[Any, float] = {}
        for q in candidate_questions:
            qid = q.get("question_id")
            if qid is None:
                continue
            q_rank = float(q.get("rank_score", 0.0))
            candidate_qid_boosts[qid] = max(candidate_qid_boosts.get(qid, 0.0), q_rank)
            if bool(q.get("lexical_exact")):
                exact_qid_boosts[qid] = max(exact_qid_boosts.get(qid, 0.0), q_rank)

        # Strict gate: если найден точный вопрос в базе, берём ответы только его qid.
        if settings.RAG_STRICT_EXACT_GATE and exact_qid_boosts:
            exact_answers = self._answers_by_question_ids(exact_qid_boosts)
            dedup_exact: dict[str, dict[str, Any]] = {}
            for item in exact_answers:
                key = str(item.get("answer_id") or f"{item.get('question_id')}::{item.get('content', '')[:80]}")
                prev = dedup_exact.get(key)
                if prev is None or float(item.get("rank_score", 0.0)) > float(prev.get("rank_score", 0.0)):
                    dedup_exact[key] = item
            ordered_exact = sorted(
                dedup_exact.values(),
                key=lambda x: (
                    float(x.get("rank_score", 0.0)),
                    int(x.get("favorites_count", 0) or 0),
                    1 if x.get("is_best") else 0,
                ),
                reverse=True,
            )
            results = ordered_exact[:top_k]
            debug_log(
                enabled=settings.ASSESSMENT_DEBUG_LOGS,
                stage='rag_result',
                message=(
                    f"retrieve_reference_answers strict_exact_match={True} "
                    f"exact_qids={list(exact_qid_boosts.keys())} "
                    f"found={len(results)}"
                ),
                console_limit=settings.DEBUG_CONSOLE_PREVIEW_LIMIT,
                file_limit=None,
                file_path=settings.DEBUG_LOG_FILE if settings.DEBUG_LOG_TO_FILE else None,
            )
            if results:
                return results

        expanded_query_parts = [question_text]
        expanded_query_parts.extend(
            q.get("question_title", "")
            for q in candidate_questions[: min(4, len(candidate_questions))]
            if q.get("question_title")
        )
        if tags:
            expanded_query_parts.append(f"Теги: {', '.join(tags)}")
        expanded_query = "\n".join(expanded_query_parts)

        docs_with_scores: list[tuple[Any, float | None]] = []
        try:
            docs_scored = self.vectorstore.similarity_search_with_relevance_scores(
                query=expanded_query,
                k=max(top_k * 4, top_k),
                filter={"doc_type": "answer"},
            )
            docs_with_scores = [(doc, _safe_score(score)) for doc, score in docs_scored]
        except Exception:
            docs = self.vectorstore.similarity_search(
                query=expanded_query,
                k=max(top_k * 4, top_k),
                filter={"doc_type": "answer"},
            )
            docs_with_scores = [(doc, None) for doc in docs]

        question_norm = _normalize(question_text)
        ranked_results: list[dict[str, Any]] = []
        for doc, relevance_score in docs_with_scores:
            question_title = doc.metadata.get("question_title", "")
            question_id = doc.metadata.get("question_id")
            doc_tags = _tags_to_list(doc.metadata.get("tags", ""))
            title_norm = _normalize(question_title)
            title_exact = title_norm == question_norm
            title_soft = question_norm in title_norm or title_norm in question_norm
            lexical_overlap = _lexical_overlap_ratio(question_text, question_title) if settings.RAG_HYBRID_LEXICAL_BOOST else 0.0
            qid_match = 1.0 if question_id in candidate_qids else 0.0
            tag_overlap = _tag_overlap_ratio(tags, doc_tags)
            base = relevance_score if relevance_score is not None else 0.45
            lexical_signal = (1.0 if title_exact else 0.6 if title_soft else 0.0) if settings.RAG_HYBRID_LEXICAL_BOOST else 0.0
            if settings.RAG_HYBRID_LEXICAL_BOOST:
                rank_score = (
                    0.50 * base
                    + 0.25 * qid_match
                    + 0.10 * tag_overlap
                    + 0.10 * lexical_signal
                    + 0.05 * lexical_overlap
                )
            else:
                rank_score = 0.65 * base + 0.25 * qid_match + 0.10 * tag_overlap

            # Минимальный порог после re-ranking убирает явный шум.
            if rank_score < 0.35:
                continue

            content = doc.page_content
            # Срезаем префиксы контекста, чтобы в ответ шёл текст эталона.
            if "Ответ:" in content:
                content = content.split("Ответ:", 1)[1].strip()

            ranked_results.append(
                {
                    "question_title": question_title,
                    "question_id": question_id,
                    "content": content,
                    "tags": doc_tags,
                    "answer_id": doc.metadata.get("answer_id"),
                    "is_best": bool(doc.metadata.get("is_best", False)),
                    "favorites_count": int(doc.metadata.get("favorites_count", 0) or 0),
                    "relevance_score": relevance_score,
                    "rank_score": round(rank_score, 4),
                    "source_qid_match": bool(qid_match),
                    "lexical_overlap": round(lexical_overlap, 4),
                }
            )

        # Добавляем прямые попадания по question_id.
        ranked_results.extend(self._answers_by_question_ids(candidate_qid_boosts))

        # Dedup по answer_id (или fallback ключу), оставляем лучший rank.
        dedup: dict[str, dict[str, Any]] = {}
        for item in ranked_results:
            key = str(item.get("answer_id") or f"{item.get('question_id')}::{item.get('content', '')[:80]}")
            prev = dedup.get(key)
            if prev is None or float(item.get("rank_score", 0.0)) > float(prev.get("rank_score", 0.0)):
                dedup[key] = item

        ordered = sorted(
            dedup.values(),
            key=lambda x: float(x.get("rank_score", 0.0)),
            reverse=True,
        )
        results = ordered[:top_k]
        debug_log(
            enabled=settings.ASSESSMENT_DEBUG_LOGS,
            stage='rag_result',
            message=(
                f"retrieve_reference_answers found={len(results)} "
                f"scores={[r.get('relevance_score') for r in results]} "
                f"rank={[r.get('rank_score') for r in results]} "
                f"qids={[r.get('question_id') for r in results]}"
            ),
            console_limit=settings.DEBUG_CONSOLE_PREVIEW_LIMIT,
            file_limit=None,
            file_path=settings.DEBUG_LOG_FILE if settings.DEBUG_LOG_TO_FILE else None,
        )
        return results

    def is_exact_question_exists(self, question_text: str, top_k: int = 3) -> bool:
        normalized = _normalize(question_text)
        docs = self.vectorstore.similarity_search(
            query=question_text,
            k=top_k,
            filter={"doc_type": "question"},
        )
        for doc in docs:
            candidate = _normalize(doc.metadata.get("question_title", ""))
            if candidate and candidate == normalized:
                return True
        return False

    def build_context_for_generation(
        self,
        position: str,
        tech_stack: str,
        level: str,
        topics: list[str],
        question_type: str,
        top_k: int = settings.RAG_TOP_K_GENERATION,
    ) -> str:
        examples = self._retrieve_generation_questions(
            position=position,
            tech_stack=tech_stack,
            level=level,
            topics=topics,
            question_type=question_type,
            top_k=top_k,
        )
        debug_log(
            enabled=settings.GENERATION_DEBUG_LOGS,
            stage='rag_result',
            message=(
                "build_context_for_generation reranked="
                f"{[(e.get('question_title'), e.get('generation_rank_score'), e.get('source')) for e in examples[:5]]}"
            ),
            console_limit=settings.DEBUG_CONSOLE_PREVIEW_LIMIT,
            file_limit=None,
            file_path=settings.DEBUG_LOG_FILE if settings.DEBUG_LOG_TO_FILE else None,
        )
        if not examples:
            return "Примеры из базы не найдены."

        lines = ["Примеры релевантных вопросов из локальной базы:"]
        for i, ex in enumerate(examples, 1):
            tags = ", ".join(ex.get("tags", []))
            lines.append(f"{i}. {ex['question_title']} (теги: {tags})")
        lines.append("Сгенерируй новый вопрос: не копируй примеры дословно.")
        return "\n".join(lines)

    def build_context_for_assessment(
        self,
        question_text: str,
        tags: list[str] | None = None,
        top_k: int = settings.RAG_TOP_K_EVAL,
    ) -> tuple[str, list[dict[str, Any]]]:
        refs = self.retrieve_reference_answers(question_text, tags=tags, top_k=top_k)
        if not refs:
            return "Эталонные ответы не найдены.", []

        lines = ["Эталонные ответы из локальной базы:"]
        for i, ref in enumerate(refs, 1):
            snippet = ref["content"][: settings.RAG_REFERENCE_SNIPPET_CHARS]
            lines.append(
                f"{i}. Вопрос-источник: {ref['question_title']}\n"
                f"Ответ-эталон: {snippet}"
            )
        return "\n\n".join(lines), refs

