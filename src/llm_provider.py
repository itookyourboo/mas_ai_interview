"""
Провайдеры LLM и embeddings.

Поддерживает:
- Ollama (локально)
- GigaChat (обратная совместимость)
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_gigachat import GigaChat
from langchain_ollama import ChatOllama, OllamaEmbeddings

try:
    import settings
except ModuleNotFoundError:  # pragma: no cover - fallback для запуска как src.*
    from src import settings


class PrefixedEmbeddings(Embeddings):
    """
    Обёртка над любой `Embeddings`-моделью, добавляющая task-префиксы.

    Зачем:
        Современные instruction-tuned эмбеддеры (Nomic v2, BGE-M3, GTE и т.п.)
        обучались с разными префиксами для документов и запросов. Без них
        retrieval-качество существенно падает (для Nomic v2 — порядка 5–10
        пунктов nDCG@10).

    Как:
        - `embed_documents(texts)` -> к каждому тексту препендится `document_prefix`
        - `embed_query(text)`      -> к тексту препендится `query_prefix`
        Если префикс пустой ("") — обёртка прозрачна.
    """

    def __init__(
        self,
        inner: Embeddings,
        document_prefix: str = "",
        query_prefix: str = "",
    ) -> None:
        self._inner = inner
        self._document_prefix = document_prefix or ""
        self._query_prefix = query_prefix or ""

    @property
    def inner(self) -> Embeddings:
        return self._inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{self._document_prefix}{t}" for t in texts]
        return self._inner.embed_documents(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(f"{self._query_prefix}{text}")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{self._document_prefix}{t}" for t in texts]
        return await self._inner.aembed_documents(prefixed)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._inner.aembed_query(f"{self._query_prefix}{text}")


def _provider_name() -> str:
    return (settings.LLM_PROVIDER or "ollama").strip().lower()


def get_chat_llm(temperature: float = 0.2) -> Any:
    """
    Вернуть chat LLM в соответствии с конфигурацией.
    """
    provider = _provider_name()

    if provider == "gigachat":
        return GigaChat(
            credentials=settings.MODEL_API_KEY,
            verify_ssl_certs=False,
            model=settings.MODEL_NAME,
            base_url=settings.MODEL_BASE_URL or None,
            temperature=temperature,
            scope="GIGACHAT_API_PERS",
        )

    # По умолчанию локальный Ollama
    return ChatOllama(
        model=settings.OLLAMA_CHAT_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
    )


def get_embeddings() -> Any:
    """
    Вернуть embedding-модель в соответствии с конфигурацией.

    Если в settings заданы непустые `RAG_EMBED_DOCUMENT_PREFIX` /
    `RAG_EMBED_QUERY_PREFIX`, базовый эмбеддер оборачивается в
    `PrefixedEmbeddings`, чтобы документы и запросы получали корректные
    task-префиксы (для Nomic v2 — `search_document: ` и `search_query: `).
    """
    provider = _provider_name()

    if provider == "gigachat":
        raise ValueError(
            "Embeddings for gigachat are not configured in this project. "
            "Use LLM_PROVIDER=ollama for local RAG."
        )

    base = OllamaEmbeddings(
        model=settings.OLLAMA_EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )

    doc_prefix = getattr(settings, "RAG_EMBED_DOCUMENT_PREFIX", "") or ""
    query_prefix = getattr(settings, "RAG_EMBED_QUERY_PREFIX", "") or ""

    if not doc_prefix and not query_prefix:
        return base

    return PrefixedEmbeddings(
        inner=base,
        document_prefix=doc_prefix,
        query_prefix=query_prefix,
    )
