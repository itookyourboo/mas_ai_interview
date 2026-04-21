"""
Провайдеры LLM и embeddings.

Поддерживает:
- Ollama (локально)
- GigaChat (обратная совместимость)
"""

from __future__ import annotations

from typing import Any

from langchain_gigachat import GigaChat
from langchain_ollama import ChatOllama, OllamaEmbeddings

try:
    import settings
except ModuleNotFoundError:  # pragma: no cover - fallback для запуска как src.*
    from src import settings


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
    """
    provider = _provider_name()

    if provider == "gigachat":
        raise ValueError(
            "Embeddings for gigachat are not configured in this project. "
            "Use LLM_PROVIDER=ollama for local RAG."
        )

    return OllamaEmbeddings(
        model=settings.OLLAMA_EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )
