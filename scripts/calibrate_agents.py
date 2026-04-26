"""
Калибровка оценочных агентов по шкале 1..5.

Сценарий:
1) Формирует 10 вопросов (каждый с follow-up).
2) Для каждого вопроса формирует 5 ответов (целевые уровни 1..5).
3) Прогоняет все кейсы через AssessmentCoordinator.
4) Считает метрики соответствия шкале и сохраняет отчет.

Запуск:
    uv run python scripts/calibrate_agents.py
"""

from __future__ import annotations

import asyncio
import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from types import MethodType
from typing import Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import settings as _settings
from src import debug_utils as _debug_utils
from src import llm_provider as _llm_provider
from src import parse_hh as _parse_hh
from src.rag import retriever as _rag_retriever

# Совместимость с импортами вида "import settings" внутри src-модулей.
sys.modules.setdefault("settings", _settings)
sys.modules.setdefault("debug_utils", _debug_utils)
sys.modules.setdefault("llm_provider", _llm_provider)
sys.modules.setdefault("parse_hh", _parse_hh)
sys.modules.setdefault("rag.retriever", _rag_retriever)

from src.agents import AssessmentCoordinator, assessment_result_to_dict


@dataclass
class CalibrationQuestion:
    question: str
    question_type: str
    tech_stack: str
    candidate_level: str
    tags: list[str]
    follow_ups: list[str]
    answers_by_score: dict[int, str]


def _clamp_1_5(value: float) -> int:
    rounded = int(round(value))
    if rounded < 1:
        return 1
    if rounded > 5:
        return 5
    return rounded


def _normalize_followups_local(follow_ups: list[str] | None) -> list[str]:
    if not follow_ups:
        return []
    normalized: list[str] = []
    for item in follow_ups:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _build_dataset() -> list[CalibrationQuestion]:
    return [
        CalibrationQuestion(
            question="Что такое FastAPI и зачем в нем используются Pydantic-модели?",
            question_type="теория",
            tech_stack="Python, FastAPI, Pydantic",
            candidate_level="Junior",
            tags=["Python", "FastAPI", "Pydantic"],
            follow_ups=[
                "Как включается автодокументация в FastAPI?",
                "Зачем нужна валидация входных данных на уровне схем?",
            ],
            answers_by_score={
                1: "Не знаю. Наверное, это база данных.",
                2: "FastAPI это фреймворк для API. Pydantic как-то нужен для данных.",
                3: "FastAPI — веб-фреймворк для Python, ориентирован на быстрые API. Pydantic-модели задают схему, валидируют вход и помогают сериализовать ответ.",
                4: "FastAPI — современный ASGI-фреймворк для API. Pydantic задает типы и валидацию, поэтому API получает корректные данные и понятные ошибки. Автодокументация доступна на /docs и /redoc. Схемы уменьшают число runtime-ошибок.",
                5: "FastAPI строится на ASGI/Starlette и типизации Python, а Pydantic обеспечивает контракт API: валидацию, парсинг, сериализацию и стабильные ошибки. Это улучшает качество интеграций, тестируемость и поддержку. /docs и /redoc генерируются автоматически через OpenAPI. При росте проекта схемы помогают контролировать обратную совместимость и версионирование.",
            },
        ),
        CalibrationQuestion(
            question="В чем разница между asyncio.gather и asyncio.create_task?",
            question_type="теория",
            tech_stack="Python, asyncio",
            candidate_level="Middle",
            tags=["Python", "Asyncio"],
            follow_ups=[
                "Когда уместно использовать return_exceptions=True в gather?",
                "Как правильно дождаться завершения фоновой задачи?",
            ],
            answers_by_score={
                1: "Оба одинаковые, просто разные названия.",
                2: "gather запускает задачи вместе, create_task тоже запускает.",
                3: "create_task планирует корутину как отдельную задачу и возвращает Task. gather агрегирует ожидание нескольких awaitable и возвращает результаты списком.",
                4: "create_task полезен для управления жизненным циклом фоновой задачи, а gather — когда нужно дождаться набора корутин и собрать результаты. return_exceptions=True нужен, когда не хотим падать на первом исключении. Фоновую задачу нужно явно await/cancel в shutdown.",
                5: "create_task создает и регистрирует Task в loop, что удобно для фона и тонкого контроля (cancel, status, callbacks). gather — механизм синхронной агрегации нескольких awaitable с предсказуемым порядком результатов. С return_exceptions=True исключения становятся элементами результата, что полезно для partial-failure сценариев. Для корректного завершения фоновых задач нужен явный ownership: хранить ссылки, отменять в shutdown и дожидаться cancellation.",
            },
        ),
        CalibrationQuestion(
            question="Напишите SQL-запрос, который вернет количество заказов по каждому пользователю, включая пользователей без заказов.",
            question_type="код",
            tech_stack="SQL, PostgreSQL",
            candidate_level="Junior",
            tags=["SQL", "PostgreSQL", "Базы данных"],
            follow_ups=[
                "Почему нужен LEFT JOIN, а не INNER JOIN?",
                "Что вернет COUNT(*) и COUNT(order_id) в этом случае?",
            ],
            answers_by_score={
                1: "SELECT * FROM orders;",
                2: "Можно сделать join users и orders и count.",
                3: "SELECT u.id, COUNT(o.id) AS orders_count FROM users u LEFT JOIN orders o ON o.user_id = u.id GROUP BY u.id;",
                4: "SELECT u.id, u.name, COUNT(o.id) AS orders_count FROM users u LEFT JOIN orders o ON o.user_id = u.id GROUP BY u.id, u.name; LEFT JOIN нужен, чтобы сохранить пользователей без заказов. COUNT(o.id) не считает NULL.",
                5: "SELECT u.id, u.name, COUNT(o.id) AS orders_count FROM users u LEFT JOIN orders o ON o.user_id = u.id GROUP BY u.id, u.name ORDER BY u.id; LEFT JOIN обязателен, потому что INNER JOIN отрежет пользователей без заказов. В этом паттерне COUNT(*) даст минимум 1 строку на пользователя после join, а COUNT(o.id) корректно посчитает только реальные заказы (без NULL).",
            },
        ),
        CalibrationQuestion(
            question="Объясните, что такое транзакция в PostgreSQL и зачем уровни изоляции.",
            question_type="теория",
            tech_stack="PostgreSQL, SQL",
            candidate_level="Middle",
            tags=["PostgreSQL", "SQL", "Транзакции"],
            follow_ups=[
                "Какие аномалии предотвращает SERIALIZABLE?",
                "Почему нельзя всегда выбирать максимальную изоляцию?",
            ],
            answers_by_score={
                1: "Транзакция это просто запрос.",
                2: "Транзакция нужна чтобы данные были корректны.",
                3: "Транзакция — группа операций, выполняемых как единое целое (ACID). Уровни изоляции определяют видимость изменений между конкурентными транзакциями.",
                4: "Транзакция дает атомарность и согласованность изменений. Изоляция управляет конкурентным доступом: от Read Committed до Serializable. Более высокий уровень снижает аномалии, например phantom reads, но может повышать блокировки и откаты.",
                5: "Транзакция в PostgreSQL обеспечивает ACID: либо все изменения фиксируются, либо откатываются. Уровни изоляции задают баланс консистентности и производительности. SERIALIZABLE предотвращает грязные/неповторяемые/фантомные чтения и сериализационные конфликты ценой роста конфликтов и ретраев; поэтому его выбирают точечно для критичных бизнес-инвариантов, а не глобально.",
            },
        ),
        CalibrationQuestion(
            question="Как спроектировать endpoint в FastAPI для создания пользователя с проверкой уникальности email?",
            question_type="системный дизайн",
            tech_stack="Python, FastAPI, PostgreSQL",
            candidate_level="Middle",
            tags=["FastAPI", "PostgreSQL", "API"],
            follow_ups=[
                "Какой HTTP-статус вернуть при дубликате email?",
                "Где лучше делать проверку: в приложении или в БД?",
            ],
            answers_by_score={
                1: "Просто вставлять в таблицу, как получится.",
                2: "Нужно сделать post endpoint и проверить email.",
                3: "Endpoint POST /users принимает email, имя и сохраняет пользователя. Перед вставкой проверяем, что email не занят, иначе возвращаем ошибку.",
                4: "POST /users с Pydantic-схемой, в БД ставим UNIQUE индекс на email. При конфликте возвращаем 409 Conflict с понятным сообщением. Проверка в приложении помогает UX, но истинная гарантия уникальности должна быть в БД.",
                5: "Дизайн: POST /users, валидация схемой, нормализация email (lowercase), транзакционная вставка с UNIQUE(email). На конфликт — 409 Conflict и структурированная ошибка. Проверка в приложении может дать быстрый фидбек, но race conditions закрывает только ограничение БД. Для надежности использовать upsert/обработку IntegrityError, логирование, идемпотентность для повторных запросов и тесты конкурентных сценариев.",
            },
        ),
        CalibrationQuestion(
            question="Найдите проблему в коде: shared_list = []; for i in range(1000): threading.Thread(target=lambda: shared_list.append(1)).start()",
            question_type="отладка",
            tech_stack="Python, threading",
            candidate_level="Middle",
            tags=["Python", "Потоки", "Отладка"],
            follow_ups=[
                "Почему итоговая длина списка может быть неожиданной?",
                "Как корректно дождаться завершения потоков?",
            ],
            answers_by_score={
                1: "Проблем нет, код правильный.",
                2: "Наверное, потоков много, но в целом нормально.",
                3: "Есть проблема синхронизации и отсутствует join, поэтому результат может быть непредсказуемым на момент проверки.",
                4: "Нужно сохранять ссылки на потоки и делать join, иначе main завершится раньше. Для потокобезопасности лучше использовать lock/queue. Ожидаемая длина может не успеть набраться из-за отсутствия ожидания.",
                5: "Код содержит несколько рисков: отсутствует управление жизненным циклом потоков (нет join), отсутствует контроль ошибок в потоках и нет ограничений по количеству потоков. Даже если append в CPython атомарен, без join состояние читается до завершения задач. Правильнее использовать ThreadPoolExecutor/Queue, хранить список потоков и явно join, а при необходимости защищать общие структуры lock-ами.",
            },
        ),
        CalibrationQuestion(
            question="Что такое индексы в PostgreSQL и когда они могут ухудшить производительность?",
            question_type="теория",
            tech_stack="PostgreSQL",
            candidate_level="Middle",
            tags=["PostgreSQL", "Индексы", "Базы данных"],
            follow_ups=[
                "Чем B-Tree отличается от GIN на высоком уровне?",
                "Почему слишком много индексов вредно для INSERT/UPDATE?",
            ],
            answers_by_score={
                1: "Индексы не нужны, они всегда мешают.",
                2: "Индекс ускоряет поиск.",
                3: "Индекс — структура для ускорения чтения. Но каждый индекс нужно обновлять при изменении данных, поэтому запись может замедляться.",
                4: "Индексы ускоряют WHERE/JOIN/ORDER BY, но занимают место и удорожают вставки/обновления. B-Tree универсален для сравнений и диапазонов, GIN полезен для составных значений и полнотекста.",
                5: "Индекс — отдельная структура доступа (например, B-Tree), ускоряющая чтение за счет дополнительного места и write-overhead. При каждом INSERT/UPDATE/DELETE движок поддерживает все релевантные индексы, поэтому избыток индексов ухудшает throughput записи и autovacuum-поведение. B-Tree хорош для =, <, >, ORDER BY; GIN эффективен для массивов/JSONB/FTS с множеством ключей. Индексы создают на основании реальных запросов и профилирования, а не «на всякий случай».",
            },
        ),
        CalibrationQuestion(
            question="Как в Python работает менеджер контекста и зачем нужен with?",
            question_type="теория",
            tech_stack="Python",
            candidate_level="Junior",
            tags=["Python", "ООП"],
            follow_ups=[
                "Какие методы вызываются у контекстного менеджера?",
                "Что происходит при исключении внутри блока with?",
            ],
            answers_by_score={
                1: "with это цикл.",
                2: "with нужен, чтобы открыть файл.",
                3: "Менеджер контекста управляет ресурсом: вход/выход из блока, например файл закрывается автоматически.",
                4: "with вызывает __enter__ и __exit__, поэтому ресурс корректно освобождается даже при ошибках. Это снижает утечки и делает код читаемее.",
                5: "with — синтаксис для детерминированного управления ресурсами через протокол __enter__/__exit__. Вход подготавливает ресурс, выход вызывается всегда, включая исключения. __exit__ получает тип/значение/traceback ошибки и может подавить исключение, вернув True. Это ключевой паттерн для файлов, соединений, транзакций и временных контекстов.",
            },
        ),
        CalibrationQuestion(
            question="Опишите подход к кешированию ответа API списка товаров.",
            question_type="системный дизайн",
            tech_stack="Python, FastAPI, Redis",
            candidate_level="Middle",
            tags=["FastAPI", "Redis", "Кэширование"],
            follow_ups=[
                "Как выбрать TTL?",
                "Что делать с инвалидацией при изменении товара?",
            ],
            answers_by_score={
                1: "Кэш не нужен.",
                2: "Можно положить ответ в Redis.",
                3: "Кэшируем популярные запросы в Redis с TTL и при промахе берем данные из БД.",
                4: "Используем cache-aside: ключ зависит от фильтров/пагинации, TTL выбираем по требованию к актуальности. При изменениях можно удалять связанные ключи или сокращать TTL.",
                5: "Практичный дизайн: cache-aside в Redis с ключом вида products:{filters_hash}:{page}. TTL подбираем по SLA свежести и частоте изменений (например 30–120 сек), для hot-ключей — probabilistic refresh. При изменении товара используем targeted invalidation (по тегам/префиксам) или versioned keys. Добавляем защиту от stampede (single-flight/lock), метрики hit ratio и fallback на БД.",
            },
        ),
        CalibrationQuestion(
            question="Чем отличается list от tuple в Python и когда что использовать?",
            question_type="теория",
            tech_stack="Python",
            candidate_level="Junior",
            tags=["Python", "Типы данных"],
            follow_ups=[
                "Почему tuple может быть ключом словаря, а list нет?",
                "Есть ли разница по памяти и скорости?",
            ],
            answers_by_score={
                1: "Ничем не отличаются.",
                2: "tuple нельзя менять, list можно.",
                3: "list — изменяемый тип, tuple — неизменяемый. tuple часто используют для фиксированных наборов данных.",
                4: "tuple hashable (при hashable элементах), поэтому подходит для ключей dict/set. list изменяем, поэтому как ключ не годится. tuple обычно компактнее по памяти.",
                5: "Ключевое различие — mutability: list изменяемый, tuple неизменяемый. Из-за неизменяемости tuple может быть hashable и использоваться как ключ (если элементы тоже hashable). tuple часто чуть экономнее по памяти и быстрее в итерации/создании для фиксированных наборов. list выбирают для динамических коллекций, tuple — для стабильных структур и семантики «запись неизменна».",
            },
        ),
    ]


def _build_confusion_matrix(rows: list[dict[str, Any]]) -> list[list[int]]:
    matrix = [[0 for _ in range(5)] for _ in range(5)]
    for row in rows:
        exp_idx = int(row["expected_score"]) - 1
        pred_idx = int(row["predicted_rounded"]) - 1
        matrix[exp_idx][pred_idx] += 1
    return matrix


def _per_agent_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    agent_values: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        target = int(row["expected_score"])
        for score in row["agent_scores"]:
            if score.get("error"):
                continue
            name = str(score.get("agent_name", "unknown"))
            agent_values.setdefault(name, []).append((target, float(score.get("score", 0.0))))

    result: dict[str, dict[str, Any]] = {}
    for agent_name, values in agent_values.items():
        mae = mean(abs(t - p) for t, p in values) if values else None
        exact = (
            sum(1 for t, p in values if int(round(p)) == t) / len(values)
            if values
            else None
        )
        result[agent_name] = {
            "samples": len(values),
            "mae": round(mae, 4) if mae is not None else None,
            "exact_match_rate": round(exact, 4) if exact is not None else None,
            "avg_predicted_score": round(mean(p for _, p in values), 4) if values else None,
        }
    return result


def _answer_preview(text: str, limit: int = 220) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def _agent_scores_brief(agent_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    brief: list[dict[str, Any]] = []
    for item in agent_scores:
        brief.append(
            {
                "agent_name": item.get("agent_name"),
                "score": item.get("score"),
                "error": item.get("error", False),
                "feedback": item.get("feedback", ""),
            }
        )
    return brief


def _format_agent_scores_line(agent_scores: list[dict[str, Any]]) -> str:
    if not agent_scores:
        return "agents: none"
    parts: list[str] = []
    for item in agent_scores:
        name = str(item.get("agent_name", "unknown"))
        if item.get("error"):
            parts.append(f"{name}=ERR")
            continue
        score = item.get("score")
        if score is None:
            parts.append(f"{name}=n/a")
            continue
        try:
            parts.append(f"{name}={float(score):.2f}")
        except (TypeError, ValueError):
            parts.append(f"{name}={score}")
    return "agents: " + ", ".join(parts)


def _parse_scores_filter(raw: str | None) -> list[int]:
    if not raw:
        return [1, 2, 3, 4, 5]
    values: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            score = int(token)
        except ValueError:
            continue
        if 1 <= score <= 5:
            values.append(score)
    unique_sorted = sorted(set(values))
    return unique_sorted or [1, 2, 3, 4, 5]


async def run_calibration(
    *,
    case_timeout_sec: float = 300.0,
    coordinator_mode: str = "sequential",
    max_concurrency: int = 1,
    save_case_logs: bool = True,
    scores_filter: list[int] | None = None,
    max_cases: int | None = None,
    quiet: bool = False,
    skip_followup_validation: bool = False,
    disable_assessment_debug: bool = False,
    use_question_cache: bool = True,
    fail_fast_on_error: bool = True,
) -> dict[str, Any]:
    if disable_assessment_debug:
        _settings.ASSESSMENT_DEBUG_LOGS = False

    dataset = _build_dataset()
    selected_scores = sorted(set(scores_filter or [1, 2, 3, 4, 5]))
    rows: list[dict[str, Any]] = []
    case_entries: list[tuple[int, int, CalibrationQuestion, int, str]] = []
    case_no = 0
    for q_idx, case in enumerate(dataset, start=1):
        for expected_score in selected_scores:
            case_no += 1
            case_entries.append(
                (
                    case_no,
                    q_idx,
                    case,
                    expected_score,
                    case.answers_by_score[expected_score],
                )
            )
    if max_cases is not None and max_cases > 0:
        case_entries = case_entries[:max_cases]
    total_cases = len(case_entries)
    if total_cases == 0:
        raise RuntimeError("No calibration cases selected. Check --scores/--max-cases.")

    out_dir = PROJECT_ROOT / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_log_path = out_dir / f"agent_calibration_cases_{ts}.jsonl"

    worker_count = max(1, int(max_concurrency))
    semaphore = asyncio.Semaphore(worker_count)
    case_logs: list[dict[str, Any]] = []
    coordinators = [
        AssessmentCoordinator(mode=coordinator_mode, request_delay=0.0)
        for _ in range(worker_count)
    ]
    coordinator_locks = [asyncio.Lock() for _ in range(worker_count)]
    print(
        "Runtime models: "
        f"chat={_settings.OLLAMA_CHAT_MODEL} "
        f"embed={_settings.OLLAMA_EMBED_MODEL} "
        f"provider={_settings.LLM_PROVIDER}",
        flush=True,
    )
    first_retriever = coordinators[0].retriever if coordinators else None
    if first_retriever is not None:
        collection_name = getattr(getattr(first_retriever, "vectorstore", None), "_collection", None)
        resolved_collection = (
            getattr(collection_name, "name", None) if collection_name is not None else _settings.RAG_COLLECTION_NAME
        )
        print(
            "Runtime RAG: "
            f"db_path={_settings.RAG_DB_PATH} "
            f"collection={resolved_collection}",
            flush=True,
        )

    def _install_speedups(coordinator: AssessmentCoordinator) -> None:
        if use_question_cache and coordinator.retriever:
            ref_cache: dict[tuple[str, tuple[str, ...], int], tuple[str, list[dict[str, Any]]]] = {}
            original_retrieve = coordinator.retriever.build_context_for_assessment

            def cached_retrieve(
                *,
                question_text: str,
                tags: list[str] | None = None,
                top_k: int = 5,
            ) -> tuple[str, list[dict[str, Any]]]:
                tags_key = tuple(
                    sorted(
                        {
                            str(t).strip().lower()
                            for t in (tags or [])
                            if str(t).strip()
                        }
                    )
                )
                key = (str(question_text or "").strip(), tags_key, int(top_k))
                cached = ref_cache.get(key)
                if cached is not None:
                    return cached
                value = original_retrieve(question_text=question_text, tags=tags or [], top_k=top_k)
                ref_cache[key] = value
                return value

            coordinator.retriever.build_context_for_assessment = cached_retrieve  # type: ignore[method-assign]

        if skip_followup_validation:
            async def passthrough_followups(
                self: AssessmentCoordinator,
                main_question: str,
                follow_ups: list[str] | None,
                candidate_level: str,
                tech_stack: str,
            ) -> list[str]:
                _ = (main_question, candidate_level, tech_stack)
                return _normalize_followups_local(follow_ups)

            coordinator._validate_followups = MethodType(passthrough_followups, coordinator)
            return

        if use_question_cache:
            followup_cache: dict[tuple[str, tuple[str, ...], str, str], tuple[str, ...]] = {}
            original_validate = coordinator._validate_followups

            async def cached_validate(
                self: AssessmentCoordinator,
                main_question: str,
                follow_ups: list[str] | None,
                candidate_level: str,
                tech_stack: str,
            ) -> list[str]:
                key = (
                    str(main_question or "").strip(),
                    tuple(_normalize_followups_local(follow_ups)),
                    str(candidate_level or "").strip().lower(),
                    str(tech_stack or "").strip().lower(),
                )
                cached = followup_cache.get(key)
                if cached is not None:
                    return list(cached)
                approved = await original_validate(
                    main_question=main_question,
                    follow_ups=follow_ups,
                    candidate_level=candidate_level,
                    tech_stack=tech_stack,
                )
                followup_cache[key] = tuple(approved)
                return approved

            coordinator._validate_followups = MethodType(cached_validate, coordinator)

    for coordinator in coordinators:
        _install_speedups(coordinator)

    start_ts = perf_counter()
    progress_lock = asyncio.Lock()
    progress = {"done": 0}

    async def run_single_case(
        case_id: int,
        q_idx: int,
        case: CalibrationQuestion,
        expected_score: int,
        answer: str,
    ) -> dict[str, Any]:
        started = perf_counter()
        async with semaphore:
            if not quiet:
                print(
                    f"[{case_id}/{total_cases}] Q{q_idx} expected={expected_score} | "
                    f"type={case.question_type} level={case.candidate_level}",
                    flush=True,
                )
                print(f"  question: {case.question}", flush=True)
                print(
                    f"  follow_ups({len(case.follow_ups)}): "
                    + ("; ".join(case.follow_ups) if case.follow_ups else "none"),
                    flush=True,
                )
                print(f"  answer_preview: {_answer_preview(answer)}", flush=True)

            coordinator_idx = (case_id - 1) % worker_count
            coordinator = coordinators[coordinator_idx]
            error_text = ""
            try:
                async with coordinator_locks[coordinator_idx]:
                    assessed = await asyncio.wait_for(
                        coordinator.assess_answer(
                            question=case.question,
                            answer=answer,
                            question_type=case.question_type,
                            tech_stack=case.tech_stack,
                            candidate_level=case.candidate_level,
                            question_tags=case.tags,
                            follow_ups=case.follow_ups,
                        ),
                        timeout=case_timeout_sec,
                    )
                assessed_dict = assessment_result_to_dict(assessed)
                predicted = float(assessed_dict["total_score"])
            except asyncio.TimeoutError:
                error_text = f"timeout>{case_timeout_sec}s"
                assessed_dict = {
                    "total_score": 1.0,
                    "agent_scores": [],
                    "final_feedback": "Calibration timeout",
                    "recommendation": "Требуется ручная проверка",
                    "retrieval_confidence": "unknown",
                    "retrieval_score": None,
                    "retrieval_references": 0,
                }
                predicted = 1.0
            except Exception as e:
                error_text = str(e)
                assessed_dict = {
                    "total_score": 1.0,
                    "agent_scores": [],
                    "final_feedback": f"Calibration error: {e}",
                    "recommendation": "Требуется ручная проверка",
                    "retrieval_confidence": "unknown",
                    "retrieval_score": None,
                    "retrieval_references": 0,
                }
                predicted = 1.0

            row = {
                "case_id": case_id,
                "question_index": q_idx,
                "question": case.question,
                "question_type": case.question_type,
                "candidate_level": case.candidate_level,
                "tech_stack": case.tech_stack,
                "tags": list(case.tags),
                "follow_ups": list(case.follow_ups),
                "expected_score": expected_score,
                "answer": answer,
                "answer_preview": _answer_preview(answer),
                "predicted_score": round(predicted, 3),
                "predicted_rounded": _clamp_1_5(predicted),
                "agent_scores": assessed_dict.get("agent_scores", []),
                "final_feedback": assessed_dict.get("final_feedback", ""),
                "recommendation": assessed_dict.get("recommendation", ""),
                "retrieval_confidence": assessed_dict.get("retrieval_confidence", "unknown"),
                "retrieval_score": assessed_dict.get("retrieval_score"),
                "retrieval_references": assessed_dict.get("retrieval_references", 0),
                "run_error": error_text,
                "latency_sec": round(perf_counter() - started, 3),
            }
            if error_text:
                print(
                    (
                        f"[error] case={case_id}/{total_cases} q={q_idx} "
                        f"expected={expected_score} type={case.question_type} "
                        f"error={error_text}"
                    ),
                    flush=True,
                )
            if not quiet:
                print(
                    f"  -> predicted={row['predicted_score']:.2f} "
                    f"rounded={row['predicted_rounded']} "
                    f"recommendation={row['recommendation']}"
                    + (f" error={error_text}" if error_text else ""),
                    flush=True,
                )
                print(
                    f"  -> {_format_agent_scores_line(row.get('agent_scores', []))}",
                    flush=True,
                )

            async with progress_lock:
                progress["done"] += 1
                done = progress["done"]
                elapsed_sec = perf_counter() - start_ts
                avg_sec = elapsed_sec / done if done > 0 else 0.0
                left = max(0, total_cases - done)
                eta_sec = avg_sec * left
                print(
                    (
                        f"[progress] {done}/{total_cases} "
                        f"({(done / total_cases) * 100:.1f}%) "
                        f"elapsed={_format_duration(elapsed_sec)} "
                        f"eta={_format_duration(eta_sec)} "
                        f"avg={avg_sec:.1f}s/case"
                    ),
                    flush=True,
                )
            return row

    tasks = [
        asyncio.create_task(run_single_case(*entry))
        for entry in case_entries
    ]
    rows = []
    for done_task in asyncio.as_completed(tasks):
        row = await done_task
        rows.append(row)
        if fail_fast_on_error and row.get("run_error"):
            for task in tasks:
                if not task.done():
                    task.cancel()
            break

    if fail_fast_on_error and any(r.get("run_error") for r in rows):
        for task in tasks:
            if not task.done():
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    rows.sort(key=lambda r: int(r["case_id"]))
    case_logs.extend(rows)

    if save_case_logs:
        with case_log_path.open("w", encoding="utf-8") as f:
            for item in case_logs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved case logs: {case_log_path}", flush=True)

    failed_cases = [
        row
        for row in rows
        if int(row["expected_score"]) != int(row["predicted_rounded"])
    ]
    failed_cases_log = [
        {
            "case_id": row["case_id"],
            "question_index": row["question_index"],
            "question": row["question"],
            "follow_ups": row.get("follow_ups", []),
            "expected_score": row["expected_score"],
            "predicted_score": row["predicted_score"],
            "predicted_rounded": row["predicted_rounded"],
            "answer": row["answer"],
            "run_error": row.get("run_error", ""),
            "recommendation": row.get("recommendation", ""),
            "final_feedback": row.get("final_feedback", ""),
            "agent_scores": _agent_scores_brief(row.get("agent_scores", [])),
        }
        for row in failed_cases
    ]

    failed_log_path = out_dir / f"agent_calibration_failed_{ts}.jsonl"
    with failed_log_path.open("w", encoding="utf-8") as f:
        for item in failed_cases_log:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(
        f"Saved failed-cases log: {failed_log_path} "
        f"(count={len(failed_cases_log)})",
        flush=True,
    )

    if not rows:
        raise RuntimeError("Calibration produced no rows. Check fail-fast / runtime errors.")

    mae = mean(abs(r["expected_score"] - r["predicted_score"]) for r in rows)
    exact_rate = sum(
        1 for r in rows if int(r["expected_score"]) == int(r["predicted_rounded"])
    ) / len(rows)
    within_1_rate = (
        sum(
            1
            for r in rows
            if abs(int(r["expected_score"]) - int(r["predicted_rounded"])) <= 1
        )
        / len(rows)
    )
    latencies = sorted(float(r.get("latency_sec", 0.0) or 0.0) for r in rows)
    p50_idx = len(latencies) // 2
    p95_idx = int(0.95 * (len(latencies) - 1)) if len(latencies) > 1 else 0
    p50_latency = latencies[p50_idx] if latencies else 0.0
    p95_latency = latencies[p95_idx] if latencies else 0.0

    per_expected: dict[int, dict[str, float | None]] = {}
    for score in selected_scores:
        subset = [r for r in rows if int(r["expected_score"]) == score]
        if not subset:
            per_expected[score] = {
                "count": 0.0,
                "avg_predicted": None,
                "avg_rounded": None,
                "exact_match_rate": None,
            }
            continue
        per_expected[score] = {
            "count": float(len(subset)),
            "avg_predicted": round(mean(r["predicted_score"] for r in subset), 4),
            "avg_rounded": round(mean(r["predicted_rounded"] for r in subset), 4),
            "exact_match_rate": round(
                sum(1 for r in subset if r["predicted_rounded"] == score) / len(subset),
                4,
            ),
        }

    confusion = _build_confusion_matrix(rows)
    agents_metrics = _per_agent_metrics(rows)
    retrieval_distribution = {
        "high": sum(1 for r in rows if r["retrieval_confidence"] == "high"),
        "medium": sum(1 for r in rows if r["retrieval_confidence"] == "medium"),
        "low": sum(1 for r in rows if r["retrieval_confidence"] == "low"),
        "unknown": sum(1 for r in rows if r["retrieval_confidence"] == "unknown"),
    }

    return {
        "created_at": datetime.now().isoformat(),
        "dataset_summary": {
            "questions": len(dataset),
            "answers_per_question": len(selected_scores),
            "selected_scores": selected_scores,
            "total_cases": len(rows),
        "max_cases": max_cases,
            "max_concurrency": worker_count,
            "coordinator_mode": coordinator_mode,
            "case_timeout_sec": case_timeout_sec,
            "quiet": quiet,
            "skip_followup_validation": skip_followup_validation,
            "disable_assessment_debug": disable_assessment_debug,
            "use_question_cache": use_question_cache,
        },
        "overall_metrics": {
            "mae_total_score": round(mae, 4),
            "exact_match_rate_rounded": round(exact_rate, 4),
            "within_1_rate_rounded": round(within_1_rate, 4),
        },
        "speed_metrics": {
            "avg_latency_sec": round(mean(latencies), 4) if latencies else None,
            "p50_latency_sec": round(p50_latency, 4),
            "p95_latency_sec": round(p95_latency, 4),
        },
        "per_expected_score_metrics": per_expected,
        "confusion_matrix_rounded": confusion,
        "per_agent_metrics": agents_metrics,
        "retrieval_confidence_distribution": retrieval_distribution,
        "failed_cases_count": len(failed_cases),
        "failed_cases_log_path": str(failed_log_path),
        "rows": rows,
        "dataset": [asdict(item) for item in dataset],
    }


def _print_short_report(report: dict[str, Any]) -> None:
    overall = report["overall_metrics"]
    speed = report.get("speed_metrics", {})
    print("\n=== Calibration summary ===")
    print(f"Total cases: {report['dataset_summary']['total_cases']}")
    print(f"MAE (total_score): {overall['mae_total_score']:.3f}")
    print(f"Exact match (rounded): {overall['exact_match_rate_rounded']:.1%}")
    print(f"Within-1 (rounded): {overall.get('within_1_rate_rounded', 0.0):.1%}")
    if speed:
        print(
            "Latency (sec): "
            f"avg={speed.get('avg_latency_sec')} "
            f"p50={speed.get('p50_latency_sec')} "
            f"p95={speed.get('p95_latency_sec')}"
        )
    print("Per expected score:")
    for score in report["dataset_summary"].get("selected_scores", [1, 2, 3, 4, 5]):
        row = report["per_expected_score_metrics"][score]
        if row["count"] == 0:
            print(f"  {score}: no samples")
        else:
            print(
                f"  {score}: avg_pred={row['avg_predicted']:.2f}, "
                f"exact={row['exact_match_rate']:.1%}"
            )
    print("Retrieval confidence distribution:", report["retrieval_confidence_distribution"])


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent calibration suite")
    parser.add_argument(
        "--case-timeout-sec",
        type=float,
        default=300.0,
        help="Timeout per calibration case in seconds",
    )
    parser.add_argument(
        "--mode",
        default="sequential",
        choices=["parallel", "sequential"],
        help="AssessmentCoordinator mode for calibration",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of calibration cases processed concurrently",
    )
    parser.add_argument(
        "--no-case-logs",
        action="store_true",
        help="Disable writing per-case jsonl log file",
    )
    parser.add_argument(
        "--scores",
        default="1,2,3,4,5",
        help="Comma-separated expected scores subset, e.g. '5' or '3,4,5'",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional hard limit for number of calibration cases after score filtering",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-case console output",
    )
    parser.add_argument(
        "--skip-followup-validation",
        action="store_true",
        help="Skip LLM follow-up validation in calibration (faster, less strict)",
    )
    parser.add_argument(
        "--disable-assessment-debug",
        action="store_true",
        help="Disable verbose assessment debug logs for faster calibration",
    )
    parser.add_argument(
        "--no-question-cache",
        action="store_true",
        help="Disable per-question caching of retrieval/follow-up validation",
    )
    parser.add_argument(
        "--no-fail-fast-on-error",
        action="store_true",
        help="Disable fail-fast behavior (by default fail-fast is enabled)",
    )
    args = parser.parse_args()
    selected_scores = _parse_scores_filter(args.scores)
    print(
        "Calibration config: "
        f"mode={args.mode}, max_concurrency={args.max_concurrency}, "
        f"case_timeout_sec={args.case_timeout_sec}, scores={selected_scores}, "
        f"max_cases={args.max_cases}, "
        f"quiet={args.quiet}, skip_followup_validation={args.skip_followup_validation}, "
        f"disable_assessment_debug={args.disable_assessment_debug}, "
        f"use_question_cache={not args.no_question_cache}, "
        f"fail_fast_on_error={not args.no_fail_fast_on_error}",
        flush=True,
    )
    print(
        "Tip: keep Streamlit/app idle during calibration to avoid model contention.",
        flush=True,
    )

    report = await run_calibration(
        case_timeout_sec=args.case_timeout_sec,
        coordinator_mode=args.mode,
        max_concurrency=args.max_concurrency,
        save_case_logs=not args.no_case_logs,
        scores_filter=selected_scores,
        max_cases=args.max_cases,
        quiet=args.quiet,
        skip_followup_validation=args.skip_followup_validation,
        disable_assessment_debug=args.disable_assessment_debug,
        use_question_cache=not args.no_question_cache,
        fail_fast_on_error=not args.no_fail_fast_on_error,
    )
    out_dir = PROJECT_ROOT / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"agent_calibration_{ts}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_short_report(report)
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
