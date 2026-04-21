"""
Главный модуль системы AI-собеседований.

Включает:
- Генерацию вопросов на основе вакансии с hh.ru
- Оценку ответов кандидатов
- Интеграцию с парсером hh.ru
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

import settings
from debug_utils import debug_log
from llm_provider import get_chat_llm
from parse_hh import VacancyInfo, parse_vacancy
from rag.retriever import LocalRAGRetriever


# ==============================
# Инициализация LLM
# ==============================

def get_llm():
    """Получить инстанс LLM."""
    return get_chat_llm(temperature=0.3)


def _debug_log(message: str):
    """Печатать debug-логи генерации, если включено в settings."""
    if settings.GENERATION_DEBUG_LOGS:
        debug_log(
            enabled=True,
            stage='info',
            message=message,
            console_limit=settings.DEBUG_CONSOLE_PREVIEW_LIMIT,
            file_limit=settings.DEBUG_CONSOLE_PREVIEW_LIMIT,
            file_path=settings.DEBUG_LOG_FILE if settings.DEBUG_LOG_TO_FILE else None,
        )


def _preview(text: str, limit: int = 240) -> str:
    value = (text or '').strip().replace('\n', ' ')
    return value if len(value) <= limit else value[:limit] + '...'


def _normalize_verdict(text: str) -> str:
    return (
        (text or '')
        .upper()
        .replace('*', '')
        .replace('`', '')
        .replace(':', ' ')
        .strip()
    )


def _is_approved_verdict(text: str) -> bool:
    normalized = _normalize_verdict(text)
    return ('ОДОБРЕНО' in normalized) and ('ОТКЛОН' not in normalized)


def _extract_json_block(content: str) -> dict | None:
    text = (content or '').strip()
    if text.startswith('```'):
        parts = text.split('```')
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith('json'):
                text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _sanitize_question_text(raw: str) -> str:
    """
    Вычленить текст самого вопроса из "раздутого" ответа LLM.
    """
    text = (raw or '').strip()
    if not text:
        return text

    lowered = text.lower()
    for marker in ['**вопрос:**', 'вопрос:', '### вопрос', 'задание:']:
        idx = lowered.find(marker.lower())
        if idx != -1:
            text = text[idx + len(marker):].strip()
            break

    # Обрезаем после блоков, которые не должны попадать в итоговый вопрос.
    stop_markers = ['ожидаемый ответ', 'критерии оценки', 'пример кода', 'dockerfile', 'requirements.txt']
    lowered = text.lower()
    cut = len(text)
    for marker in stop_markers:
        idx = lowered.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    text = text[:cut].strip()

    # Нормализуем whitespace.
    text = ' '.join(text.split())
    return text


def _reason_code_from_text(reason: str) -> str:
    r = (reason or '').lower()
    if 'технолог' in r:
        return 'MISSING_TECH'
    if 'несколько' in r or 'нескольк' in r:
        return 'MULTI_PART'
    if 'сложн' in r or 'уровн' in r:
        return 'LEVEL_MISMATCH'
    if 'дубликат' in r:
        return 'DUPLICATE'
    if 'формат' in r or 'json' in r:
        return 'FORMAT_ERROR'
    if 'кратк' in r or 'обшир' in r:
        return 'TOO_LONG'
    return 'OTHER'


def _is_unknown_tech_stack(value: str) -> bool:
    text = (value or '').strip().lower()
    return text in {'', 'не указано', 'unknown', 'n/a', '-'}


def _question_has_explicit_tech_signal(question_text: str) -> bool:
    q = (question_text or '').lower()
    tech_markers = (
        'python',
        'fastapi',
        'django',
        'flask',
        'sql',
        'postgres',
        'mysql',
        'redis',
        'mongodb',
        'docker',
        'kubernetes',
        'asyncio',
        'sqlalchemy',
        'asyncpg',
        'grpc',
        'jwt',
        'oauth',
        'api',
        'rest',
        'graphql',
    )
    return any(marker in q for marker in tech_markers)


class GenerateSelfCheck(BaseModel):
    single_question: bool = True
    mentions_tech: bool = True
    max_2_sentences: bool = True


class GenerateStructuredOutput(BaseModel):
    question_text: str = Field(min_length=5)
    question_type_hint: Literal['теория', 'код', 'отладка', 'системный дизайн']
    self_check: GenerateSelfCheck


class ValidateStructuredOutput(BaseModel):
    approved: bool
    reason_code: Literal[
        'MISSING_TECH',
        'MULTI_PART',
        'LEVEL_MISMATCH',
        'TOO_LONG',
        'DUPLICATE',
        'OTHER',
    ]
    reason_text: str
    normalized_question: str


def _as_dict(value: object) -> dict:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _pretty_json(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _stage_log(
    stage: str,
    message: str,
    question_index: int | None = None,
    *,
    file_unlimited: bool = False,
):
    debug_log(
        enabled=settings.GENERATION_DEBUG_LOGS,
        stage=stage,
        message=message,
        question_index=question_index,
        console_limit=settings.DEBUG_CONSOLE_PREVIEW_LIMIT,
        file_limit=None if file_unlimited else settings.DEBUG_CONSOLE_PREVIEW_LIMIT,
        file_path=settings.DEBUG_LOG_FILE if settings.DEBUG_LOG_TO_FILE else None,
    )


# ==============================
# Модели данных
# ==============================

@dataclass
class InterviewParams:
    """Параметры интервью."""
    position: str
    tech_stack: str
    level: str
    topics: list[str]
    time_limit: int
    num_questions: int
    company: str = ''
    description: str = ''
    
    @classmethod
    def from_vacancy(cls, vacancy: VacancyInfo, num_questions: int = 5, 
                     time_limit: int = 60) -> 'InterviewParams':
        """Создать параметры из вакансии hh.ru."""
        # Определяем уровень по опыту
        experience = vacancy.experience.lower()
        if 'без опыта' in experience or '1–3' in experience or '1-3' in experience:
            level = 'Junior'
        elif '3–6' in experience or '3-6' in experience:
            level = 'Middle'
        else:
            level = 'Senior'
        
        # Извлекаем технологии из skills
        tech_stack = vacancy.skills if vacancy.skills else 'Не указано'
        
        # Определяем темы на основе описания и навыков
        topics = cls._extract_topics(vacancy.description, vacancy.skills)
        
        return cls(
            position=vacancy.title,
            tech_stack=tech_stack,
            level=level,
            topics=topics,
            time_limit=time_limit,
            num_questions=num_questions,
            company=vacancy.company,
            description=vacancy.description[:2000],  # Ограничиваем длину
        )
    
    @staticmethod
    def _extract_topics(description: str, skills: str) -> list[str]:
        """Извлечь темы из описания вакансии."""
        # Базовые темы
        topics = []
        
        combined = (description + ' ' + skills).lower()
        
        # Проверяем наличие ключевых слов
        topic_keywords = {
            'API': ['api', 'rest', 'graphql', 'swagger', 'openapi'],
            'Базы данных': ['postgresql', 'mysql', 'mongodb', 'redis', 'sql', 'база данных', 'database'],
            'Асинхронность': ['async', 'asyncio', 'асинхрон', 'celery', 'rabbitmq', 'kafka'],
            'Docker/Kubernetes': ['docker', 'kubernetes', 'k8s', 'контейнер'],
            'Тестирование': ['test', 'pytest', 'unittest', 'тест', 'qa'],
            'CI/CD': ['ci/cd', 'jenkins', 'gitlab', 'github actions', 'devops'],
            'Микросервисы': ['микросервис', 'microservice', 'grpc'],
            'Безопасность': ['security', 'безопасност', 'auth', 'jwt', 'oauth'],
            'Архитектура': ['архитектур', 'design pattern', 'паттерн', 'solid'],
            'Веб-фреймворки': ['fastapi', 'django', 'flask', 'aiohttp'],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in combined for kw in keywords):
                topics.append(topic)
        
        # Если тем мало, добавляем базовые
        if len(topics) < 3:
            topics.extend(['Алгоритмы', 'ООП', 'Основы языка'])
        
        return topics[:6]  # Максимум 6 тем


# Дефолтные параметры (для обратной совместимости)
INTERVIEW_PARAMS = {
    'position': 'Backend-разработчик',
    'tech_stack': 'Python, FastAPI, PostgreSQL, Redis',
    'level': 'Middle',
    'topics': ['API', 'Кэширование', 'Базы данных', 'Асинхронность'],
    'time_limit': 60,
    'num_questions': 5,
}


# ==============================
# Состояние графа
# ==============================

class QuestionState(TypedDict):
    index: int
    plan: str
    question: str
    validated: str  # 'ОДОБРЕНО' или 'ОТКЛОНЁН: ...'
    formatted: dict | None
    attempts: int
    rag_context: str
    retry_hint: str
    rejection_reason: str
    retry_history: str
    reason_code: str


class OverallState(TypedDict):
    questions_to_generate: int
    completed_questions: list[dict]
    current_index: int


# ==============================
# Промпты генерации вопросов
# ==============================

# Определяем тип вопроса на основе его порядкового номера
QUESTION_PROGRESSION = {
    # (номер вопроса, общее количество) -> (тип, сложность)
    'early': {
        'type': 'теория',
        'complexity': 'простая',
        'description': 'Базовый теоретический вопрос на понимание концепций. Короткий ответ.',
    },
    'middle': {
        'type': 'теория или отладка',
        'complexity': 'средняя',
        'description': 'Вопрос на понимание с примерами или анализ кода. Требует объяснения.',
    },
    'late': {
        'type': 'код или системный дизайн',
        'complexity': 'сложная',
        'description': 'Практическая задача на написание кода или проектирование решения.',
    },
}


def get_question_type_for_index(index: int, total: int) -> dict:
    """Определить тип вопроса по его номеру."""
    progress = index / max(total - 1, 1)  # 0.0 - 1.0
    
    if progress < 0.4:
        return QUESTION_PROGRESSION['early']
    elif progress < 0.7:
        return QUESTION_PROGRESSION['middle']
    else:
        return QUESTION_PROGRESSION['late']


# 1. Планирование вопроса
design_prompt = ChatPromptTemplate.from_messages([
    ('system', '''Вы — архитектор собеседований с глубоким пониманием технических интервью.
Создайте детальный план для одного вопроса собеседования.

ВАЖНО: Строго следуйте указанному типу и сложности вопроса!

План должен включать:
1. Тип вопроса (строго соответствует указанному)
2. Сложность (строго соответствует указанной)
3. Основные концепции для проверки
4. Ожидаемое время на ответ (2-5 мин для теории, 5-10 мин для кода)
5. Критерии оценки

Для ТЕОРИИ: задайте конкретный вопрос с коротким ожидаемым ответом (определение, различие, принцип).
Для КОДА: сформулируйте небольшую задачу (функция на 5-15 строк, не целая система).
Для ОТЛАДКИ: приведите короткий фрагмент кода с ошибкой.'''),
    ('human', '''Параметры собеседования:
Позиция: {position}
Технологии: {tech_stack}
Уровень кандидата: {level}
Темы: {topics}

Это вопрос №{index} из {total_questions}.

=== ОБЯЗАТЕЛЬНЫЕ ТРЕБОВАНИЯ К ЭТОМУ ВОПРОСУ ===
Тип вопроса: {question_type}
Сложность: {question_complexity}
Описание: {question_description}
================================================

RAG-контекст (примеры из локальной базы):
{rag_context}

Подсказка из предыдущей неудачной попытки (если есть):
{retry_hint}

Создайте план вопроса, СТРОГО следуя указанному типу и сложности.
НЕ создавайте вопросы типа "разработайте систему" или "спроектируйте архитектуру" для теоретических вопросов.'''),
])

# 2. Генерация вопроса
generate_prompt = ChatPromptTemplate.from_messages([
    ('system', '''Вы — опытный технический интервьюер.
Создайте чёткий, понятный технический вопрос на русском языке.

ВАЖНЫЕ ТРЕБОВАНИЯ:
1. Для ТЕОРЕТИЧЕСКИХ вопросов:
   - Задайте конкретный вопрос с ожидаемым коротким ответом
   - Примеры: "Что такое X?", "В чём разница между X и Y?", "Какие преимущества у X?"
   - НЕ просите "разработать", "спроектировать", "реализовать"

2. Для вопросов с КОДОМ:
   - Сформулируйте небольшую конкретную задачу
   - Пример: "Напишите функцию, которая...", "Реализуйте метод для..."
   - Задача должна решаться за 5-15 строк кода

3. Для ОТЛАДКИ:
   - Приведите конкретный код с ошибкой
   - Спросите: "Найдите ошибку" или "Почему код не работает?"

4. ИЗБЕГАЙТЕ:
   - Слишком общих формулировок ("расскажите о...")
   - Вопросов на проектирование целых систем (если это не последний вопрос)
   - Вопросов с несколькими частями

ВЕРНИТЕ ОТВЕТ СТРОГО В JSON:
{{
  "question_text": "<только текст вопроса, без ожидаемого ответа, без markdown>",
  "question_type_hint": "<теория|код|отладка|системный дизайн>",
  "self_check": {{
    "single_question": true,
    "mentions_tech": true,
    "max_2_sentences": true
  }}
}}'''),
    ('human', '''План вопроса:
{plan}

Контекст примеров из локальной базы:
{rag_examples}

История предыдущих неудачных попыток (кратко):
{retry_history}

Создайте ОДИН конкретный вопрос на русском языке, строго следуя плану и типу вопроса.'''),
])

# 3. Валидация вопроса
validate_prompt = ChatPromptTemplate.from_messages([
    ('system', '''Вы — эксперт по качеству технических собеседований.
Проверьте вопрос по следующим критериям:

1. Ясность — понятен ли вопрос? Конкретен ли он?
2. Соответствие уровню — подходит ли для уровня {level}?
3. Релевантность — соответствует ли стеку технологий?
4. Краткость — можно ли ответить за разумное время (2-10 минут)?
5. Однозначность — один ли вопрос задан? (не несколько вопросов сразу)

ОТКЛОНИТЕ вопрос, если:
- Он слишком общий ("расскажите всё о...")
- Требует проектирования целой системы (кроме вопросов по системному дизайну)
- Содержит несколько вопросов в одном
- Слишком простой или слишком сложный для уровня

Верните СТРОГО JSON:
{{
  "approved": true/false,
  "reason_code": "<MISSING_TECH|MULTI_PART|LEVEL_MISMATCH|TOO_LONG|DUPLICATE|OTHER>",
  "reason_text": "<короткая причина>",
  "normalized_question": "<очищенный текст вопроса для форматтера>"
}}
Если всё в порядке, approved=true и reason_text="ОДОБРЕНО".
ВАЖНО: если режим проверки технологий = SOFT (стек не задан), НЕ отклоняйте вопрос по причине MISSING_TECH.'''),
    ('human', '''Проверьте вопрос для уровня {level}:

{question}

Технологии: {tech_stack}
Режим проверки технологий: {tech_validation_mode}'''),
])

# 4. Форматирование вопроса
format_prompt = ChatPromptTemplate.from_messages([
    ('system', '''Вы — форматировщик данных. Преобразуйте вопрос в структурированный JSON.
Выведите ТОЛЬКО валидный JSON, без пояснений и markdown.'''),
    ('human', '''Преобразуйте вопрос в JSON со следующими полями:
- question: текст вопроса на русском
- type: один из ["код", "системный дизайн", "теория", "отладка"]
- level: "{level}"
- tags: список тем на русском (3-5 тегов)
- expected_time_min: целое число минут
- follow_ups: список из 1-3 follow-up вопросов на русском
- evaluation_criteria: список критериев оценки (3-5 пунктов)

Вопрос:
{validated_question}

Выведите ТОЛЬКО JSON.'''),
])


# ==============================
# Узлы графа генерации
# ==============================

async def design_node(
    state: QuestionState,
    llm,
    retriever: LocalRAGRetriever | None = None,
    params: dict = None,
    delay: float = 0,
) -> dict:
    """Узел планирования вопроса."""
    params = params or INTERVIEW_PARAMS
    
    # Определяем тип вопроса на основе его номера
    total = params.get('num_questions', 5)
    question_info = get_question_type_for_index(state['index'], total)
    
    rag_context = 'Примеры из базы не найдены.'
    if retriever:
        try:
            rag_context = retriever.build_context_for_generation(
                position=params.get('position', INTERVIEW_PARAMS['position']),
                tech_stack=params.get('tech_stack', INTERVIEW_PARAMS['tech_stack']),
                level=params.get('level', INTERVIEW_PARAMS['level']),
                topics=params.get('topics', INTERVIEW_PARAMS['topics']),
                question_type=question_info['type'],
            )
        except Exception as e:
            print(f'RAG-контекст генерации недоступен: {e}')

    qn = state['index'] + 1
    _stage_log(
        'design_input',
        (
            f"Design input: idx={qn}, type={question_info['type']}, "
            f"complexity={question_info['complexity']}, "
            f"topics={params.get('topics', INTERVIEW_PARAMS['topics'])}"
        ),
        question_index=qn,
    )
    _stage_log(
        'rag_result',
        f'RAG context preview (len={len(rag_context)}): {rag_context}',
        question_index=qn,
        file_unlimited=True,
    )
    _stage_log(
        'warning',
        f'Retry hint: {state.get("retry_hint", "")}',
        question_index=qn,
    )

    chain = design_prompt | llm
    response = await chain.ainvoke({
        'position': params.get('position', INTERVIEW_PARAMS['position']),
        'tech_stack': params.get('tech_stack', INTERVIEW_PARAMS['tech_stack']),
        'level': params.get('level', INTERVIEW_PARAMS['level']),
        'topics': ', '.join(params.get('topics', INTERVIEW_PARAMS['topics'])),
        'index': state['index'] + 1,
        'total_questions': total,
        'question_type': question_info['type'],
        'question_complexity': question_info['complexity'],
        'question_description': question_info['description'],
        'rag_context': rag_context,
        'retry_hint': state.get('retry_hint', '') or 'нет',
    })
    
    if delay > 0:
        await asyncio.sleep(delay)

    _stage_log(
        'design_output',
        f'Design output plan: {response.content}',
        question_index=qn,
        file_unlimited=True,
    )
    return {'plan': response.content, 'rag_context': rag_context}


async def generate_node(state: QuestionState, llm, delay: float = 0) -> dict:
    """Узел генерации вопроса."""
    qn = state['index'] + 1
    _stage_log(
        'generate_input',
        f'Generate input plan: {state["plan"]}',
        question_index=qn,
        file_unlimited=True,
    )
    structured_llm = llm.with_structured_output(GenerateStructuredOutput)
    chain = generate_prompt | structured_llm
    response = await chain.ainvoke({
        'plan': state['plan'],
        'rag_examples': state.get('rag_context', 'Примеры из базы не найдены.'),
        'retry_history': state.get('retry_history', '') or state.get('retry_hint', '') or 'нет',
    })
    
    if delay > 0:
        await asyncio.sleep(delay)

    _stage_log(
        'generate_output',
        f'Generate output (structured): {response}',
        question_index=qn,
        file_unlimited=True,
    )
    payload = _as_dict(response)
    _stage_log(
        'generate_structured',
        f'Generate payload JSON:\n{_pretty_json(payload)}',
        question_index=qn,
        file_unlimited=True,
    )
    question_text = _sanitize_question_text(str(payload.get('question_text', '')).strip())
    if question_text:
        _stage_log(
            'generate_structured',
            f'Generate selected question_text: {question_text}',
            question_index=qn,
            file_unlimited=True,
        )
        return {'question': question_text}

    # Аварийный fallback (редкий случай несоответствия structured-output контракта).
    fallback_raw = json.dumps(payload, ensure_ascii=False) if payload else str(response)
    _stage_log(
        'warning',
        f'Generate structured payload missing question_text, fallback raw: {fallback_raw}',
        question_index=qn,
        file_unlimited=True,
    )
    return {'question': _sanitize_question_text(fallback_raw)}


async def validate_node(
    state: QuestionState,
    llm,
    retriever: LocalRAGRetriever | None = None,
    params: dict = None,
    delay: float = 0,
) -> dict:
    """Узел валидации вопроса."""
    params = params or INTERVIEW_PARAMS

    sanitized_question = _sanitize_question_text(state['question'])

    if retriever:
        try:
            if retriever.is_exact_question_exists(sanitized_question):
                _stage_log(
                    'validate_verdict',
                    'Validate: rejected by exact-match duplicate check against RAG DB',
                    question_index=state['index'] + 1,
                )
                return {
                    'validated': 'ОТКЛОНЁН: вопрос почти дословно совпадает с источником из базы',
                    'rejection_reason': 'Почти дословный дубликат вопроса из базы RAG',
                    'reason_code': 'DUPLICATE',
                }
        except Exception as e:
            print(f'Проверка дубликатов RAG недоступна: {e}')

    qn = state['index'] + 1
    tech_stack_value = params.get('tech_stack', INTERVIEW_PARAMS['tech_stack'])
    tech_mode = 'SOFT' if _is_unknown_tech_stack(tech_stack_value) else 'STRICT'
    _stage_log(
        'validate_input',
        f'Validate input question: {sanitized_question} | tech_mode={tech_mode}',
        question_index=qn,
        file_unlimited=True,
    )
    structured_llm = llm.with_structured_output(ValidateStructuredOutput)
    chain = validate_prompt | structured_llm
    response = await chain.ainvoke({
        'question': sanitized_question,
        'level': params.get('level', INTERVIEW_PARAMS['level']),
        'tech_stack': tech_stack_value,
        'tech_validation_mode': tech_mode,
    })
    payload = _as_dict(response)
    validated = json.dumps(payload, ensure_ascii=False)
    
    if delay > 0:
        await asyncio.sleep(delay)

    _stage_log(
        'validate_verdict',
        f'Validate verdict: {validated}',
        question_index=qn,
        file_unlimited=True,
    )
    _stage_log(
        'validate_structured',
        f'Validate payload JSON:\n{_pretty_json(payload)}',
        question_index=qn,
        file_unlimited=True,
    )
    if payload:
        approved = bool(payload.get('approved'))
        reason_code = str(payload.get('reason_code', 'OTHER')).strip().upper() or 'OTHER'
        reason_text = str(payload.get('reason_text', '')).strip() or 'Нет причины'
        normalized_question = _sanitize_question_text(str(payload.get('normalized_question', '')).strip())

        # Guardrail: защищаемся от ложного MISSING_TECH, когда стек не задан
        # или когда технологии явно присутствуют в тексте вопроса.
        if (
            (not approved)
            and reason_code == 'MISSING_TECH'
            and (
                tech_mode == 'SOFT'
                or _question_has_explicit_tech_signal(normalized_question or sanitized_question)
            )
        ):
            _stage_log(
                'warning',
                (
                    'Override MISSING_TECH to approved due to '
                    f'tech_mode={tech_mode} and explicit tech markers in question'
                ),
                question_index=qn,
            )
            approved = True
            reason_code = ''
            reason_text = 'ОДОБРЕНО (MISSING_TECH override)'

        _stage_log(
            'validate_structured',
            (
                f'Validate parsed fields: approved={approved}, reason_code={reason_code}, '
                f'reason_text={reason_text}, normalized_question={normalized_question}'
            ),
            question_index=qn,
            file_unlimited=True,
        )
        if approved:
            return {
                'validated': 'ОДОБРЕНО',
                'rejection_reason': '',
                'question': normalized_question or sanitized_question,
                'reason_code': '',
            }
        return {
            'validated': f'ОТКЛОНЁН: {reason_text}',
            'rejection_reason': reason_text,
            'reason_code': reason_code,
            'question': normalized_question or sanitized_question,
        }

    # Аварийный fallback.
    _stage_log(
        'warning',
        'Validate structured payload is empty, fallback FORMAT_ERROR',
        question_index=qn,
    )
    return {
        'validated': 'ОТКЛОНЁН: не удалось получить structured verdict',
        'rejection_reason': 'Structured output parse failed in validate_node',
        'question': sanitized_question,
        'reason_code': 'FORMAT_ERROR',
    }


async def format_node(state: QuestionState, llm, params: dict = None, delay: float = 0) -> dict:
    """Узел форматирования вопроса."""
    params = params or INTERVIEW_PARAMS
    qn = state['index'] + 1
    if not _is_approved_verdict(state['validated']):
        _stage_log(
            'format_info',
            f'Format skipped, validate rejected: {state["validated"]}',
            question_index=qn,
            file_unlimited=True,
        )
        return {
            'formatted': None,
            'rejection_reason': state.get('rejection_reason', state.get('validated', 'Отклонено валидатором')),
        }
    
    chain = format_prompt | llm
    try:
        response = await chain.ainvoke({
            'validated_question': state['question'],
            'level': params.get('level', INTERVIEW_PARAMS['level']),
        })
        
        if delay > 0:
            await asyncio.sleep(delay)
        
        # Очищаем от возможного markdown
        content = response.content.strip()
        if content.startswith('```'):
            content = content.split('```')[1]
            if content.startswith('json'):
                content = content[4:]
        content = content.strip()
        
        formatted = json.loads(content)
        _stage_log(
            'format_info',
            "Format success: "
            f"type={formatted.get('type')} "
            f"level={formatted.get('level')} "
            f"tags={formatted.get('tags')}",
            question_index=qn,
        )
        return {'formatted': formatted, 'rejection_reason': ''}
    except Exception as e:
        print(f'Ошибка форматирования: {e}')
        _stage_log(
            'format_info',
            f'Format raw LLM output: {response.content if "response" in locals() else ""}',
            question_index=qn,
            file_unlimited=True,
        )
        return {
            'formatted': None,
            'rejection_reason': f'Ошибка форматирования JSON: {e}',
        }


def create_question_graph(
    params: dict = None,
    mode: str = None,
    request_delay: float = None,
    retriever: LocalRAGRetriever | None = None,
) -> StateGraph:
    """
    Создать граф генерации одного вопроса.
    
    Args:
        params: Параметры интервью
        mode: Режим работы ('parallel' или 'sequential'). По умолчанию из settings.API_MODE
        request_delay: Задержка между запросами. По умолчанию из settings.API_REQUEST_DELAY
    """
    params = params or INTERVIEW_PARAMS
    mode = mode or settings.API_MODE
    request_delay = request_delay if request_delay is not None else settings.API_REQUEST_DELAY
    
    # Задержка применяется только в sequential режиме
    delay = request_delay if mode == 'sequential' else 0
    
    llm = get_llm()
    graph = StateGraph(QuestionState)
    
    # Оборачиваем узлы для передачи параметров и задержки
    async def _design(state):
        return await design_node(state, llm, retriever, params, delay)
    
    async def _generate(state):
        return await generate_node(state, llm, delay)
    
    async def _validate(state):
        return await validate_node(state, llm, retriever, params, delay)
    
    async def _format(state):
        return await format_node(state, llm, params, delay)
    
    graph.add_node('design', _design)
    graph.add_node('generate', _generate)
    graph.add_node('validate', _validate)
    graph.add_node('format', _format)

    graph.set_entry_point('design')
    graph.add_edge('design', 'generate')
    graph.add_edge('generate', 'validate')
    graph.add_edge('validate', 'format')
    graph.add_edge('format', END)

    return graph.compile()


# ==============================
# Генератор интервью
# ==============================

class InterviewGenerator:
    """
    Генератор вопросов для собеседования.
    
    Режим работы настраивается через settings.API_MODE:
    - 'parallel': запросы к API выполняются без задержки (быстро, но может вызвать 429)
    - 'sequential': запросы выполняются с задержкой (медленнее, но надёжнее)
    """
    
    def __init__(self, params: InterviewParams | dict = None, 
                 mode: str = None, request_delay: float = None):
        """
        Args:
            params: Параметры интервью
            mode: Режим работы ('parallel' или 'sequential'). По умолчанию из settings.API_MODE
            request_delay: Задержка между запросами. По умолчанию из settings.API_REQUEST_DELAY
        """
        if isinstance(params, InterviewParams):
            self.params = {
                'position': params.position,
                'tech_stack': params.tech_stack,
                'level': params.level,
                'topics': params.topics,
                'time_limit': params.time_limit,
                'num_questions': params.num_questions,
                'company': params.company,
                'description': params.description,
            }
        else:
            self.params = params or INTERVIEW_PARAMS
        
        self.mode = mode or settings.API_MODE
        self.request_delay = request_delay if request_delay is not None else settings.API_REQUEST_DELAY
        self.retriever = None
        try:
            self.retriever = LocalRAGRetriever()
        except Exception as e:
            print(f'RAG-база недоступна, генерация без retrieval-контекста: {e}')
        
        self.question_graph = create_question_graph(
            self.params, 
            mode=self.mode, 
            request_delay=self.request_delay,
            retriever=self.retriever,
        )
    
    @classmethod
    def from_hh_url(cls, url: str, num_questions: int = 5, 
                    time_limit: int = 60, mode: str = None, 
                    request_delay: float = None) -> 'InterviewGenerator':
        """Создать генератор из URL вакансии hh.ru."""
        vacancy = parse_vacancy(url)
        params = InterviewParams.from_vacancy(vacancy, num_questions, time_limit)
        return cls(params, mode=mode, request_delay=request_delay)
    
    async def generate_questions(self, retries: int = 3) -> list[dict]:
        """Сгенерировать вопросы для собеседования."""
        completed = []
        num_questions = self.params.get('num_questions', 5)
        max_attempts = num_questions * 3
        attempt = 0
        last_rejection_reason = ''
        rejection_memory: list[str] = []
        
        # Задержка между попытками генерации (только в sequential режиме)
        between_questions_delay = self.request_delay if self.mode == 'sequential' else 0

        while len(completed) < num_questions and attempt < max_attempts:
            print(f'Генерация вопроса #{len(completed) + 1} (попытка {attempt + 1})...')
            retry_hint = ''
            if last_rejection_reason:
                retry_hint = (
                    "Предыдущая попытка была отклонена. "
                    f"Причина: {last_rejection_reason}. "
                    "Сгенерируй один короткий, конкретный вопрос и исправь именно эту проблему."
                )

            retry_history = 'нет'
            if rejection_memory:
                retry_history = '; '.join(rejection_memory[-3:])

            initial_state: QuestionState = {
                'index': len(completed),
                'plan': '',
                'question': '',
                'validated': '',
                'formatted': None,
                'attempts': 0,
                'rag_context': '',
                'retry_hint': retry_hint,
                'rejection_reason': '',
                'retry_history': retry_history,
                'reason_code': '',
            }

            result = await self.question_graph.ainvoke(initial_state)

            if result['formatted']:
                completed.append(result['formatted'])
                print('V | Вопрос принят.')
                last_rejection_reason = ''
                rejection_memory.clear()
            else:
                print('X | Вопрос отклонён или не распарсен. Повтор...')
                last_rejection_reason = result.get('rejection_reason', '') or result.get('validated', '')
                reason_code = (result.get('reason_code', '') or _reason_code_from_text(last_rejection_reason)).upper()
                if reason_code:
                    short_reason = last_rejection_reason[:180] if last_rejection_reason else ''
                    rejection_memory.append(f'{reason_code}: {short_reason}')
                    rejection_memory = rejection_memory[-5:]
                _stage_log(
                    'warning',
                    f'Attempt reject reason propagated: {last_rejection_reason}',
                    question_index=len(completed) + 1,
                    file_unlimited=True,
                )

            attempt += 1
            
            # Задержка между попытками генерации (только в sequential режиме)
            if between_questions_delay > 0 and attempt < max_attempts:
                await asyncio.sleep(between_questions_delay)

        return completed
    
    def get_params_summary(self) -> dict:
        """Получить сводку параметров интервью."""
        return {
            'position': self.params.get('position'),
            'level': self.params.get('level'),
            'tech_stack': self.params.get('tech_stack'),
            'topics': self.params.get('topics'),
            'num_questions': self.params.get('num_questions'),
            'time_limit': self.params.get('time_limit'),
        }


# ==============================
# Обратная совместимость
# ==============================

async def run_interview_generator() -> list[dict]:
    """Запустить генерацию вопросов (обратная совместимость)."""
    generator = InterviewGenerator(INTERVIEW_PARAMS)
    return await generator.generate_questions()


# ==============================
# Запуск
# ==============================

async def main():
    """Точка входа для тестирования."""
    print('Запуск генерации вопросов на LangGraph...')
    
    # Можно использовать URL с hh.ru:
    # generator = InterviewGenerator.from_hh_url('https://hh.ru/vacancy/123456')
    
    # Или с дефолтными параметрами:
    generator = InterviewGenerator(INTERVIEW_PARAMS)
    
    questions = await generator.generate_questions()

    output = {
        'params': generator.get_params_summary(),
        'questions': questions,
    }

    filename = 'interview_questions.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'\nСгенерировано {len(questions)} вопросов.')
    print(f'Результат: {filename}')


if __name__ == '__main__':
    asyncio.run(main())
