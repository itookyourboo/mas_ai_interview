"""
Агенты для оценивания ответов кандидатов.

Агенты:
1. Assessment Coordinator Agent - координирует оценку и применяет веса
2. Correctness Agent - проверяет функциональную корректность
3. Code Quality Agent - анализирует качество кода
4. Conceptual Understanding Agent - оценивает глубину понимания
5. Relevance Agent - проверяет релевантность ответа
6. Level Alignment Agent - оценивает соответствие уровню кандидата
7. Explanation Agent - формирует итоговую оценку
"""

import json
from dataclasses import dataclass
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate

import settings
from debug_utils import debug_log
from llm_provider import get_chat_llm
from rag.retriever import LocalRAGRetriever


# ==============================
# Модели данных
# ==============================

@dataclass
class AgentScore:
    """Оценка от одного агента."""
    agent_name: str
    score: float  # 1-5 для успешных оценок
    feedback: str
    details: dict | None = None
    error: bool = False  # True если агент завершился с ошибкой
    weight: float = 0.0  # Вес агента для расчёта общей оценки


@dataclass
class AssessmentResult:
    """Итоговый результат оценки."""
    question: str
    answer: str
    total_score: float
    agent_scores: list[AgentScore]
    final_feedback: str
    recommendation: str
    retrieval_confidence: str = 'unknown'
    retrieval_score: float | None = None
    retrieval_references: int = 0


class AssessmentState(TypedDict):
    """Состояние для графа оценивания."""
    question: str
    question_type: str
    expected_answer: str | None
    candidate_answer: str
    candidate_level: str
    tech_stack: str
    
    # Оценки агентов
    correctness_score: AgentScore | None
    code_quality_score: AgentScore | None
    conceptual_score: AgentScore | None
    relevance_score: AgentScore | None
    level_alignment_score: AgentScore | None
    
    # Итог
    final_result: AssessmentResult | None


# ==============================
# Инициализация LLM
# ==============================

def get_llm():
    """Получить инстанс LLM."""
    return get_chat_llm(temperature=0.2)


def parse_json_response(content: str) -> dict | None:
    """Безопасно распарсить JSON из ответа LLM."""
    # Очищаем от markdown
    content = content.strip()
    if content.startswith('```'):
        lines = content.split('\n')
        # Убираем первую и последнюю строки с ```
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        content = '\n'.join(lines)
    
    content = content.strip()
    
    # Пробуем найти JSON в тексте
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Пробуем найти JSON между скобками
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def _agent_log(stage: str, message: str):
    debug_log(
        enabled=settings.ASSESSMENT_DEBUG_LOGS,
        stage=stage,
        message=message,
        console_limit=settings.DEBUG_CONSOLE_PREVIEW_LIMIT,
        file_limit=None,
        file_path=settings.DEBUG_LOG_FILE if settings.DEBUG_LOG_TO_FILE else None,
    )


def _normalize_agent_score(raw_score: object) -> int:
    """
    Нормализовать оценку агента к целому диапазону [1..5].
    """
    try:
        value = float(raw_score)
    except (TypeError, ValueError):
        value = 3.0
    value = round(value)
    if value < 1:
        return 1
    if value > 5:
        return 5
    return int(value)


def _normalize_followups(value: list[str] | None) -> list[str]:
    result: list[str] = []
    for item in value or []:
        text = str(item or '').strip()
        if text:
            result.append(text)
    return result[:3]


def _format_followups_for_prompt(follow_ups: list[str] | None) -> str:
    items = _normalize_followups(follow_ups)
    if not items:
        return 'Нет follow-up вопросов.'
    return '\n'.join(f'- {item}' for item in items)


def _parse_followups_answered(raw_value: object) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 0
    if value < 0:
        return 0
    return value


def _apply_followup_gate(score: int, follow_ups: list[str] | None, followups_answered: int) -> int:
    """
    Жесткие правила для верхней границы шкалы:
    - 5 допускается только при покрытии всех follow-up вопросов.
    - если покрыты все follow-up и агент дал 4, повышаем до 5.
    """
    total_followups = len(_normalize_followups(follow_ups))
    if total_followups > 0:
        if score == 5 and followups_answered < total_followups:
            return 4
        if score == 4 and followups_answered >= total_followups:
            return 5
    return score


SCORE_RUBRIC_1_TO_5 = '''
Используйте ЕДИНУЮ шкалу оценивания (только целые числа 1..5):
1 - Ответ отсутствует и/или не соответствует заданному вопросу и/или абсолютно некорректный
2 - Ответ верный, но практически не раскрывает заданный вопрос и/или слишком краткий
3 - Ответ верный, достаточно полно раскрывает заданный вопрос, но не содержит дополнений или не затрагивает follow-up вопросы
4 - Ответ верный, раскрывает заданный вопрос, содержит релевантные примеры или корректные ответы на follow-up вопросы
5 - Ответ верный, полностью раскрывает заданный вопрос и затрагивает смежные темы, демонстрирует глубокое понимание темы кандидатом, содержит примеры и корректные ответы на все follow-up вопросы
'''

AGENT_SCORING_POLICY = '''
Применяйте эту шкалу как общий ориентир итоговой полноты и качества ответа,
но выставляйте балл СТРОГО с учетом роли текущего агента и его профильных критериев.
'''


def _normalize_tag(tag: str) -> str:
    return (tag or '').strip().lower()


def _compute_retrieval_confidence(
    refs: list[dict],
    question_tags: list[str] | None,
    expected_top_k: int,
) -> tuple[str, float, str]:
    """
    Оценить доверие к retrieval-контексту.

    Комбинируем:
    - средний similarity/relevance score (если доступен),
    - долю совпадения тегов,
    - покрытие top-k.
    """
    if not refs:
        return 'low', 0.0, 'Нет найденных эталонных ответов'

    scores = [
        float(ref['relevance_score'])
        for ref in refs
        if ref.get('relevance_score') is not None
    ]
    avg_score = sum(scores) / len(scores) if scores else 0.40

    q_tags = {_normalize_tag(t) for t in (question_tags or []) if _normalize_tag(t)}
    if q_tags:
        matched = 0
        for ref in refs:
            ref_tags = {_normalize_tag(t) for t in ref.get('tags', []) if _normalize_tag(t)}
            if ref_tags & q_tags:
                matched += 1
        tag_match_ratio = matched / len(refs)
    else:
        tag_match_ratio = 0.5

    qid_match_ratio = sum(1 for ref in refs if ref.get('source_qid_match')) / len(refs)

    coverage_ratio = min(len(refs) / max(expected_top_k, 1), 1.0)

    combined = (
        0.50 * avg_score
        + 0.20 * tag_match_ratio
        + 0.15 * qid_match_ratio
        + 0.15 * coverage_ratio
    )
    if combined >= 0.72:
        level = 'high'
    elif combined >= 0.50:
        level = 'medium'
    else:
        level = 'low'

    explanation = (
        f'avg_score={avg_score:.2f}, '
        f'tag_match={tag_match_ratio:.2f}, '
        f'qid_match={qid_match_ratio:.2f}, '
        f'coverage={coverage_ratio:.2f}, '
        f'combined={combined:.2f}'
    )
    return level, round(combined, 3), explanation


# ==============================
# Промпты для агентов
# ==============================

CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Вы — эксперт по проверке корректности ответов на технических собеседованиях.
Ваша задача — оценить функциональную корректность и соответствие фактам.

Оцените по целочисленной шкале от 1 до 5, где:
- 1: фактически неверный или отсутствующий ответ
- 2: есть существенные ошибки в корректности
- 3: в целом верно, но с заметными неточностями
- 4: корректно с минимальными неточностями
- 5: полностью корректно и без фактических ошибок, при этом корректно покрыт основной и все follow-up вопросы

ЖЕСТКОЕ ПРАВИЛО: ставьте 5 ТОЛЬКО если корректно покрыты основной и все follow-up вопросы.

Учитывайте не только основной вопрос, но и follow-up вопросы.
Фокус этого агента: фактическая и функциональная корректность ответа.

Ответьте строго в формате JSON:
{{
    "score": <целое число от 1 до 5>,
    "feedback": "<краткий отзыв>",
    "errors": ["<список найденных ошибок>"],
    "correct_points": ["<список верных моментов>"],
    "follow_ups_answered": <целое число: сколько follow-up вопросов покрыто корректно>
}}'''),
    ('human', '''Вопрос: {question}
Тип вопроса: {question_type}
Технологии: {tech_stack}

Эталонные ответы:
{reference_answers}

Follow-up вопросы:
{follow_ups}

Ответ кандидата:
{answer}

Оцените корректность ответа.'''),
])

CODE_QUALITY_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Вы — эксперт по качеству кода и best practices.
Ваша задача — проанализировать качество кода в ответе (если он есть).

Оцените по целочисленной шкале от 1 до 5.
Баллы выставляйте по качеству инженерного решения и кода, а не по полноте теории.
ЖЕСТКОЕ ПРАВИЛО: 5 ставьте только если кроме качества решения корректно покрыты основной и все follow-up вопросы.

Дополнительно учитывайте:
- Читаемость и форматирование
- Следование best practices
- Эффективность решения
- Правильное именование
- Отсутствие code smells

Если в ответе нет кода, оцените структурированность и ясность изложения.
Фокус этого агента: инженерное качество решения и стиль изложения.

Ответьте строго в формате JSON:
{{
    "score": <целое число от 1 до 5>,
    "feedback": "<краткий отзыв>",
    "strengths": ["<сильные стороны>"],
    "improvements": ["<что можно улучшить>"],
    "follow_ups_answered": <целое число: сколько follow-up вопросов покрыто>
}}'''),
    ('human', '''Вопрос: {question}
Технологии: {tech_stack}

Эталонные ответы:
{reference_answers}

Follow-up вопросы:
{follow_ups}

Ответ кандидата:
{answer}

Оцените качество кода/изложения.'''),
])

CONCEPTUAL_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Вы — эксперт по оценке глубины понимания технических концепций.
Ваша задача — оценить, насколько глубоко кандидат понимает тему.

Оцените по целочисленной шкале от 1 до 5:
- 1: поверхностное понимание, нет причинно-следственных связей
- 2: базовое понимание, без глубины
- 3: уверенные основы, но без нюансов и trade-offs
- 4: хорошее понимание, объясняет почему и как
- 5: глубокое понимание, затрагивает нюансы и смежные темы, и корректно покрывает основной и все follow-up вопросы

ЖЕСТКОЕ ПРАВИЛО: 5 ставьте только при полном покрытии основного и всех follow-up вопросов.

Фокус этого агента: глубина понимания концепций, причин и trade-offs.

Ответьте строго в формате JSON:
{{
    "score": <целое число от 1 до 5>,
    "feedback": "<краткий отзыв>",
    "understanding_level": "<поверхностное|базовое|среднее|хорошее|глубокое>",
    "missed_concepts": ["<пропущенные важные концепции>"],
    "follow_ups_answered": <целое число: сколько follow-up вопросов покрыто корректно>
}}'''),
    ('human', '''Вопрос: {question}
Тип вопроса: {question_type}
Технологии: {tech_stack}

Эталонные ответы:
{reference_answers}

Follow-up вопросы:
{follow_ups}

Ответ кандидата:
{answer}

Оцените глубину понимания.'''),
])

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Вы — эксперт по оценке релевантности ответов.
Ваша задача — проверить, насколько ответ соответствует заданному вопросу.

Оцените по целочисленной шкале от 1 до 5:
- 1: ответ не по вопросу
- 2: частично по теме, много лишнего
- 3: по теме, но покрытие неполное
- 4: хорошо соответствует основному и части follow-up
- 5: полно и точно покрывает основной вопрос и ВСЕ follow-up вопросы

ЖЕСТКОЕ ПРАВИЛО: 5 возможно только если покрыты все follow-up вопросы.

Проверяйте релевантность к основному вопросу и follow-up вопросам.
Фокус этого агента: соответствие ответа поставленным вопросам.

Ответьте строго в формате JSON:
{{
    "score": <целое число от 1 до 5>,
    "feedback": "<краткий отзыв>",
    "answered_parts": ["<на что ответил>"],
    "missing_parts": ["<что не затронул>"],
    "off_topic": ["<что было лишним>"],
    "follow_ups_answered": <целое число: сколько follow-up вопросов покрыто>
}}'''),
    ('human', '''Вопрос: {question}

Эталонные ответы:
{reference_answers}

Follow-up вопросы:
{follow_ups}

Ответ кандидата:
{answer}

Оцените релевантность ответа.'''),
])

LEVEL_ALIGNMENT_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Вы — эксперт по оценке соответствия ответа уровню кандидата.
Ваша задача — оценить, соответствует ли ответ заявленному уровню.

Уровни:
- Junior: базовые знания, простые решения
- Middle: уверенное владение, понимание trade-offs
- Senior: глубокая экспертиза, архитектурное мышление

Оцените по целочисленной шкале от 1 до 5:
- 1: уровень ответа сильно ниже ожидаемого
- 2: заметно ниже ожидаемого уровня
- 3: частично соответствует уровню
- 4: хорошо соответствует уровню
- 5: полностью соответствует или превосходит ожидаемый уровень и корректно покрывает основной и все follow-up вопросы

ЖЕСТКОЕ ПРАВИЛО: 5 ставьте только при полном покрытии follow-up вопросов.

Оцените, насколько ответ соответствует ожиданиям от уровня {level}
с учетом основного вопроса и follow-up вопросов.
Фокус этого агента: соответствие глубины и качества ответа уровню кандидата.

Ответьте строго в формате JSON:
{{
    "score": <целое число от 1 до 5>,
    "feedback": "<краткий отзыв>",
    "actual_level": "<Junior|Middle|Senior>",
    "gap_analysis": "<анализ разрыва между ожидаемым и фактическим уровнем>",
    "follow_ups_answered": <целое число: сколько follow-up вопросов покрыто корректно>
}}'''),
    ('human', '''Ожидаемый уровень кандидата: {level}
Вопрос: {question}
Технологии: {tech_stack}

Эталонные ответы:
{reference_answers}

Follow-up вопросы:
{follow_ups}

Ответ кандидата:
{answer}

Оцените соответствие уровню.'''),
])

EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Вы — координатор оценки собеседования.
На основе оценок от разных экспертов сформируйте итоговую оценку и рекомендацию.

Веса для итоговой оценки:
- Корректность: 30%
- Качество кода/изложения: 15%
- Глубина понимания: 25%
- Релевантность: 15%
- Соответствие уровню: 15%

Сформируйте итоговый отзыв и рекомендацию.

Используйте агрегированные оценки профильных агентов по шкале 1..5.
Для итоговой рекомендации учитывайте агрегированные оценки профильных агентов,
не подменяйте их своей предметной экспертизой.

Ответьте строго в формате JSON:
{{
    "total_score": <взвешенная средняя оценка 1.0-5.0 с одним знаком после запятой>,
    "final_feedback": "<итоговый отзыв, 2-3 предложения>",
    "recommendation": "<сильный кандидат|хороший кандидат|требует развития|не соответствует уровню>",
    "key_strengths": ["<ключевые сильные стороны>"],
    "areas_to_improve": ["<области для развития>"]
}}'''),
    ('human', '''Вопрос: {question}
Уровень кандидата: {level}

Эталонные ответы:
{reference_answers}

Follow-up вопросы:
{follow_ups}

Оценки экспертов:

Корректность ({correctness_score}/5):
{correctness_feedback}

Качество кода ({code_quality_score}/5):
{code_quality_feedback}

Глубина понимания ({conceptual_score}/5):
{conceptual_feedback}

Релевантность ({relevance_score}/5):
{relevance_feedback}

Соответствие уровню ({level_alignment_score}/5):
{level_alignment_feedback}

Сформируйте итоговую оценку.'''),
])

FOLLOWUP_VALIDATE_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Вы — валидатор follow-up вопросов технического интервью.
Проверьте follow-up вопрос по тем же критериям, что и основной вопрос:
1) Ясность и конкретность
2) Соответствие уровню кандидата
3) Релевантность стеку (если стек задан)
4) Краткость
5) Однозначность (один вопрос, без двойных формулировок)

Верните строго JSON:
{{
  "approved": true/false,
  "reason_code": "<MISSING_TECH|MULTI_PART|LEVEL_MISMATCH|TOO_LONG|OTHER>",
  "reason_text": "<короткая причина>",
  "normalized_question": "<очищенный follow-up вопрос>"
}}'''),
    ('human', '''Основной вопрос:
{main_question}

Follow-up вопрос:
{follow_up}

Уровень: {level}
Технологии: {tech_stack}
Режим проверки технологий: {tech_validation_mode}'''),
])


# ==============================
# Агенты
# ==============================

# Веса агентов для расчёта общей оценки
AGENT_WEIGHTS = {
    'Correctness Agent': 0.30,
    'Code Quality Agent': 0.15,
    'Conceptual Understanding Agent': 0.25,
    'Relevance Agent': 0.15,
    'Level Alignment Agent': 0.15,
}


class CorrectnessAgent:
    """Агент проверки корректности ответа."""
    
    WEIGHT = AGENT_WEIGHTS['Correctness Agent']
    
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
        self.chain = CORRECTNESS_PROMPT | self.llm
    
    async def evaluate(
        self,
        question: str,
        answer: str,
        question_type: str,
        tech_stack: str,
        reference_answers: str = 'Эталонные ответы не найдены.',
        follow_ups: list[str] | None = None,
    ) -> AgentScore:
        """Оценить корректность ответа."""
        _agent_log('agent_start', f'CorrectnessAgent start question={question[:160]} answer={answer[:200]}')
        _agent_log('rag_result', f'CorrectnessAgent references={reference_answers}')
        try:
            response = await self.chain.ainvoke({
                'question': question,
                'answer': answer,
                'question_type': question_type,
                'tech_stack': tech_stack,
                'reference_answers': reference_answers,
                'follow_ups': _format_followups_for_prompt(follow_ups),
            })
            
            data = parse_json_response(response.content)
            if data:
                score = _normalize_agent_score(data.get('score', 3))
                followups_answered = _parse_followups_answered(data.get('follow_ups_answered', 0))
                score = _apply_followup_gate(score, follow_ups, followups_answered)
                return AgentScore(
                    agent_name='Correctness Agent',
                    score=score,
                    feedback=data.get('feedback', 'Оценка выполнена'),
                    details={
                        'errors': data.get('errors', []),
                        'correct_points': data.get('correct_points', []),
                        'follow_ups_answered': followups_answered,
                    },
                    error=False,
                    weight=self.WEIGHT,
                )
        except Exception as e:
            print(f'Ошибка CorrectnessAgent: {e}')
            _agent_log('agent_error', f'CorrectnessAgent error: {e}')
        
        return AgentScore(
            agent_name='Correctness Agent',
            score=0.0,
            feedback='⚠️ Агент завершился с ошибкой',
            details=None,
            error=True,
            weight=self.WEIGHT,
        )


class CodeQualityAgent:
    """Агент оценки качества кода."""
    
    WEIGHT = AGENT_WEIGHTS['Code Quality Agent']
    
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
        self.chain = CODE_QUALITY_PROMPT | self.llm
    
    async def evaluate(
        self,
        question: str,
        answer: str,
        tech_stack: str,
        reference_answers: str = 'Эталонные ответы не найдены.',
        follow_ups: list[str] | None = None,
    ) -> AgentScore:
        """Оценить качество кода/изложения."""
        _agent_log('agent_start', f'CodeQualityAgent start question={question[:160]} answer={answer[:200]}')
        _agent_log('rag_result', f'CodeQualityAgent references={reference_answers}')
        try:
            response = await self.chain.ainvoke({
                'question': question,
                'answer': answer,
                'tech_stack': tech_stack,
                'reference_answers': reference_answers,
                'follow_ups': _format_followups_for_prompt(follow_ups),
            })
            
            data = parse_json_response(response.content)
            if data:
                score = _normalize_agent_score(data.get('score', 3))
                followups_answered = _parse_followups_answered(data.get('follow_ups_answered', 0))
                score = _apply_followup_gate(score, follow_ups, followups_answered)
                return AgentScore(
                    agent_name='Code Quality Agent',
                    score=score,
                    feedback=data.get('feedback', 'Оценка выполнена'),
                    details={
                        'strengths': data.get('strengths', []),
                        'improvements': data.get('improvements', []),
                        'follow_ups_answered': followups_answered,
                    },
                    error=False,
                    weight=self.WEIGHT,
                )
        except Exception as e:
            print(f'Ошибка CodeQualityAgent: {e}')
            _agent_log('agent_error', f'CodeQualityAgent error: {e}')
        
        return AgentScore(
            agent_name='Code Quality Agent',
            score=0.0,
            feedback='⚠️ Агент завершился с ошибкой',
            details=None,
            error=True,
            weight=self.WEIGHT,
        )


class ConceptualUnderstandingAgent:
    """Агент оценки глубины понимания."""
    
    WEIGHT = AGENT_WEIGHTS['Conceptual Understanding Agent']
    
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
        self.chain = CONCEPTUAL_PROMPT | self.llm
    
    async def evaluate(
        self,
        question: str,
        answer: str,
        question_type: str,
        tech_stack: str,
        reference_answers: str = 'Эталонные ответы не найдены.',
        follow_ups: list[str] | None = None,
    ) -> AgentScore:
        """Оценить глубину понимания."""
        _agent_log('agent_start', f'ConceptualUnderstandingAgent start question={question[:160]} answer={answer[:200]}')
        _agent_log('rag_result', f'ConceptualUnderstandingAgent references={reference_answers}')
        try:
            response = await self.chain.ainvoke({
                'question': question,
                'answer': answer,
                'question_type': question_type,
                'tech_stack': tech_stack,
                'reference_answers': reference_answers,
                'follow_ups': _format_followups_for_prompt(follow_ups),
            })
            
            data = parse_json_response(response.content)
            if data:
                score = _normalize_agent_score(data.get('score', 3))
                followups_answered = _parse_followups_answered(data.get('follow_ups_answered', 0))
                score = _apply_followup_gate(score, follow_ups, followups_answered)
                return AgentScore(
                    agent_name='Conceptual Understanding Agent',
                    score=score,
                    feedback=data.get('feedback', 'Оценка выполнена'),
                    details={
                        'understanding_level': data.get('understanding_level', ''),
                        'missed_concepts': data.get('missed_concepts', []),
                        'follow_ups_answered': followups_answered,
                    },
                    error=False,
                    weight=self.WEIGHT,
                )
        except Exception as e:
            print(f'Ошибка ConceptualUnderstandingAgent: {e}')
            _agent_log('agent_error', f'ConceptualUnderstandingAgent error: {e}')
        
        return AgentScore(
            agent_name='Conceptual Understanding Agent',
            score=0.0,
            feedback='⚠️ Агент завершился с ошибкой',
            details=None,
            error=True,
            weight=self.WEIGHT,
        )


class RelevanceAgent:
    """Агент проверки релевантности ответа."""
    
    WEIGHT = AGENT_WEIGHTS['Relevance Agent']
    
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
        self.chain = RELEVANCE_PROMPT | self.llm
    
    async def evaluate(
        self,
        question: str,
        answer: str,
        reference_answers: str = 'Эталонные ответы не найдены.',
        follow_ups: list[str] | None = None,
    ) -> AgentScore:
        """Оценить релевантность ответа."""
        _agent_log('agent_start', f'RelevanceAgent start question={question[:160]} answer={answer[:200]}')
        _agent_log('rag_result', f'RelevanceAgent references={reference_answers}')
        try:
            response = await self.chain.ainvoke({
                'question': question,
                'answer': answer,
                'reference_answers': reference_answers,
                'follow_ups': _format_followups_for_prompt(follow_ups),
            })
            
            data = parse_json_response(response.content)
            if data:
                score = _normalize_agent_score(data.get('score', 3))
                followups_answered = _parse_followups_answered(data.get('follow_ups_answered', 0))
                score = _apply_followup_gate(score, follow_ups, followups_answered)
                return AgentScore(
                    agent_name='Relevance Agent',
                    score=score,
                    feedback=data.get('feedback', 'Оценка выполнена'),
                    details={
                        'answered_parts': data.get('answered_parts', []),
                        'missing_parts': data.get('missing_parts', []),
                        'off_topic': data.get('off_topic', []),
                        'follow_ups_answered': followups_answered,
                    },
                    error=False,
                    weight=self.WEIGHT,
                )
        except Exception as e:
            print(f'Ошибка RelevanceAgent: {e}')
            _agent_log('agent_error', f'RelevanceAgent error: {e}')
        
        return AgentScore(
            agent_name='Relevance Agent',
            score=0.0,
            feedback='⚠️ Агент завершился с ошибкой',
            details=None,
            error=True,
            weight=self.WEIGHT,
        )


class LevelAlignmentAgent:
    """Агент оценки соответствия уровню кандидата."""
    
    WEIGHT = AGENT_WEIGHTS['Level Alignment Agent']
    
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
        self.chain = LEVEL_ALIGNMENT_PROMPT | self.llm
    
    async def evaluate(
        self,
        question: str,
        answer: str,
        level: str,
        tech_stack: str,
        reference_answers: str = 'Эталонные ответы не найдены.',
        follow_ups: list[str] | None = None,
    ) -> AgentScore:
        """Оценить соответствие уровню."""
        _agent_log('agent_start', f'LevelAlignmentAgent start question={question[:160]} answer={answer[:200]}')
        _agent_log('rag_result', f'LevelAlignmentAgent references={reference_answers}')
        try:
            response = await self.chain.ainvoke({
                'question': question,
                'answer': answer,
                'level': level,
                'tech_stack': tech_stack,
                'reference_answers': reference_answers,
                'follow_ups': _format_followups_for_prompt(follow_ups),
            })
            
            data = parse_json_response(response.content)
            if data:
                score = _normalize_agent_score(data.get('score', 3))
                followups_answered = _parse_followups_answered(data.get('follow_ups_answered', 0))
                score = _apply_followup_gate(score, follow_ups, followups_answered)
                return AgentScore(
                    agent_name='Level Alignment Agent',
                    score=score,
                    feedback=data.get('feedback', 'Оценка выполнена'),
                    details={
                        'actual_level': data.get('actual_level', ''),
                        'gap_analysis': data.get('gap_analysis', ''),
                        'follow_ups_answered': followups_answered,
                    },
                    error=False,
                    weight=self.WEIGHT,
                )
        except Exception as e:
            print(f'Ошибка LevelAlignmentAgent: {e}')
            _agent_log('agent_error', f'LevelAlignmentAgent error: {e}')
        
        return AgentScore(
            agent_name='Level Alignment Agent',
            score=0.0,
            feedback='⚠️ Агент завершился с ошибкой',
            details=None,
            error=True,
            weight=self.WEIGHT,
        )


class ExplanationAgent:
    """Агент формирования итоговой оценки."""
    
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
        self.chain = EXPLANATION_PROMPT | self.llm
    
    @staticmethod
    def _calculate_weighted_score(agent_scores: list[AgentScore]) -> tuple[float, int]:
        """
        Рассчитать взвешенную оценку, исключая агентов с ошибками.
        
        Returns:
            (weighted_score, success_count) - оценка и количество успешных агентов
        """
        # Фильтруем только успешных агентов
        valid_scores = [s for s in agent_scores if not s.error]
        
        if not valid_scores:
            return 0.0, 0
        
        # Пересчитываем веса для успешных агентов
        total_weight = sum(s.weight for s in valid_scores)
        
        if total_weight == 0:
            return 0.0, 0
        
        # Нормализуем веса и считаем оценку
        weighted_sum = sum(s.score * (s.weight / total_weight) for s in valid_scores)
        
        return weighted_sum, len(valid_scores)
    
    async def compile_assessment(
        self,
        question: str,
        answer: str,
        level: str,
        reference_answers: str,
        follow_ups: list[str] | None,
        correctness: AgentScore,
        code_quality: AgentScore,
        conceptual: AgentScore,
        relevance: AgentScore,
        level_alignment: AgentScore,
    ) -> AssessmentResult:
        """Сформировать итоговую оценку."""
        agent_scores = [correctness, code_quality, conceptual, relevance, level_alignment]
        
        # Рассчитаем взвешенную оценку (исключая агентов с ошибками)
        weighted_score, success_count = self._calculate_weighted_score(agent_scores)
        weighted_score = round(weighted_score, 1)
        error_count = len(agent_scores) - success_count
        
        # Если все агенты с ошибками - возвращаем результат с нулевой оценкой
        if success_count == 0:
            return AssessmentResult(
                question=question,
                answer=answer,
                total_score=0.0,
                agent_scores=agent_scores,
                final_feedback=f'⚠️ Все {error_count} агентов завершились с ошибкой. Оценка невозможна.',
                recommendation='Требуется ручная проверка',
            )
        
        # Формируем информацию об ошибках для промпта
        error_note = ''
        if error_count > 0:
            error_agents = [s.agent_name for s in agent_scores if s.error]
            error_note = f'\n\nВНИМАНИЕ: {error_count} агент(ов) завершились с ошибкой и не учитываются: {", ".join(error_agents)}'
        
        try:
            response = await self.chain.ainvoke({
                'question': question,
                'level': level,
                'reference_answers': reference_answers,
                'follow_ups': _format_followups_for_prompt(follow_ups),
                'correctness_score': correctness.score if not correctness.error else 'ОШИБКА',
                'correctness_feedback': correctness.feedback,
                'code_quality_score': code_quality.score if not code_quality.error else 'ОШИБКА',
                'code_quality_feedback': code_quality.feedback,
                'conceptual_score': conceptual.score if not conceptual.error else 'ОШИБКА',
                'conceptual_feedback': conceptual.feedback,
                'relevance_score': relevance.score if not relevance.error else 'ОШИБКА',
                'relevance_feedback': relevance.feedback,
                'level_alignment_score': level_alignment.score if not level_alignment.error else 'ОШИБКА',
                'level_alignment_feedback': level_alignment.feedback,
            })
            
            data = parse_json_response(response.content)
            if data:
                final_feedback = data.get('final_feedback', 'Оценка сформирована на основе анализа экспертов.')
                if error_count > 0:
                    final_feedback += f' (⚠️ {error_count} агент(ов) не учтены из-за ошибок)'
                
                return AssessmentResult(
                    question=question,
                    answer=answer,
                    total_score=weighted_score,  # Используем пересчитанную оценку
                    agent_scores=agent_scores,
                    final_feedback=final_feedback,
                    recommendation=data.get('recommendation', 'Требуется ручная проверка'),
                )
        except Exception as e:
            print(f'Ошибка ExplanationAgent: {e}')
        
        # Fallback - возвращаем результат с рассчитанной оценкой
        final_feedback = 'Оценка сформирована автоматически на основе весов экспертов.'
        if error_count > 0:
            final_feedback += f' (⚠️ {error_count} агент(ов) не учтены из-за ошибок)'
        
        return AssessmentResult(
            question=question,
            answer=answer,
            total_score=weighted_score,
            agent_scores=agent_scores,
            final_feedback=final_feedback,
            recommendation=self._get_recommendation(weighted_score),
        )
    
    @staticmethod
    def _get_recommendation(score: float) -> str:
        """Получить рекомендацию по оценке."""
        if score >= 4.5:
            return 'Сильный кандидат'
        elif score >= 3.5:
            return 'Хороший кандидат'
        elif score >= 2.5:
            return 'Требует развития'
        else:
            return 'Не соответствует уровню'


class AssessmentCoordinator:
    """
    Координатор оценки ответов.
    Запускает всех агентов и собирает итоговый результат.
    
    Режим работы настраивается через settings.API_MODE:
    - 'parallel': все агенты запускаются параллельно (быстро, но может вызвать 429)
    - 'sequential': агенты запускаются последовательно с задержкой (медленнее, но надёжнее)
    """
    
    def __init__(self, llm=None, mode: str = None, request_delay: float = None):
        """
        Args:
            llm: Экземпляр LLM (по умолчанию создаётся из settings)
            mode: Режим работы ('parallel' или 'sequential'). По умолчанию из settings.API_MODE
            request_delay: Задержка между запросами в секундах. По умолчанию из settings.API_REQUEST_DELAY
        """
        self.llm = llm or get_llm()
        self.mode = mode or settings.API_MODE
        self.request_delay = request_delay if request_delay is not None else settings.API_REQUEST_DELAY
        self.retriever = None
        try:
            self.retriever = LocalRAGRetriever()
        except Exception as e:
            print(f'RAG-база недоступна, оценка без эталонных ответов: {e}')
        
        self.correctness_agent = CorrectnessAgent(self.llm)
        self.code_quality_agent = CodeQualityAgent(self.llm)
        self.conceptual_agent = ConceptualUnderstandingAgent(self.llm)
        self.relevance_agent = RelevanceAgent(self.llm)
        self.level_alignment_agent = LevelAlignmentAgent(self.llm)
        self.explanation_agent = ExplanationAgent(self.llm)

    @staticmethod
    def _is_unknown_tech_stack(value: str) -> bool:
        text = (value or '').strip().lower()
        return text in {'', 'не указано', 'unknown', 'n/a', '-'}

    @staticmethod
    def _has_explicit_tech_signal(text: str) -> bool:
        q = (text or '').lower()
        markers = (
            'python', 'fastapi', 'django', 'flask', 'sql', 'postgres', 'mysql',
            'redis', 'mongodb', 'docker', 'kubernetes', 'asyncio', 'sqlalchemy',
            'asyncpg', 'grpc', 'jwt', 'oauth', 'api', 'rest', 'graphql',
        )
        return any(m in q for m in markers)

    async def _validate_followups(
        self,
        main_question: str,
        follow_ups: list[str] | None,
        candidate_level: str,
        tech_stack: str,
    ) -> list[str]:
        items = _normalize_followups(follow_ups)
        if not items:
            return []

        chain = FOLLOWUP_VALIDATE_PROMPT | self.llm
        tech_mode = 'SOFT' if self._is_unknown_tech_stack(tech_stack) else 'STRICT'
        approved: list[str] = []

        for fu in items:
            try:
                response = await chain.ainvoke({
                    'main_question': main_question,
                    'follow_up': fu,
                    'level': candidate_level,
                    'tech_stack': tech_stack,
                    'tech_validation_mode': tech_mode,
                })
                data = parse_json_response(response.content) or {}
                fu_approved = bool(data.get('approved'))
                reason_code = str(data.get('reason_code', '')).upper().strip()
                normalized_question = str(data.get('normalized_question', fu)).strip() or fu
                if (
                    not fu_approved
                    and reason_code == 'MISSING_TECH'
                    and (tech_mode == 'SOFT' or self._has_explicit_tech_signal(normalized_question))
                ):
                    fu_approved = True
                if fu_approved:
                    approved.append(normalized_question)
                else:
                    _agent_log('warning', f'Follow-up rejected: {fu} | reason={data.get("reason_text", "")}')
            except Exception as e:
                _agent_log('agent_error', f'Follow-up validation error: {e}')
                # Fallback: не теряем follow-up при сбое валидатора.
                approved.append(fu)

        return approved
    
    async def _run_parallel(
        self,
        question: str,
        answer: str,
        question_type: str,
        tech_stack: str,
        candidate_level: str,
        reference_answers: str,
        follow_ups: list[str] | None = None,
    ) -> tuple:
        """Запустить агентов параллельно."""
        import asyncio
        
        results = await asyncio.gather(
            self.correctness_agent.evaluate(question, answer, question_type, tech_stack, reference_answers, follow_ups),
            self.code_quality_agent.evaluate(question, answer, tech_stack, reference_answers, follow_ups),
            self.conceptual_agent.evaluate(question, answer, question_type, tech_stack, reference_answers, follow_ups),
            self.relevance_agent.evaluate(question, answer, reference_answers, follow_ups),
            self.level_alignment_agent.evaluate(question, answer, candidate_level, tech_stack, reference_answers, follow_ups),
        )
        return results
    
    async def _run_sequential(
        self,
        question: str,
        answer: str,
        question_type: str,
        tech_stack: str,
        candidate_level: str,
        reference_answers: str,
        follow_ups: list[str] | None = None,
    ) -> tuple:
        """Запустить агентов последовательно с задержкой."""
        import asyncio
        
        print('  → Correctness Agent...')
        correctness = await self.correctness_agent.evaluate(
            question, answer, question_type, tech_stack, reference_answers, follow_ups
        )
        await asyncio.sleep(self.request_delay)
        
        print('  → Code Quality Agent...')
        code_quality = await self.code_quality_agent.evaluate(
            question, answer, tech_stack, reference_answers, follow_ups
        )
        await asyncio.sleep(self.request_delay)
        
        print('  → Conceptual Understanding Agent...')
        conceptual = await self.conceptual_agent.evaluate(
            question, answer, question_type, tech_stack, reference_answers, follow_ups
        )
        await asyncio.sleep(self.request_delay)
        
        print('  → Relevance Agent...')
        relevance = await self.relevance_agent.evaluate(
            question, answer, reference_answers, follow_ups
        )
        await asyncio.sleep(self.request_delay)
        
        print('  → Level Alignment Agent...')
        level_alignment = await self.level_alignment_agent.evaluate(
            question, answer, candidate_level, tech_stack, reference_answers, follow_ups
        )
        await asyncio.sleep(self.request_delay)
        
        return (correctness, code_quality, conceptual, relevance, level_alignment)
    
    async def assess_answer(
        self,
        question: str,
        answer: str,
        question_type: str = 'теория',
        tech_stack: str = 'Python',
        candidate_level: str = 'Middle',
        question_tags: list[str] | None = None,
        follow_ups: list[str] | None = None,
    ) -> AssessmentResult:
        """
        Оценить ответ кандидата с помощью всех агентов.
        Режим работы (parallel/sequential) берётся из конфигурации.
        """
        import asyncio
        
        reference_context = 'Эталонные ответы не найдены.'
        refs: list[dict] = []
        if self.retriever:
            try:
                reference_context, refs = self.retriever.build_context_for_assessment(
                    question_text=question,
                    tags=question_tags or [],
                    top_k=settings.RAG_TOP_K_EVAL,
                )
            except Exception as e:
                print(f'Не удалось получить эталонные ответы из RAG: {e}')
        retrieval_confidence, retrieval_score, retrieval_explanation = _compute_retrieval_confidence(
            refs=refs,
            question_tags=question_tags,
            expected_top_k=settings.RAG_TOP_K_EVAL,
        )
        _agent_log(
            'rag_result',
            (
                f'AssessmentCoordinator references_count={len(refs)} '
                f'confidence={retrieval_confidence} '
                f'score={retrieval_score:.3f} '
                f'details={retrieval_explanation}'
            ),
        )
        _agent_log('rag_result', f'AssessmentCoordinator references: {reference_context}')

        validated_follow_ups = await self._validate_followups(
            main_question=question,
            follow_ups=follow_ups,
            candidate_level=candidate_level,
            tech_stack=tech_stack,
        )
        _agent_log(
            'rag_result',
            f'Validated follow-ups: total={len(_normalize_followups(follow_ups))} approved={len(validated_follow_ups)}',
        )

        # Выбираем режим выполнения
        if self.mode == 'sequential':
            results = await self._run_sequential(
                question, answer, question_type, tech_stack, candidate_level, reference_context, validated_follow_ups
            )
        else:
            results = await self._run_parallel(
                question, answer, question_type, tech_stack, candidate_level, reference_context, validated_follow_ups
            )
        
        correctness, code_quality, conceptual, relevance, level_alignment = results
        
        # Задержка перед финальным агентом в sequential режиме
        if self.mode == 'sequential':
            print('  → Explanation Agent...')
        
        # Формируем итоговую оценку
        final_result = await self.explanation_agent.compile_assessment(
            question=question,
            answer=answer,
            level=candidate_level,
            reference_answers=reference_context,
            follow_ups=validated_follow_ups,
            correctness=correctness,
            code_quality=code_quality,
            conceptual=conceptual,
            relevance=relevance,
            level_alignment=level_alignment,
        )
        final_result.retrieval_confidence = retrieval_confidence
        final_result.retrieval_score = retrieval_score
        final_result.retrieval_references = len(refs)

        if retrieval_confidence == 'low':
            final_result.final_feedback += (
                ' (⚠️ Низкая релевантность retrieval-контекста; '
                'рекомендуется ручная перепроверка)'
            )
        elif retrieval_confidence == 'medium':
            final_result.final_feedback += (
                ' (ℹ️ Средняя релевантность retrieval-контекста)'
            )
        _agent_log(
            'agent_result',
            (
                f'Assessment finished total_score={final_result.total_score:.2f} '
                f'recommendation={final_result.recommendation} '
                f'retrieval_confidence={final_result.retrieval_confidence} '
                f'retrieval_score={final_result.retrieval_score}'
            ),
        )
        
        return final_result


# ==============================
# Вспомогательные функции
# ==============================

def assessment_result_to_dict(result: AssessmentResult) -> dict:
    """Преобразовать результат оценки в словарь."""
    # Считаем количество ошибок
    error_count = sum(1 for s in result.agent_scores if s.error)
    success_count = len(result.agent_scores) - error_count
    
    return {
        'question': result.question,
        'answer': result.answer,
        'total_score': result.total_score,
        'final_feedback': result.final_feedback,
        'recommendation': result.recommendation,
        'retrieval_confidence': result.retrieval_confidence,
        'retrieval_score': result.retrieval_score,
        'retrieval_references': result.retrieval_references,
        'agents_success': success_count,
        'agents_error': error_count,
        'agent_scores': [
            {
                'agent_name': score.agent_name,
                'score': score.score,
                'feedback': score.feedback,
                'details': score.details,
                'error': score.error,
                'weight': score.weight,
            }
            for score in result.agent_scores
        ],
    }


def dict_to_assessment_result(data: dict) -> AssessmentResult:
    """Преобразовать словарь в результат оценки."""
    agent_scores = [
        AgentScore(
            agent_name=s['agent_name'],
            score=s['score'],
            feedback=s['feedback'],
            details=s.get('details'),
            error=s.get('error', False),
            weight=s.get('weight', 0.0),
        )
        for s in data.get('agent_scores', [])
    ]
    
    return AssessmentResult(
        question=data['question'],
        answer=data['answer'],
        total_score=data['total_score'],
        agent_scores=agent_scores,
        final_feedback=data['final_feedback'],
        recommendation=data['recommendation'],
        retrieval_confidence=data.get('retrieval_confidence', 'unknown'),
        retrieval_score=data.get('retrieval_score'),
        retrieval_references=data.get('retrieval_references', 0),
    )
