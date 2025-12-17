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
from langchain_gigachat import GigaChat

import settings


# ==============================
# Модели данных
# ==============================

@dataclass
class AgentScore:
    """Оценка от одного агента."""
    agent_name: str
    score: float  # 0-10
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
    return GigaChat(
        credentials=settings.MODEL_API_KEY,
        verify_ssl_certs=False,
        model=settings.MODEL_NAME,
        temperature=0.2,
        scope='GIGACHAT_API_PERS',
    )


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


# ==============================
# Промпты для агентов
# ==============================

CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Вы — эксперт по проверке корректности ответов на технических собеседованиях.
Ваша задача — оценить функциональную корректность и соответствие фактам.

Оцените по шкале от 0 до 10, где:
- 0-2: Полностью неверный ответ или грубые ошибки
- 3-4: Есть существенные ошибки
- 5-6: Частично верно, есть неточности
- 7-8: В целом верно, минимальные неточности
- 9-10: Полностью корректный ответ

Ответьте строго в формате JSON:
{{
    "score": <число от 0 до 10>,
    "feedback": "<краткий отзыв>",
    "errors": ["<список найденных ошибок>"],
    "correct_points": ["<список верных моментов>"]
}}'''),
    ('human', '''Вопрос: {question}
Тип вопроса: {question_type}
Технологии: {tech_stack}

Ответ кандидата:
{answer}

Оцените корректность ответа.'''),
])

CODE_QUALITY_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Вы — эксперт по качеству кода и best practices.
Ваша задача — проанализировать качество кода в ответе (если он есть).

Оцените по шкале от 0 до 10:
- Читаемость и форматирование
- Следование best practices
- Эффективность решения
- Правильное именование
- Отсутствие code smells

Если в ответе нет кода, оцените структурированность и ясность изложения.

Ответьте строго в формате JSON:
{{
    "score": <число от 0 до 10>,
    "feedback": "<краткий отзыв>",
    "strengths": ["<сильные стороны>"],
    "improvements": ["<что можно улучшить>"]
}}'''),
    ('human', '''Вопрос: {question}
Технологии: {tech_stack}

Ответ кандидата:
{answer}

Оцените качество кода/изложения.'''),
])

CONCEPTUAL_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Вы — эксперт по оценке глубины понимания технических концепций.
Ваша задача — оценить, насколько глубоко кандидат понимает тему.

Оцените по шкале от 0 до 10:
- 0-2: Поверхностное понимание, только заученные определения
- 3-4: Базовое понимание без глубины
- 5-6: Среднее понимание, знает основы
- 7-8: Хорошее понимание, может объяснить почему
- 9-10: Глубокое понимание, видит связи и нюансы

Ответьте строго в формате JSON:
{{
    "score": <число от 0 до 10>,
    "feedback": "<краткий отзыв>",
    "understanding_level": "<поверхностное|базовое|среднее|хорошее|глубокое>",
    "missed_concepts": ["<пропущенные важные концепции>"]
}}'''),
    ('human', '''Вопрос: {question}
Тип вопроса: {question_type}
Технологии: {tech_stack}

Ответ кандидата:
{answer}

Оцените глубину понимания.'''),
])

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ('system', '''Вы — эксперт по оценке релевантности ответов.
Ваша задача — проверить, насколько ответ соответствует заданному вопросу.

Оцените по шкале от 0 до 10:
- 0-2: Ответ не по теме
- 3-4: Частично по теме, много лишнего
- 5-6: В целом по теме, есть отклонения
- 7-8: Хорошо соответствует вопросу
- 9-10: Точно отвечает на вопрос

Ответьте строго в формате JSON:
{{
    "score": <число от 0 до 10>,
    "feedback": "<краткий отзыв>",
    "answered_parts": ["<на что ответил>"],
    "missing_parts": ["<что не затронул>"],
    "off_topic": ["<что было лишним>"]
}}'''),
    ('human', '''Вопрос: {question}

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

Оцените по шкале от 0 до 10, насколько ответ соответствует ожиданиям от уровня {level}.

Ответьте строго в формате JSON:
{{
    "score": <число от 0 до 10>,
    "feedback": "<краткий отзыв>",
    "actual_level": "<Junior|Middle|Senior>",
    "gap_analysis": "<анализ разрыва между ожидаемым и фактическим уровнем>"
}}'''),
    ('human', '''Ожидаемый уровень кандидата: {level}
Вопрос: {question}
Технологии: {tech_stack}

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

Ответьте строго в формате JSON:
{{
    "total_score": <взвешенная средняя оценка>,
    "final_feedback": "<итоговый отзыв, 2-3 предложения>",
    "recommendation": "<сильный кандидат|хороший кандидат|требует развития|не соответствует уровню>",
    "key_strengths": ["<ключевые сильные стороны>"],
    "areas_to_improve": ["<области для развития>"]
}}'''),
    ('human', '''Вопрос: {question}
Уровень кандидата: {level}

Оценки экспертов:

Корректность ({correctness_score}/10):
{correctness_feedback}

Качество кода ({code_quality_score}/10):
{code_quality_feedback}

Глубина понимания ({conceptual_score}/10):
{conceptual_feedback}

Релевантность ({relevance_score}/10):
{relevance_feedback}

Соответствие уровню ({level_alignment_score}/10):
{level_alignment_feedback}

Сформируйте итоговую оценку.'''),
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
    
    async def evaluate(self, question: str, answer: str, 
                       question_type: str, tech_stack: str) -> AgentScore:
        """Оценить корректность ответа."""
        try:
            response = await self.chain.ainvoke({
                'question': question,
                'answer': answer,
                'question_type': question_type,
                'tech_stack': tech_stack,
            })
            
            data = parse_json_response(response.content)
            if data:
                return AgentScore(
                    agent_name='Correctness Agent',
                    score=float(data.get('score', 5)),
                    feedback=data.get('feedback', 'Оценка выполнена'),
                    details={
                        'errors': data.get('errors', []),
                        'correct_points': data.get('correct_points', []),
                    },
                    error=False,
                    weight=self.WEIGHT,
                )
        except Exception as e:
            print(f'Ошибка CorrectnessAgent: {e}')
        
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
    
    async def evaluate(self, question: str, answer: str, 
                       tech_stack: str) -> AgentScore:
        """Оценить качество кода/изложения."""
        try:
            response = await self.chain.ainvoke({
                'question': question,
                'answer': answer,
                'tech_stack': tech_stack,
            })
            
            data = parse_json_response(response.content)
            if data:
                return AgentScore(
                    agent_name='Code Quality Agent',
                    score=float(data.get('score', 5)),
                    feedback=data.get('feedback', 'Оценка выполнена'),
                    details={
                        'strengths': data.get('strengths', []),
                        'improvements': data.get('improvements', []),
                    },
                    error=False,
                    weight=self.WEIGHT,
                )
        except Exception as e:
            print(f'Ошибка CodeQualityAgent: {e}')
        
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
    
    async def evaluate(self, question: str, answer: str,
                       question_type: str, tech_stack: str) -> AgentScore:
        """Оценить глубину понимания."""
        try:
            response = await self.chain.ainvoke({
                'question': question,
                'answer': answer,
                'question_type': question_type,
                'tech_stack': tech_stack,
            })
            
            data = parse_json_response(response.content)
            if data:
                return AgentScore(
                    agent_name='Conceptual Understanding Agent',
                    score=float(data.get('score', 5)),
                    feedback=data.get('feedback', 'Оценка выполнена'),
                    details={
                        'understanding_level': data.get('understanding_level', ''),
                        'missed_concepts': data.get('missed_concepts', []),
                    },
                    error=False,
                    weight=self.WEIGHT,
                )
        except Exception as e:
            print(f'Ошибка ConceptualUnderstandingAgent: {e}')
        
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
    
    async def evaluate(self, question: str, answer: str) -> AgentScore:
        """Оценить релевантность ответа."""
        try:
            response = await self.chain.ainvoke({
                'question': question,
                'answer': answer,
            })
            
            data = parse_json_response(response.content)
            if data:
                return AgentScore(
                    agent_name='Relevance Agent',
                    score=float(data.get('score', 5)),
                    feedback=data.get('feedback', 'Оценка выполнена'),
                    details={
                        'answered_parts': data.get('answered_parts', []),
                        'missing_parts': data.get('missing_parts', []),
                        'off_topic': data.get('off_topic', []),
                    },
                    error=False,
                    weight=self.WEIGHT,
                )
        except Exception as e:
            print(f'Ошибка RelevanceAgent: {e}')
        
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
    
    async def evaluate(self, question: str, answer: str,
                       level: str, tech_stack: str) -> AgentScore:
        """Оценить соответствие уровню."""
        try:
            response = await self.chain.ainvoke({
                'question': question,
                'answer': answer,
                'level': level,
                'tech_stack': tech_stack,
            })
            
            data = parse_json_response(response.content)
            if data:
                return AgentScore(
                    agent_name='Level Alignment Agent',
                    score=float(data.get('score', 5)),
                    feedback=data.get('feedback', 'Оценка выполнена'),
                    details={
                        'actual_level': data.get('actual_level', ''),
                        'gap_analysis': data.get('gap_analysis', ''),
                    },
                    error=False,
                    weight=self.WEIGHT,
                )
        except Exception as e:
            print(f'Ошибка LevelAlignmentAgent: {e}')
        
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
        if score >= 8:
            return 'Сильный кандидат'
        elif score >= 6:
            return 'Хороший кандидат'
        elif score >= 4:
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
        
        self.correctness_agent = CorrectnessAgent(self.llm)
        self.code_quality_agent = CodeQualityAgent(self.llm)
        self.conceptual_agent = ConceptualUnderstandingAgent(self.llm)
        self.relevance_agent = RelevanceAgent(self.llm)
        self.level_alignment_agent = LevelAlignmentAgent(self.llm)
        self.explanation_agent = ExplanationAgent(self.llm)
    
    async def _run_parallel(
        self,
        question: str,
        answer: str,
        question_type: str,
        tech_stack: str,
        candidate_level: str,
    ) -> tuple:
        """Запустить агентов параллельно."""
        import asyncio
        
        results = await asyncio.gather(
            self.correctness_agent.evaluate(question, answer, question_type, tech_stack),
            self.code_quality_agent.evaluate(question, answer, tech_stack),
            self.conceptual_agent.evaluate(question, answer, question_type, tech_stack),
            self.relevance_agent.evaluate(question, answer),
            self.level_alignment_agent.evaluate(question, answer, candidate_level, tech_stack),
        )
        return results
    
    async def _run_sequential(
        self,
        question: str,
        answer: str,
        question_type: str,
        tech_stack: str,
        candidate_level: str,
    ) -> tuple:
        """Запустить агентов последовательно с задержкой."""
        import asyncio
        
        print('  → Correctness Agent...')
        correctness = await self.correctness_agent.evaluate(
            question, answer, question_type, tech_stack
        )
        await asyncio.sleep(self.request_delay)
        
        print('  → Code Quality Agent...')
        code_quality = await self.code_quality_agent.evaluate(
            question, answer, tech_stack
        )
        await asyncio.sleep(self.request_delay)
        
        print('  → Conceptual Understanding Agent...')
        conceptual = await self.conceptual_agent.evaluate(
            question, answer, question_type, tech_stack
        )
        await asyncio.sleep(self.request_delay)
        
        print('  → Relevance Agent...')
        relevance = await self.relevance_agent.evaluate(
            question, answer
        )
        await asyncio.sleep(self.request_delay)
        
        print('  → Level Alignment Agent...')
        level_alignment = await self.level_alignment_agent.evaluate(
            question, answer, candidate_level, tech_stack
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
    ) -> AssessmentResult:
        """
        Оценить ответ кандидата с помощью всех агентов.
        Режим работы (parallel/sequential) берётся из конфигурации.
        """
        import asyncio
        
        # Выбираем режим выполнения
        if self.mode == 'sequential':
            results = await self._run_sequential(
                question, answer, question_type, tech_stack, candidate_level
            )
        else:
            results = await self._run_parallel(
                question, answer, question_type, tech_stack, candidate_level
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
            correctness=correctness,
            code_quality=code_quality,
            conceptual=conceptual,
            relevance=relevance,
            level_alignment=level_alignment,
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
    )
