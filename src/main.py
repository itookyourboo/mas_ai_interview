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
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat import GigaChat
from langgraph.graph import END, StateGraph

import settings
from parse_hh import VacancyInfo, parse_vacancy


# ==============================
# Инициализация LLM
# ==============================

def get_llm():
    """Получить инстанс LLM."""
    return GigaChat(
        credentials=settings.MODEL_API_KEY,
        verify_ssl_certs=False,
        model=settings.MODEL_NAME,
        temperature=0.3,
        scope='GIGACHAT_API_PERS',
    )


llm = get_llm()


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
   - Вопросов с несколькими частями'''),
    ('human', '''План вопроса:
{plan}

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

Если всё в порядке — напишите "ОДОБРЕНО".
Если есть проблемы — напишите "ОТКЛОНЁН: [подробная причина]".'''),
    ('human', '''Проверьте вопрос для уровня {level}:

{question}

Технологии: {tech_stack}'''),
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

async def design_node(state: QuestionState, params: dict = None, delay: float = 0) -> dict:
    """Узел планирования вопроса."""
    params = params or INTERVIEW_PARAMS
    
    # Определяем тип вопроса на основе его номера
    total = params.get('num_questions', 5)
    question_info = get_question_type_for_index(state['index'], total)
    
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
    })
    
    if delay > 0:
        await asyncio.sleep(delay)
    
    return {'plan': response.content}


async def generate_node(state: QuestionState, delay: float = 0) -> dict:
    """Узел генерации вопроса."""
    chain = generate_prompt | llm
    response = await chain.ainvoke({'plan': state['plan']})
    
    if delay > 0:
        await asyncio.sleep(delay)
    
    return {'question': response.content}


async def validate_node(state: QuestionState, params: dict = None, delay: float = 0) -> dict:
    """Узел валидации вопроса."""
    params = params or INTERVIEW_PARAMS
    chain = validate_prompt | llm
    response = await chain.ainvoke({
        'question': state['question'],
        'level': params.get('level', INTERVIEW_PARAMS['level']),
        'tech_stack': params.get('tech_stack', INTERVIEW_PARAMS['tech_stack']),
    })
    validated = response.content.strip()
    
    if delay > 0:
        await asyncio.sleep(delay)
    
    return {'validated': validated}


async def format_node(state: QuestionState, params: dict = None, delay: float = 0) -> dict:
    """Узел форматирования вопроса."""
    params = params or INTERVIEW_PARAMS
    if not state['validated'].startswith('ОДОБРЕНО'):
        return {'formatted': None}
    
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
        return {'formatted': formatted}
    except Exception as e:
        print(f'Ошибка форматирования: {e}')
        return {'formatted': None}


def create_question_graph(params: dict = None, mode: str = None, request_delay: float = None) -> StateGraph:
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
    
    graph = StateGraph(QuestionState)
    
    # Оборачиваем узлы для передачи параметров и задержки
    async def _design(state):
        return await design_node(state, params, delay)
    
    async def _generate(state):
        return await generate_node(state, delay)
    
    async def _validate(state):
        return await validate_node(state, params, delay)
    
    async def _format(state):
        return await format_node(state, params, delay)
    
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
        
        self.question_graph = create_question_graph(
            self.params, 
            mode=self.mode, 
            request_delay=self.request_delay
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
        
        # Задержка между попытками генерации (только в sequential режиме)
        between_questions_delay = self.request_delay if self.mode == 'sequential' else 0

        while len(completed) < num_questions and attempt < max_attempts:
            print(f'Генерация вопроса #{len(completed) + 1} (попытка {attempt + 1})...')

            initial_state: QuestionState = {
                'index': len(completed),
                'plan': '',
                'question': '',
                'validated': '',
                'formatted': None,
                'attempts': 0,
            }

            result = await self.question_graph.ainvoke(initial_state)

            if result['formatted']:
                completed.append(result['formatted'])
                print('V | Вопрос принят.')
            else:
                print('X | Вопрос отклонён или не распарсен. Повтор...')

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
