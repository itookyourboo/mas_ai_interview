import asyncio
import json
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat import GigaChat
from langgraph.graph import END, StateGraph

import settings


llm = GigaChat(
    credentials=settings.MODEL_API_KEY,
    verify_ssl_certs=False,
    model=settings.MODEL_NAME,
    temperature=0.3,
)

# Параметры интервью
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
# Промпты на русском
# ==============================

# 1. Планирование
design_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'Вы — архитектор собеседований. Создайте план одного вопроса.'),
        (
            'human',
            'Параметры:\nПозиция: {position}\nТехнологии: {tech_stack}\nУровень: {level}\n'
            'Темы: {topics}\nВремя интервью: {time_limit} мин.\n'
            'Создайте план для вопроса №{index}. Укажите тип, сложность, концепции и время на ответ.',
        ),
    ],
)

# 2. Генерация
generate_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'Вы — генератор вопросов. Создайте чёткий технический вопрос на русском.'),
        ('human', 'План вопроса:\n{plan}\nСоздайте вопрос на русском языке.'),
    ],
)

# 3. Валидация
validate_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'Вы — валидатор. Проверьте вопрос.'),
        (
            'human',
            'Проверьте вопрос на русском языке:\n{question}\n'
            'Оцените: ясность, соответствие уровню Middle, релевантность стеку Python/FastAPI, '
            'отсутствие предвзятости и уникальность. '
            'Если всё в порядке — напишите "ОДОБРЕНО". Иначе — "ОТКЛОНЁН: [причина]".',
        ),
    ],
)

# 4. Форматирование
format_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'Вы — форматировщик. Выведите строго JSON.'),
        (
            'human',
            'Преобразуйте вопрос в JSON со следующими полями:\n'
            '- question: текст вопроса на русском\n'
            '- type: "код", "системный дизайн", "теория", "отладка"\n'
            '- level: "Middle"\n'
            '- tags: список тем на русском\n'
            '- expected_time_min: целое число\n'
            '- follow_ups: список из 1–3 follow-up на русском\n\n'
            'Вопрос:\n{validated_question}\n\n'
            'Выведите ТОЛЬКО JSON, без пояснений.',
        ),
    ],
)


# ==============================
# Узлы графа
# ==============================

async def design_node(state: QuestionState) -> dict:
    chain = design_prompt | llm
    response = await chain.ainvoke(
        {
            'position': INTERVIEW_PARAMS['position'],
            'tech_stack': INTERVIEW_PARAMS['tech_stack'],
            'level': INTERVIEW_PARAMS['level'],
            'topics': ', '.join(INTERVIEW_PARAMS['topics']),
            'time_limit': INTERVIEW_PARAMS['time_limit'],
            'index': state['index'] + 1,
        },
    )
    return {'plan': response.content}


async def generate_node(state: QuestionState) -> dict:
    chain = generate_prompt | llm
    response = await chain.ainvoke({'plan': state['plan']})
    return {'question': response.content}


async def validate_node(state: QuestionState) -> dict:
    chain = validate_prompt | llm
    response = await chain.ainvoke({'question': state['question']})
    validated = response.content.strip()
    return {'validated': validated}


async def format_node(state: QuestionState) -> dict:
    if not state['validated'].startswith('ОДОБРЕНО'):
        return {'formatted': None}
    chain = format_prompt | llm
    try:
        response = await chain.ainvoke({'validated_question': state['question']})
        # Принудительно парсим JSON
        formatted = json.loads(response.content.strip())
        return {'formatted': formatted}
    except Exception as e:
        return {'formatted': None}


# Обработка одного вопроса (подграф)
def create_question_graph() -> StateGraph:
    graph = StateGraph(QuestionState)
    graph.add_node('design', design_node)
    graph.add_node('generate', generate_node)
    graph.add_node('validate', validate_node)
    graph.add_node('format', format_node)

    graph.set_entry_point('design')
    graph.add_edge('design', 'generate')
    graph.add_edge('generate', 'validate')
    graph.add_edge('validate', 'format')
    graph.add_edge('format', END)

    return graph.compile()


# ==============================
# Основной граф (управление N вопросами)
# ==============================

async def generate_question(
    index: int,
    question_graph: StateGraph,
    retries: int = 3,
) -> dict | None:
    print(f'Генерация вопроса №{index}')
    initial_state: QuestionState = {
        'index': index,
        'plan': '',
        'question': '',
        'validated': '',
        'formatted': None,
        'attempts': 0,
    }

    for i in range(1, retries + 1):
        result = await question_graph.ainvoke(initial_state)

        if result['formatted']:
            print(f'✅ Вопрос №{index} принят')
            return result['formatted']

        print(f'❌ Вопрос №{index} отклонён или не распарсен с попытки №{i}')

    print(f'❌ Вопрос №{index} отклонён после {retries} попыток')
    return None


async def run_interview_generator() -> list[dict]:
    question_graph = create_question_graph()
    completed = []
    index = 0

    while len(completed) < INTERVIEW_PARAMS['num_questions'] and index < INTERVIEW_PARAMS['num_questions'] * 3:
        print(f'Генерация вопроса #{len(completed) + 1} (попытка {index + 1})...')

        initial_state: QuestionState = {
            'index': len(completed),
            'plan': '',
            'question': '',
            'validated': '',
            'formatted': None,
            'attempts': 0,
        }

        result = await question_graph.ainvoke(initial_state)

        if result['formatted']:
            completed.append(result['formatted'])
            print('✅ Вопрос принят.')
        else:
            print('❌ Вопрос отклонён или не распарсен. Повтор...')

        index += 1

    return completed


# ==============================
# Запуск
# ==============================

async def main():
    print('Запуск генерации вопросов на LangGraph...')
    questions = await run_interview_generator()

    output = {'questions': questions}

    filename = 'interview_questions_langgraph.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'\nСгенерировано {len(questions)} вопросов.')
    print(f'Результат: {filename}')


if __name__ == '__main__':
    asyncio.run(main())
