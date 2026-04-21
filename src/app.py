"""
Streamlit приложение для AI-собеседований.

Функции:
- Прохождение собеседования (интерфейс кандидата)
- Просмотр результатов (интерфейс HR/интервьюера)
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import nest_asyncio
import streamlit as st

# Разрешаем вложенные event loops (нужно для Streamlit + asyncio).
# В некоторых окружениях uvloop используется как policy, а nest_asyncio не умеет
# патчить uvloop.Loop — в этом случае мягко переключаемся на стандартный asyncio loop.
try:
    policy_module = asyncio.get_event_loop_policy().__class__.__module__
    if policy_module.startswith('uvloop'):
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    nest_asyncio.apply()
except ValueError as e:
    # Не блокируем запуск приложения: fallback на обычный event loop.
    # run_async ниже корректно обработает выполнение корутин.
    print(f'Предупреждение: nest_asyncio не применён ({e}). Используется fallback.')

# Настройка страницы
st.set_page_config(
    page_title='AI Собеседование',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Импорты из нашего проекта
from main import InterviewGenerator, InterviewParams, INTERVIEW_PARAMS
from agents import AssessmentCoordinator, assessment_result_to_dict, dict_to_assessment_result
from parse_hh import parse_vacancy


# ==============================
# Константы и пути
# ==============================

DATA_DIR = Path(__file__).parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

INTERVIEWS_DIR = DATA_DIR / 'interviews'
INTERVIEWS_DIR.mkdir(exist_ok=True)

RESULTS_DIR = DATA_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# ==============================
# Вспомогательные функции
# ==============================

def run_async(coro):
    """Запустить async функцию в синхронном контексте."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError('Event loop is closed')
        return loop.run_until_complete(coro)
    except RuntimeError:
        # Фолбэк для случаев, когда текущий loop отсутствует/закрыт.
        return asyncio.run(coro)


def save_interview(interview_id: str, data: dict):
    """Сохранить данные интервью."""
    filepath = INTERVIEWS_DIR / f'{interview_id}.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_interview(interview_id: str) -> dict | None:
    """Загрузить данные интервью."""
    filepath = INTERVIEWS_DIR / f'{interview_id}.json'
    if filepath.exists():
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
    return None


def list_interviews() -> list[dict]:
    """Получить список всех интервью."""
    interviews = []
    for filepath in INTERVIEWS_DIR.glob('*.json'):
        try:
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)
                interviews.append({
                    'id': filepath.stem,
                    'candidate_name': data.get('candidate_name', 'Без имени'),
                    'position': data.get('params', {}).get('position', 'Не указано'),
                    'date': data.get('date', 'Не указано'),
                    'status': data.get('status', 'unknown'),
                    'total_score': data.get('total_score'),
                })
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(interviews, key=lambda x: x.get('date', ''), reverse=True)


def generate_interview_id() -> str:
    """Сгенерировать ID интервью."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


# ==============================
# Стили
# ==============================

st.markdown('''
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .question-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .score-high { background-color: #28a745; color: white !important; }
    .score-medium { background-color: #ffc107; color: #212529 !important; }
    .score-low { background-color: #dc3545; color: white !important; }
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        color: #212529;
    }
    .result-card h2, .result-card p {
        color: #212529;
    }
    .agent-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #667eea;
        color: #212529;
    }
    .agent-card strong {
        color: #333;
    }
    .agent-card small {
        color: #555;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
''', unsafe_allow_html=True)


# ==============================
# Сайдбар навигация
# ==============================

def render_sidebar():
    """Отрисовать сайдбар."""
    with st.sidebar:
        st.title('🤖 AI Собеседование')
        st.markdown('---')
        
        page = st.radio(
            'Навигация',
            ['🎯 Новое собеседование', '📋 Результаты', '⚙️ Настройки'],
            index=0,
        )
        
        st.markdown('---')
        st.markdown('''
        **О системе:**
        
        Мультиагентная система для проведения
        технических собеседований с использованием AI.
        
        - Генерация вопросов по вакансии
        - Оценка ответов несколькими агентами
        - Детальная обратная связь
        ''')
        
        return page


# ==============================
# Страница нового собеседования
# ==============================

def render_new_interview():
    """Страница создания и прохождения нового собеседования."""
    st.markdown('<h1 class="main-header">🎯 Новое собеседование</h1>', unsafe_allow_html=True)
    
    # Инициализация состояния
    if 'interview_stage' not in st.session_state:
        st.session_state.interview_stage = 'setup'
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'answers' not in st.session_state:
        st.session_state.answers = []
    if 'interview_params' not in st.session_state:
        st.session_state.interview_params = None
    if 'interview_id' not in st.session_state:
        st.session_state.interview_id = None
    
    # Этап настройки
    if st.session_state.interview_stage == 'setup':
        render_setup_stage()
    
    # Этап генерации вопросов
    elif st.session_state.interview_stage == 'generating':
        render_generating_stage()
    
    # Этап прохождения
    elif st.session_state.interview_stage == 'interview':
        render_interview_stage()
    
    # Этап оценки
    elif st.session_state.interview_stage == 'evaluating':
        render_evaluating_stage()
    
    # Этап результатов
    elif st.session_state.interview_stage == 'completed':
        render_completed_stage()


def render_setup_stage():
    """Этап настройки собеседования."""
    st.subheader('Настройка собеседования')
    
    # Табы для разных способов настройки
    tab1, tab2 = st.tabs(['📝 Ввести параметры вручную', '🔗 Загрузить с hh.ru'])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            candidate_name = st.text_input('Имя кандидата', key='candidate_name_manual')
            position = st.text_input('Позиция', value='Backend-разработчик')
            tech_stack = st.text_input('Технологии', value='Python, FastAPI, PostgreSQL')
            
        with col2:
            level = st.selectbox('Уровень', ['Junior', 'Middle', 'Senior'], index=1)
            num_questions = st.slider('Количество вопросов', 3, 10, 5)
            time_limit = st.slider('Время на интервью (мин)', 15, 90, 45)
        
        topics = st.multiselect(
            'Темы',
            ['API', 'Базы данных', 'Асинхронность', 'Docker', 'Тестирование',
             'CI/CD', 'Микросервисы', 'Безопасность', 'Архитектура', 'ООП', 'Алгоритмы'],
            default=['API', 'Базы данных', 'Асинхронность'],
        )
        
        if st.button('🚀 Начать собеседование', type='primary', key='start_manual'):
            if not candidate_name:
                st.error('Введите имя кандидата')
                return
            
            st.session_state.interview_params = {
                'position': position,
                'tech_stack': tech_stack,
                'level': level,
                'topics': topics,
                'time_limit': time_limit,
                'num_questions': num_questions,
                'company': '',
                'description': '',
            }
            st.session_state.candidate_name = candidate_name
            st.session_state.interview_id = generate_interview_id()
            st.session_state.interview_stage = 'generating'
            st.rerun()
    
    with tab2:
        candidate_name_hh = st.text_input('Имя кандидата', key='candidate_name_hh')
        hh_url = st.text_input('Ссылка на вакансию hh.ru', 
                               placeholder='https://hh.ru/vacancy/123456')
        
        col1, col2 = st.columns(2)
        with col1:
            num_questions_hh = st.slider('Количество вопросов', 3, 10, 5, key='num_q_hh')
        with col2:
            time_limit_hh = st.slider('Время на интервью (мин)', 15, 90, 45, key='time_hh')
        
        if st.button('🚀 Загрузить и начать', type='primary', key='start_hh'):
            if not candidate_name_hh:
                st.error('Введите имя кандидата')
                return
            if not hh_url:
                st.error('Введите ссылку на вакансию')
                return
            
            with st.spinner('Загрузка вакансии с hh.ru...'):
                try:
                    vacancy = parse_vacancy(hh_url)
                    params = InterviewParams.from_vacancy(
                        vacancy, 
                        num_questions=num_questions_hh,
                        time_limit=time_limit_hh
                    )
                    
                    st.session_state.interview_params = {
                        'position': params.position,
                        'tech_stack': params.tech_stack,
                        'level': params.level,
                        'topics': params.topics,
                        'time_limit': params.time_limit,
                        'num_questions': params.num_questions,
                        'company': params.company,
                        'description': params.description,
                    }
                    st.session_state.candidate_name = candidate_name_hh
                    st.session_state.interview_id = generate_interview_id()
                    st.session_state.interview_stage = 'generating'
                    st.rerun()
                    
                except Exception as e:
                    st.error(f'Ошибка загрузки вакансии: {e}')


def render_generating_stage():
    """Этап генерации вопросов."""
    st.subheader('⏳ Генерация вопросов...')
    
    params = st.session_state.interview_params
    
    st.info(f'''
    **Позиция:** {params["position"]}  
    **Уровень:** {params["level"]}  
    **Технологии:** {params["tech_stack"]}  
    **Количество вопросов:** {params["num_questions"]}
    ''')
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        generator = InterviewGenerator(params)
        
        status_text.text('Инициализация генератора...')
        progress_bar.progress(10)
        
        status_text.text('Генерация вопросов... Это может занять 1-2 минуты.')
        questions = run_async(generator.generate_questions())
        
        progress_bar.progress(100)
        status_text.text(f'Сгенерировано {len(questions)} вопросов!')
        
        st.session_state.questions = questions
        st.session_state.current_question = 0
        st.session_state.answers = [None] * len(questions)
        st.session_state.interview_stage = 'interview'
        
        # Сохраняем промежуточные данные
        save_interview(st.session_state.interview_id, {
            'candidate_name': st.session_state.candidate_name,
            'params': params,
            'questions': questions,
            'status': 'in_progress',
            'date': datetime.now().isoformat(),
        })
        
        st.rerun()
        
    except Exception as e:
        st.error(f'Ошибка генерации: {e}')
        if st.button('↩️ Вернуться к настройкам'):
            st.session_state.interview_stage = 'setup'
            st.rerun()


def render_interview_stage():
    """Этап прохождения собеседования."""
    questions = st.session_state.questions
    current = st.session_state.current_question
    total = len(questions)
    
    # Прогресс
    progress = (current) / total
    st.progress(progress)
    st.markdown(f'**Вопрос {current + 1} из {total}**')
    
    # Текущий вопрос
    question = questions[current]
    
    st.markdown(f'''
    <div class="question-card">
        <h3>❓ {question.get("question", "Вопрос не загружен")}</h3>
        <p><strong>Тип:</strong> {question.get("type", "N/A")} | 
        <strong>Время:</strong> ~{question.get("expected_time_min", 5)} мин</p>
        <p><strong>Теги:</strong> {", ".join(question.get("tags", []))}</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Поле для ответа
    current_answer = st.session_state.answers[current] or ''
    answer = st.text_area(
        'Ваш ответ:',
        value=current_answer,
        height=300,
        placeholder='Введите ваш ответ здесь...',
        key=f'answer_{current}',
    )
    
    # Сохраняем ответ
    st.session_state.answers[current] = answer
    
    # Follow-up вопросы (подсказка)
    with st.expander('💡 Дополнительные вопросы (follow-up)'):
        follow_ups = question.get('follow_ups', [])
        for i, fu in enumerate(follow_ups, 1):
            st.markdown(f'{i}. {fu}')
    
    # Навигация
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if current > 0:
            if st.button('⬅️ Назад'):
                st.session_state.current_question = current - 1
                st.rerun()
    
    with col2:
        # Показываем статус ответов
        answered = sum(1 for a in st.session_state.answers if a)
        st.markdown(f'Отвечено: {answered}/{total}')
    
    with col3:
        if current < total - 1:
            if st.button('Далее ➡️'):
                st.session_state.current_question = current + 1
                st.rerun()
        else:
            if st.button('✅ Завершить', type='primary'):
                # Проверяем, что все вопросы отвечены
                unanswered = [i + 1 for i, a in enumerate(st.session_state.answers) if not a]
                if unanswered:
                    st.warning(f'Не отвечены вопросы: {", ".join(map(str, unanswered))}')
                else:
                    st.session_state.interview_stage = 'evaluating'
                    st.rerun()
    
    # Кнопка принудительного завершения
    st.markdown('---')
    if st.button('⏭️ Завершить досрочно'):
        st.session_state.interview_stage = 'evaluating'
        st.rerun()


def render_evaluating_stage():
    """Этап оценки ответов."""
    st.subheader('🔍 Оценка ответов...')
    
    questions = st.session_state.questions
    answers = st.session_state.answers
    params = st.session_state.interview_params
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_placeholder = st.empty()
    
    coordinator = AssessmentCoordinator()
    assessments = []
    
    for i, (q, a) in enumerate(zip(questions, answers)):
        if not a:
            assessments.append(None)
            continue
        
        status_text.text(f'Оценка вопроса {i + 1} из {len(questions)}...')
        progress_bar.progress((i + 1) / len(questions))
        
        try:
            result = run_async(coordinator.assess_answer(
                question=q.get('question', ''),
                answer=a,
                question_type=q.get('type', 'теория'),
                tech_stack=params.get('tech_stack', ''),
                candidate_level=params.get('level', 'Middle'),
                question_tags=q.get('tags', []),
                follow_ups=q.get('follow_ups', []),
            ))
            assessments.append(assessment_result_to_dict(result))
        except Exception as e:
            st.error(f'Ошибка оценки вопроса {i + 1}: {e}')
            assessments.append(None)
    
    # Вычисляем общую оценку
    valid_scores = [a['total_score'] for a in assessments if a]
    total_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    
    st.session_state.assessments = assessments
    st.session_state.total_score = total_score
    
    # Сохраняем результаты
    save_interview(st.session_state.interview_id, {
        'candidate_name': st.session_state.candidate_name,
        'params': params,
        'questions': questions,
        'answers': answers,
        'assessments': assessments,
        'total_score': total_score,
        'status': 'completed',
        'date': datetime.now().isoformat(),
    })
    
    status_text.text('Оценка завершена!')
    progress_bar.progress(100)
    
    st.session_state.interview_stage = 'completed'
    st.rerun()


def render_completed_stage():
    """Этап отображения результатов."""
    st.subheader('🎉 Собеседование завершено!')
    
    total_score = st.session_state.total_score
    
    # Определяем цвет и текст рекомендации
    if total_score >= 4.5:
        score_class = 'score-high'
        recommendation = '✅ Сильный кандидат'
    elif total_score >= 3.5:
        score_class = 'score-medium'
        recommendation = '👍 Хороший кандидат'
    elif total_score >= 2.5:
        score_class = 'score-medium'
        recommendation = '⚠️ Требует развития'
    else:
        score_class = 'score-low'
        recommendation = '❌ Не соответствует уровню'
    
    # Общая оценка
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f'''
        <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; color: #212529;">
            <h2 style="color: #212529;">Общая оценка</h2>
            <div class="score-badge {score_class}" style="font-size: 2rem;">
                {total_score:.1f}/5
            </div>
            <p style="margin-top: 1rem; color: #212529;">{recommendation}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        **Кандидат:** {st.session_state.candidate_name}  
        **Позиция:** {st.session_state.interview_params.get("position")}  
        **Уровень:** {st.session_state.interview_params.get("level")}  
        **Отвечено вопросов:** {sum(1 for a in st.session_state.answers if a)}/{len(st.session_state.questions)}
        ''')
    
    st.markdown('---')
    
    # Детали по каждому вопросу
    st.subheader('📊 Детальные результаты')
    
    for i, (q, assessment) in enumerate(zip(st.session_state.questions, st.session_state.assessments)):
        if not assessment:
            continue
        
        with st.expander(f'Вопрос {i + 1}: {q.get("question", "")[:50]}...'):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                score = assessment['total_score']
                if score >= 4.5:
                    st.success(f'Оценка: {score:.1f}/5')
                elif score >= 3.5:
                    st.info(f'Оценка: {score:.1f}/5')
                elif score >= 2.5:
                    st.warning(f'Оценка: {score:.1f}/5')
                else:
                    st.error(f'Оценка: {score:.1f}/5')
            
            with col2:
                st.markdown(f"**{assessment['final_feedback']}**")
            
            # Ответ кандидата
            st.markdown('**Ответ кандидата:**')
            st.text(assessment['answer'][:500] + '...' if len(assessment['answer']) > 500 else assessment['answer'])
            
            # Оценки агентов
            error_count = assessment.get('agents_error', 0)
            success_count = assessment.get('agents_success', len(assessment.get('agent_scores', [])))
            
            if error_count > 0:
                st.markdown(f'**Оценки экспертов:** ({success_count} успешно, {error_count} с ошибкой)')
            else:
                st.markdown('**Оценки экспертов:**')
            
            for agent_score in assessment.get('agent_scores', []):
                is_error = agent_score.get('error', False)
                
                if is_error:
                    # Агент с ошибкой - показываем красным
                    st.markdown(f'''
                    <div class="agent-card" style="border-left-color: #dc3545; background: #fff5f5;">
                        <strong style="color: #dc3545;">⚠️ {agent_score["agent_name"]}</strong>: НЕ УЧТЁН
                        <br><small style="color: #721c24;">{agent_score["feedback"]}</small>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    # Успешный агент
                    st.markdown(f'''
                    <div class="agent-card">
                        <strong>{agent_score["agent_name"]}</strong>: {agent_score["score"]:.1f}/5
                        <br><small>{agent_score["feedback"]}</small>
                    </div>
                    ''', unsafe_allow_html=True)
    
    # Кнопки действий
    st.markdown('---')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button('🔄 Новое собеседование'):
            # Сбрасываем состояние
            for key in ['interview_stage', 'questions', 'current_question', 
                        'answers', 'interview_params', 'interview_id', 
                        'assessments', 'total_score', 'candidate_name']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button('📥 Скачать отчёт'):
            report = {
                'interview_id': st.session_state.interview_id,
                'candidate_name': st.session_state.candidate_name,
                'params': st.session_state.interview_params,
                'total_score': total_score,
                'recommendation': recommendation,
                'details': st.session_state.assessments,
            }
            st.download_button(
                label='💾 Скачать JSON',
                data=json.dumps(report, ensure_ascii=False, indent=2),
                file_name=f'interview_{st.session_state.interview_id}.json',
                mime='application/json',
            )


# ==============================
# Страница результатов
# ==============================

def render_results():
    """Страница просмотра результатов."""
    st.markdown('<h1 class="main-header">📋 Результаты собеседований</h1>', unsafe_allow_html=True)
    
    interviews = list_interviews()
    
    if not interviews:
        st.info('Нет сохранённых собеседований.')
        return
    
    # Фильтры
    col1, col2 = st.columns(2)
    
    with col1:
        status_filter = st.selectbox(
            'Статус',
            ['Все', 'completed', 'in_progress'],
            format_func=lambda x: {
                'Все': 'Все',
                'completed': '✅ Завершённые',
                'in_progress': '⏳ В процессе',
            }.get(x, x)
        )
    
    with col2:
        search = st.text_input('🔍 Поиск по имени', '')
    
    # Фильтруем
    filtered = interviews
    if status_filter != 'Все':
        filtered = [i for i in filtered if i['status'] == status_filter]
    if search:
        filtered = [i for i in filtered if search.lower() in i['candidate_name'].lower()]
    
    # Отображаем
    st.markdown(f'Найдено: {len(filtered)} собеседований')
    
    for interview in filtered:
        with st.expander(f"{interview['candidate_name']} — {interview['position']} ({interview['date'][:10]})"):
            
            # Загружаем полные данные
            full_data = load_interview(interview['id'])
            
            if not full_data:
                st.error('Не удалось загрузить данные')
                continue
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric('Общая оценка', f"{full_data.get('total_score', 0):.1f}/5")
            
            with col2:
                st.metric('Статус', '✅ Завершено' if full_data['status'] == 'completed' else '⏳ В процессе')
            
            with col3:
                params = full_data.get('params', {})
                st.metric('Уровень', params.get('level', 'N/A'))
            
            # Детали по вопросам
            if full_data.get('assessments'):
                st.markdown('---')
                st.markdown('**Результаты по вопросам:**')
                
                for i, (q, a, assessment) in enumerate(zip(
                    full_data.get('questions', []),
                    full_data.get('answers', []),
                    full_data.get('assessments', [])
                )):
                    if not assessment:
                        continue
                    
                    score = assessment['total_score']
                    color = 'green' if score >= 4.5 else ('orange' if score >= 2.5 else 'red')
                    
                    st.markdown(f'''
                    **{i + 1}. {q.get("question", "")[:60]}...**  
                    Оценка: :{color}[{score:.1f}/5] — {assessment.get("final_feedback", "")}
                    ''')
            
            # Кнопки
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f'📥 Скачать отчёт', key=f'download_{interview["id"]}'):
                    st.download_button(
                        label='💾 JSON',
                        data=json.dumps(full_data, ensure_ascii=False, indent=2),
                        file_name=f'interview_{interview["id"]}.json',
                        mime='application/json',
                        key=f'dl_btn_{interview["id"]}',
                    )
            
            with col2:
                if st.button(f'🗑️ Удалить', key=f'delete_{interview["id"]}'):
                    filepath = INTERVIEWS_DIR / f'{interview["id"]}.json'
                    if filepath.exists():
                        filepath.unlink()
                        st.success('Удалено!')
                        st.rerun()


# ==============================
# Страница настроек
# ==============================

def render_settings():
    """Страница настроек."""
    st.markdown('<h1 class="main-header">⚙️ Настройки</h1>', unsafe_allow_html=True)
    
    st.subheader('LLM и RAG настройки')
    
    st.info('''
    Настройки хранятся в файле `.env` в корне проекта.
    
    Ключевые переменные:
    - `LLM_PROVIDER` — `ollama` или `gigachat`
    - `OLLAMA_CHAT_MODEL` — модель генерации/оценки
    - `OLLAMA_EMBED_MODEL` — модель эмбеддингов для RAG
    - `RAG_DB_PATH` — путь к локальной векторной базе
    ''')
    
    # Проверка настроек
    st.subheader('Проверка конфигурации')
    
    import settings as s
    
    col1, col2 = st.columns(2)
    
    with col1:
        if s.LLM_PROVIDER == 'ollama':
            st.success(f'✅ LLM_PROVIDER: {s.LLM_PROVIDER}')
        elif s.MODEL_API_KEY:
            st.success('✅ MODEL_API_KEY настроен')
        else:
            st.error('❌ Нет настроек для выбранного LLM_PROVIDER')
    
    with col2:
        if s.LLM_PROVIDER == 'ollama':
            st.success(f'✅ OLLAMA_CHAT_MODEL: {s.OLLAMA_CHAT_MODEL}')
        elif s.MODEL_NAME:
            st.success(f'✅ MODEL_NAME: {s.MODEL_NAME}')
        else:
            st.warning('⚠️ Имя модели не указано')

    rag_path = Path(s.RAG_DB_PATH)
    if rag_path.exists():
        st.success(f'✅ RAG база найдена: {s.RAG_DB_PATH}')
    else:
        st.warning(
            f'⚠️ RAG база не найдена: {s.RAG_DB_PATH}. '
            'Выполните: uv run python -m src.rag.build_index'
        )
    
    st.markdown('---')
    
    st.subheader('Статистика')
    
    interviews = list_interviews()
    completed = [i for i in interviews if i['status'] == 'completed']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric('Всего собеседований', len(interviews))
    
    with col2:
        st.metric('Завершённых', len(completed))
    
    with col3:
        if completed:
            avg_score = sum(i.get('total_score', 0) or 0 for i in completed) / len(completed)
            st.metric('Средняя оценка', f'{avg_score:.1f}')
        else:
            st.metric('Средняя оценка', 'N/A')


# ==============================
# Главная функция
# ==============================

def main():
    """Главная функция приложения."""
    page = render_sidebar()
    
    if page == '🎯 Новое собеседование':
        render_new_interview()
    elif page == '📋 Результаты':
        render_results()
    elif page == '⚙️ Настройки':
        render_settings()


if __name__ == '__main__':
    main()
