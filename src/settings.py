import os

from dotenv import load_dotenv


load_dotenv()

# Настройки модели
MODEL_API_KEY = os.getenv('MODEL_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_BASE_URL = os.getenv('MODEL_BASE_URL')

# Провайдер LLM: 'ollama' (локально) или 'gigachat'
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'ollama')

# Ollama
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_CHAT_MODEL = os.getenv('OLLAMA_CHAT_MODEL', 'qwen2.5:14b-instruct')
OLLAMA_EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')

# Настройки режима работы API
# Режим выполнения запросов: 'parallel' (параллельно) или 'sequential' (последовательно)
# Используйте 'sequential' если получаете ошибку 429 (Too Many Requests)
API_MODE = os.getenv('API_MODE', 'parallel')

# Задержка между запросами в секундах (только для sequential режима)
API_REQUEST_DELAY = float(os.getenv('API_REQUEST_DELAY', '1.0'))

# RAG
RAG_SOURCE_JSON = os.getenv('RAG_SOURCE_JSON', 'questions_with_tags_and_answers.json')
RAG_DB_PATH = os.getenv('RAG_DB_PATH', 'data/rag/chroma')
RAG_COLLECTION_NAME = os.getenv('RAG_COLLECTION_NAME', 'python_interview_rag')
RAG_TOP_K_GENERATION = int(os.getenv('RAG_TOP_K_GENERATION', '5'))
RAG_TOP_K_EVAL = int(os.getenv('RAG_TOP_K_EVAL', '5'))
RAG_REFERENCE_SNIPPET_CHARS = int(os.getenv('RAG_REFERENCE_SNIPPET_CHARS', '700'))
RAG_CHUNK_SIZE = int(os.getenv('RAG_CHUNK_SIZE', '1200'))
RAG_CHUNK_OVERLAP = int(os.getenv('RAG_CHUNK_OVERLAP', '120'))
RAG_MAX_ANSWERS_PER_QUESTION = int(os.getenv('RAG_MAX_ANSWERS_PER_QUESTION', '3'))
RAG_MIN_ANSWER_CHARS = int(os.getenv('RAG_MIN_ANSWER_CHARS', '60'))
RAG_STRICT_EXACT_GATE = os.getenv('RAG_STRICT_EXACT_GATE', '1') == '1'
RAG_HYBRID_LEXICAL_BOOST = os.getenv('RAG_HYBRID_LEXICAL_BOOST', '1') == '1'

# Логи генерации вопросов (design/generate/validate/format)
GENERATION_DEBUG_LOGS = os.getenv('GENERATION_DEBUG_LOGS', '1') == '1'
ASSESSMENT_DEBUG_LOGS = os.getenv('ASSESSMENT_DEBUG_LOGS', '1') == '1'
DEBUG_LOG_TO_FILE = os.getenv('DEBUG_LOG_TO_FILE', '1') == '1'
DEBUG_LOG_FILE = os.getenv('DEBUG_LOG_FILE', 'data/logs/debug.log')
DEBUG_CONSOLE_PREVIEW_LIMIT = int(os.getenv('DEBUG_CONSOLE_PREVIEW_LIMIT', '240'))
