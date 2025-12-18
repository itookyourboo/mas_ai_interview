import os

from dotenv import load_dotenv


load_dotenv()

# Настройки модели
MODEL_API_KEY = os.getenv('MODEL_API_KEY')
MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_BASE_URL = os.getenv('MODEL_BASE_URL')

# Настройки режима работы API
# Режим выполнения запросов: 'parallel' (параллельно) или 'sequential' (последовательно)
# Используйте 'sequential' если получаете ошибку 429 (Too Many Requests)
API_MODE = os.getenv('API_MODE', 'parallel')

# Задержка между запросами в секундах (только для sequential режима)
API_REQUEST_DELAY = float(os.getenv('API_REQUEST_DELAY', '1.0'))
