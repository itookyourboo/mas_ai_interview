import os

from dotenv import load_dotenv


load_dotenv()

MODEL_API_KEY = os.getenv('MODEL_API_KEY'),
MODEL_NAME = os.getenv('MODEL_NAME'),
MODEL_BASE_URL = os.getenv('MODEL_BASE_URL')
