# config.py
# Глобальные настройки приложения
# Все чувствительные данные загружаются из переменных окружения

import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла (если существует)
load_dotenv()

class Config:
    """
    Конфигурация приложения.
    
    Использует паттерн "класс-конфиг" для удобства импорта
    и поддержки разных окружений (dev/prod).
    """
    
    # === БАЗОВЫЕ НАСТРОЙКИ ===
    
    # Режим отладки (в продакшене должно быть False!)
    DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    # Секретный ключ для сессий и токенов
    # В продакшене генерировать: python -c "import secrets; print(secrets.token_hex(32))"
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-shrimp-key-change-in-prod!')
    
    # Адрес и порт сервера
    SERVER_HOST = os.getenv('SERVER_HOST', '127.0.0.1')
    SERVER_PORT = int(os.getenv('SERVER_PORT', '5000'))
    
    # === БАЗА ДАННЫХ ===
    
    # Путь к SQLite базе данных пользователей
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/users.db')
    
    # Создаём директорию для БД если не существует
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    # === ПУТИ К ДАННЫМ ===
    
    # Пути к наборам данных (скачиваются с Яндекс.Диска)
    TRAIN_DATA_PATH = os.getenv('TRAIN_DATA_PATH', 'data/train.npz')
    VALID_DATA_PATH = os.getenv('VALID_DATA_PATH', 'data/valid.npz')
    TEST_DATA_PATH = os.getenv('TEST_DATA_PATH', 'data/test.npz')
    
    # Пароль для тестового архива (выдаётся жюри)
    TEST_DATA_PASSWORD = os.getenv('TEST_DATA_PASSWORD', '')
    
    # Путь для сохранения обученной модели
    MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', 'models/alien_model.pkl')
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Временная директория для загруженных файлов
    TEMP_DIR = os.getenv('TEMP_DIR', 'temp')
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # === НАСТРОЙКИ ОБУЧЕНИЯ МОДЕЛИ ===
    
    TRAIN_CONFIG = {
        'epochs': int(os.getenv('TRAIN_EPOCHS', '50')),
        'batch_size': int(os.getenv('TRAIN_BATCH_SIZE', '32')),
        'learning_rate': float(os.getenv('TRAIN_LR', '0.001')),
        'validation_split': 0.25,  # 400 из 1600 записей
        'early_stopping_patience': 5,  # остановка если нет улучшений
        'save_best_only': True  # сохранять только лучшую модель
    }
    
    # === НАСТРОЙКИ DATASPHERE (опционально) ===
    
    # Использовать ли инференс из облака вместо локальной модели
    USE_DATASPHERE_INFERENCE = os.getenv('USE_DATASPHERE', 'false').lower() == 'true'
    
    # Параметры для подключения к деплоенной модели
    DATASPHERE_NODE_ID = os.getenv('DATASPHERE_NODE_ID', '')
    DATASPHERE_FOLDER_ID = os.getenv('DATASPHERE_FOLDER_ID', '')
    DATASPHERE_IAM_TOKEN = os.getenv('DATASPHERE_IAM_TOKEN', '')
    
    # Таймаут для облачных запросов (секунды)
    DATASPHERE_TIMEOUT = int(os.getenv('DATASPHERE_TIMEOUT', '60'))
    
    # === НАСТРОЙКИ БЕЗОПАСНОСТИ ===
    
    # Время жизни токена сессии в секундах (по умолчанию 24 часа)
    TOKEN_EXPIRY_SECONDS = int(os.getenv('TOKEN_EXPIRY', '86400'))
    
    # Минимальная длина пароля
    MIN_PASSWORD_LENGTH = int(os.getenv('MIN_PASSWORD_LENGTH', '8'))
    
    # === НАСТРОЙКИ ЛОГИРОВАНИЯ ===
    
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)