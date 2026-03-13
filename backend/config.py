"""
Конфигурация проекта Pokémon AI Similarity Search.
Централизованное хранение всех параметров для модулей нейросетей,
базы данных и API.

Этот файл импортируется всеми модулями проекта для получения
единообразных настроек.
"""
import os
from pathlib import Path


# ==================== БАЗОВЫЕ ПУТИ ====================

# Корневая директория проекта (legendarniy_kostil/)
BASE_DIR = Path(__file__).parent.parent

# Директория backend (legendarniy_kostil/backend/)
BACKEND_DIR = Path(__file__).parent

# Директория для моделей нейросетей (backend/models/)
MODELS_DIR = BACKEND_DIR / 'models'

# Директория для фронтенда (legendarniy_kostil/frontend/)
FRONTEND_DIR = BASE_DIR / 'frontend'

# Путь к базе данных SQLite
DATABASE_PATH = BACKEND_DIR / 'pokemon.db'


# ==================== POKEAPI НАСТРОЙКИ ====================

# Базовый URL PokeAPI
POKEAPI_BASE = 'https://pokeapi.co/api/v2'

# Таймаут для запросов к API (секунды)
POKEAPI_TIMEOUT = 10

# Максимальное количество покемонов для загрузки (на 2024 год)
POKEAPI_LIMIT = 1302

# Задержка между пакетами запросов (мс) для уважения к API
POKEAPI_DELAY = 200

# Размер пакета для параллельных запросов
POKEAPI_BATCH_SIZE = 10


# ==================== CNN МОДУЛЬ (Визуальный поиск) ====================

CNN_CONFIG = {
    # Размер входного изображения (высота, ширина, каналы)
    'input_size': (64, 64, 3),
    
    # Размерность векторного представления (latent space)
    'embedding_dim': 32,
    
    # Размер батча при обучении
    'batch_size': 16,
    
    # Количество эпох обучения
    'epochs': 30,
    
    # Скорость обучения (learning rate)
    'learning_rate': 0.001,
    
    # Путь к файлу весов модели
    'model_path': MODELS_DIR / 'cnn' / 'cnn_model.pth',
    
    # Путь к кэшу эмбеддингов (опционально)
    'cache_path': MODELS_DIR / 'cnn' / 'embeddings_cache.pkl',
    
    # Минимальное количество покемонов со спрайтами для обучения
    'min_samples': 200,
    
    # Максимальное количество покемонов для обучения (для скорости)
    'max_samples': 500,
    
    # Коэффициент Dropout для регуляризации
    'dropout': 0.3,
    
    # Температура для контрастивной функции потерь
    'temperature': 0.1,
    
    # Количество результатов для поиска похожих
    'top_k': 5,
}


# ==================== RNN МОДУЛЬ (Классификация силы) ====================

RNN_CONFIG = {
    # Размерность входа на каждом шаге (1 стат за шаг)
    'input_size': 1,
    
    # Размерность скрытого состояния LSTM
    'hidden_dim': 32,
    
    # Количество LSTM слоёв
    'num_layers': 2,
    
    # Количество классов (weak, medium, strong)
    'num_classes': 3,
    
    # Размер батча при обучении
    'batch_size': 32,
    
    # Количество эпох обучения
    'epochs': 30,
    
    # Скорость обучения
    'learning_rate': 0.001,
    
    # Коэффициент Dropout для регуляризации
    'dropout': 0.3,
    
    # Путь к файлу весов модели
    'model_path': MODELS_DIR / 'rnn' / 'rnn_model.pth',
    
    # Минимальное количество покемонов для обучения
    'min_samples': 50,
    
    # Пороги для классификации силы (BST - Base Stat Total)
    'strength_thresholds': {
        'weak': 400,      # BST < 400
        'medium': 550,    # 400 <= BST < 550
        'strong': 9999    # BST >= 550
    },
    
    # Названия классов на русском
    'class_names_ru': {
        'weak': '🔹 Слабый',
        'medium': '🔸 Средний',
        'strong': '🔴 Сильный'
    },
}


# ==================== AUTOENCODER МОДУЛЬ (Обычность) ====================

AE_CONFIG = {
    # Размерность входных данных (8 статов + 18 типов one-hot)
    'input_dim': 26,
    
    # Размерность латентного пространства (bottleneck)
    'latent_dim': 8,
    
    # Размерности скрытых слоёв энкодера
    'hidden_dims': [16, 12],
    
    # Размер батча при обучении
    'batch_size': 32,
    
    # Количество эпох обучения
    'epochs': 30,
    
    # Скорость обучения
    'learning_rate': 0.001,
    
    # Коэффициент Dropout для регуляризации
    'dropout': 0.2,
    
    # Путь к файлу весов модели
    'model_path': MODELS_DIR / 'autoencoder' / 'ae_model.pth',
    
    # Минимальное количество покемонов для обучения
    'min_samples': 50,
    
    # Функция потерь (MSELoss)
    'loss_function': 'mse',
    
    # Уровни обычности для классификации
    'ordinariness_levels': {
        'very_ordinary': 80,   # >= 80%
        'ordinary': 60,        # 60-80%
        'unusual': 40,         # 40-60%
        'rare': 20,            # 20-40%
        'unique': 0            # < 20%
    },
}


# ==================== MLP МОДУЛЬ (Заготовка) ====================

MLP_CONFIG = {
    # Размерность входа (8 статов + 18 типов)
    'input_dim': 26,
    
    # Размерности скрытых слоёв
    'hidden_layers': [64, 32],
    
    # Размерность эмбеддинга
    'embedding_dim': 16,
    
    # Размер батча
    'batch_size': 32,
    
    # Количество эпох
    'epochs': 30,
    
    # Скорость обучения
    'learning_rate': 0.001,
    
    # Dropout
    'dropout': 0.3,
    
    # Путь к модели
    'model_path': MODELS_DIR / 'mlp' / 'mlp_model.pth',
    
    # Минимум образцов
    'min_samples': 50,
}


# ==================== ОБЩИЕ НАСТРОЙКИ ОБУЧЕНИЯ ====================

TRAINING_CONFIG = {
    # Устройство для вычислений (cuda или cpu)
    # Можно переопределить через переменную окружения USE_CUDA
    'device': 'cuda' if os.environ.get('USE_CUDA', 'false').lower() == 'true' else 'cpu',
    
    # Seed для воспроизводимости результатов
    'seed': 42,
    
    # Количество рабочих процессов для DataLoader
    'num_workers': 0,
    
    # Включить смешанную точность (AMP) для ускорения на GPU
    'use_amp': False,
    
    # Логирование метрик обучения
    'log_metrics': True,
    
    # Сохранять чекпоинты во время обучения
    'save_checkpoints': False,
    
    # Ранняя остановка при отсутствии улучшений
    'early_stopping': False,
    'early_stopping_patience': 5,
}


# ==================== БАЗА ДАННЫХ ====================

DB_CONFIG = {
    # Тип СУБД
    'type': 'sqlite',
    
    # Путь к файлу БД
    'path': str(DATABASE_PATH),
    
    # Включить внешний ключи
    'foreign_keys': True,
    
    # Таймаут для соединений (секунды)
    'timeout': 30,
    
    # Включить WAL режим для лучшей производительности
    'wal_mode': True,
    
    # Таблицы для создания
    'tables': [
        'pokemon',
        'embeddings',
        'model_results',
        'api_cache'
    ],
}


# ==================== API НАСТРОЙКИ ====================

API_CONFIG = {
    # Хост сервера
    'host': '127.0.0.1',
    
    # Порт сервера
    'port': 5000,
    
    # Режим отладки
    'debug': True,
    
    # Включить CORS
    'cors_enabled': False,
    
    # Максимальный размер запроса (MB)
    'max_content_length': 16,
    
    # Таймаут для запросов (секунды)
    'timeout': 30,
}


# ==================== ИНТЕРФЕЙС ====================

UI_CONFIG = {
    # Заголовок приложения
    'title': '🧠 Pokémon AI Similarity Search',
    
    # Тема (light, dark, auto)
    'theme': 'auto',
    
    # Язык интерфейса
    'language': 'ru',
    
    # Количество результатов по умолчанию
    'default_results': 5,
    
    # Включить анимации
    'animations': True,
    
    # Включить автодополнение
    'autocomplete': True,
    
    # Показать тултипы с подсказками
    'tooltips': True,
}


# ==================== ЛОГИРОВАНИЕ ====================

LOGGING_CONFIG = {
    # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    'level': 'INFO',
    
    # Формат логов
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    
    # Путь к файлу логов
    'file_path': BACKEND_DIR / 'logs' / 'app.log',
    
    # Максимальный размер файла логов (MB)
    'max_bytes': 10 * 1024 * 1024,
    
    # Количество резервных файлов
    'backup_count': 3,
    
    # Логировать в консоль
    'console': True,
    
    # Логировать в файл
    'file': False,
}

# Добавить после AE_CONFIG:

# Настройки MLP модуля
MLP_CONFIG = {
    # Размерность входа (4 стата: HP, ATK, DEF, SPD)
    'input_dim': 4,
    
    # Размерности скрытых слоёв
    'hidden_layers': [64, 32, 16],
    
    # Размер батча при обучении
    'batch_size': 32,
    
    # Количество эпох обучения
    'epochs': 50,
    
    # Скорость обучения
    'learning_rate': 0.001,
    
    # Коэффициент Dropout для регуляризации
    'dropout': 0.3,
    
    # Путь к файлу весов модели
    'model_path': MODELS_DIR / 'mlp' / 'mlp_model.pth',
    
    # Минимальное количество покемонов для обучения
    'min_samples': 50,
    
    # Количество симулированных боёв для обучения
    'num_battles': 40000,
    
    # Шанс на неожиданную победу (upset)
    'upset_chance': 0.15,
}

# Добавить после других конфигов:

# Настройки визуализации обучения
GRAPH_CONFIG = {
    # Цветовые схемы для разных метрик
    'colors': {
        'loss': '#667eea',
        'accuracy': '#22c55e',
        'reconstruction': '#f59e0b',
        'ordinariness': '#8b5cf6'
    },
    
    # Настройки 3D визуализации
    'plotly_3d': {
        'camera_default': {'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}},
        'colorscale': 'Viridis',
        'marker_size': 4,
        'line_width': 2
    },
    
    # Настройки 2D графиков
    'chartjs_2d': {
        'tension': 0.4,  # Плавность линий
        'point_radius': 3,
        'fill_opacity': 0.1
    },
    
    # Метки осей для разных моделей
    'axis_labels': {
        'cnn': {'y': 'Contrastive Loss', 'z': 'Embedding Distance'},
        'rnn': {'y': 'Cross-Entropy Loss', 'z': 'Accuracy'},
        'autoencoder': {'y': 'Reconstruction MSE', 'z': 'Ordinariness'},
        'mlp': {'y': 'Binary Cross-Entropy', 'z': 'Win Probability'}
    }
}


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def get_device():
    """
    Возвращает устройство для вычислений (CPU/GPU).
    
    Returns:
        str: 'cuda' или 'cpu'
    """
    return TRAINING_CONFIG['device']


def get_model_path(model_type: str) -> Path:
    """
    Возвращает путь к файлу модели для указанного типа.
    
    Args:
        model_type: Тип модели ('cnn', 'rnn', 'mlp', 'autoencoder')
    
    Returns:
        Path: Путь к файлу модели
    """
    config_map = {
        'cnn': CNN_CONFIG,
        'rnn': RNN_CONFIG,
        'mlp': MLP_CONFIG,
        'autoencoder': AE_CONFIG,
    }
    
    if model_type not in config_map:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    return config_map[model_type]['model_path']


def ensure_dirs():
    """
    Создаёт все необходимые директории если они не существуют.
    """
    dirs_to_create = [
        MODELS_DIR / 'cnn',
        MODELS_DIR / 'rnn',
        MODELS_DIR / 'mlp',
        MODELS_DIR / 'autoencoder',
        BACKEND_DIR / 'logs',
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)


def print_config_summary():
    """
    Выводит сводку конфигурации в консоль.
    """
    print("\n" + "="*60)
    print("📋 КОНФИГУРАЦИЯ ПРОЕКТА")
    print("="*60)
    print(f"📁 Базовая директория: {BASE_DIR}")
    print(f"📁 Backend директория: {BACKEND_DIR}")
    print(f"📁 Models директория: {MODELS_DIR}")
    print(f"🗄️  База данных: {DATABASE_PATH}")
    print(f"🌐 PokeAPI: {POKEAPI_BASE}")
    print(f"🖥️  Устройство: {TRAINING_CONFIG['device']}")
    print(f"🎯 CNN эпох: {CNN_CONFIG['epochs']}, мин. образцов: {CNN_CONFIG['min_samples']}")
    print(f"🧠 RNN эпох: {RNN_CONFIG['epochs']}, мин. образцов: {RNN_CONFIG['min_samples']}")
    print(f"🔄 AE эпох: {AE_CONFIG['epochs']}, мин. образцов: {AE_CONFIG['min_samples']}")
    print("="*60 + "\n")


# ==================== ИНИЦИАЛИЗАЦИЯ ПРИ ИМПОРТЕ ====================

# Создаём необходимые директории при импорте модуля
ensure_dirs()

# Выводим сводку если запускается как основной скрипт
if __name__ == '__main__':
    print_config_summary()