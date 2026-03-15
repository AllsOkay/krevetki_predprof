# backend/tests/test_app.py
# Unit-тесты для основных компонентов приложения
# Запуск: pytest backend/tests/test_app.py -v

import pytest
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock, mock_open

# Добавляем корень проекта в path для импортов
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.model.neural_net import AlienSignalNet
from backend.model.data_utils import restore_class_labels, preprocess_wav_signal
from backend.auth import hash_password, verify_password, generate_token, verify_token
from backend.database import create_user, get_user, init_db


# === ФИКСТУРЫ ===

@pytest.fixture
def temp_db(tmp_path):
    """Фикстура: временная база данных для тестов"""
    db_path = tmp_path / "test_users.db"
    # Патчим Config.DATABASE_PATH для тестов
    import config
    original_path = config.Config.DATABASE_PATH
    config.Config.DATABASE_PATH = str(db_path)
    
    init_db()
    
    yield str(db_path)
    
    # Восстанавливаем оригинальный путь
    config.Config.DATABASE_PATH = original_path


@pytest.fixture
def sample_model():
    """Фикстура: тестовая модель с маленькими размерами для быстрых тестов"""
    return AlienSignalNet(
        input_shape=(64,),  # Маленький вход для скорости
        num_classes=3,       # 3 класса
        hidden_layers=[16, 8],  # Маленькие слои
        learning_rate=0.01
    )


@pytest.fixture
def sample_signal():
    """Фикстура: тестовый аудиосигнал"""
    # Генерируем простой синусоидальный сигнал
    sample_rate = 16000
    duration = 0.1  # 100 мс для скорости
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * 440 * t).astype(np.int16)  # 440 Гц
    return signal


# === ТЕСТЫ НЕЙРОСЕТИ ===

class TestAlienSignalNet:
    """Тесты для класса нейронной сети"""
    
    def test_initialization(self, sample_model):
        """Проверка корректной инициализации модели"""
        assert sample_model.input_shape == (64,)
        assert sample_model.num_classes == 3
        assert len(sample_model.weights) == 3  # 2 скрытых + 1 выходной
        assert sample_model.is_trained == False
    
    def test_forward_pass_shape(self, sample_model):
        """Проверка формы выхода при прямом проходе"""
        batch_size = 5
        x = np.random.randn(batch_size, 64).astype(np.float32)
        
        output = sample_model.forward(x)
        
        assert output.shape == (batch_size, 3)  # (batch, num_classes)
        # Проверка что это вероятности (сумма ~1)
        assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)
    
    def test_predict_returns_classes(self, sample_model):
        """Проверка что predict возвращает индексы классов"""
        x = np.random.randn(10, 64).astype(np.float32)
        predictions = sample_model.predict(x)
        
        assert predictions.shape == (10,)
        assert np.all((predictions >= 0) & (predictions < 3))
        assert predictions.dtype == np.int64
    
    def test_save_and_load(self, sample_model, tmp_path):
        """Проверка сохранения и загрузки модели"""
        save_path = tmp_path / "test_model.pkl"
        
        # Сохраняем
        sample_model.save(str(save_path))
        assert save_path.exists()
        
        # Загружаем
        loaded = AlienSignalNet.load(str(save_path))
        
        # Проверяем что параметры совпадают
        assert loaded.input_shape == sample_model.input_shape
        assert loaded.num_classes == sample_model.num_classes
        assert loaded.is_trained == True
        
        # Проверяем что предсказания совпадают
        x = np.random.randn(3, 64).astype(np.float32)
        orig_pred = sample_model.predict(x)
        loaded_pred = loaded.predict(x)
        assert np.array_equal(orig_pred, loaded_pred)


# === ТЕСТЫ ПРЕДОБРАБОТКИ ДАННЫХ ===

class TestDataUtils:
    """Тесты для модуля data_utils"""
    
    def test_restore_class_labels_from_strings(self):
        """Восстановление меток из строк в целые числа"""
        # Имитация "повреждённых" строковых меток
        labels = np.array(['civilization_alpha', 'civilization_beta', 
                          'civilization_alpha', 'civilization_gamma'])
        
        restored = restore_class_labels(labels)
        
        # Проверка типа
        assert restored.dtype in [np.int32, np.int64, int]
        # Проверка диапазона
        assert np.all(restored >= 0)
        assert np.all(restored < 3)  # 3 уникальных класса
        # Проверка соответствия
        assert restored[0] == restored[2]  # alpha == alpha
        assert restored[0] != restored[1]  # alpha != beta
    
    def test_restore_class_labels_already_numeric(self):
        """Если метки уже числовые - возвращаем как есть"""
        labels = np.array([0, 2, 1, 0, 2])
        restored = restore_class_labels(labels)
        assert np.array_equal(restored, labels)
    
    def test_preprocess_signal_output_shape(self, sample_signal):
        """Проверка формы выхода предобработки сигнала"""
        processed = preprocess_wav_signal(sample_signal, target_length=1600)
        
        # Спектрограмма: (freq_bins, time_frames)
        assert processed.ndim == 2
        assert processed.dtype == np.float32
        # Нормализация к [0, 1]
        assert np.all(processed >= 0)
        assert np.all(processed <= 1)


# === ТЕСТЫ АУТЕНТИФИКАЦИИ ===

class TestAuth:
    """Тесты для модуля аутентификации"""
    
    def test_password_hashing_and_verification(self):
        """Проверка хэширования и верификации паролей"""
        password = "SecureP@ssw0rd123"
        
        # Хэшируем
        hashed = hash_password(password)
        
        # Проверяем что хэш содержит соль
        assert '$' in hashed
        salt, hash_value = hashed.split('$')
        assert len(salt) == 32  # 16 байт в hex
        
        # Верификация верного пароля
        assert verify_password(password, hashed) == True
        
        # Верификация неверного пароля
        assert verify_password("wrong_password", hashed) == False
    
    def test_token_generation_and_verification(self):
        """Проверка генерации и проверки токенов"""
        token = generate_token('testuser', 'user')
        
        # Токен не пустой
        assert len(token) > 0
        
        # Проверка валидного токена
        assert verify_token(token) == True
        assert verify_token(token, required_role='user') == True
        
        # Проверка неверной роли
        assert verify_token(token, required_role='admin') == False
        
        # Проверка несуществующего токена
        assert verify_token('invalid_token_12345') == False
    
    def test_token_expiration(self):
        """Проверка истечения токена (мокаем время)"""
        import time
        from backend import auth
        
        # Сохраняем оригинальную функцию времени
        original_time = time.time
        
        try:
            # Мокаем время: сначала "сейчас", потом "позже"
            mock_time = MagicMock()
            mock_time.side_effect = [1000, 1000, 2000]  # Начальное, при генерации, при проверке
            
            with patch('backend.auth.time', mock_time):
                # Патчим также Config для короткого времени жизни токена
                with patch('backend.auth.Config.TOKEN_EXPIRY_SECONDS', 500):
                    token = generate_token('expiretest', 'user')
                    
                    # Сразу после создания - валиден
                    assert verify_token(token) == True
                    
                    # После "истечения" (500 сек прошло) - не валиден
                    assert verify_token(token) == False
                    
        finally:
            # Восстанавливаем оригинал
            time.time = original_time


# === ТЕСТЫ БАЗЫ ДАННЫХ ===

class TestDatabase:
    """Тесты для работы с базой данных"""
    
    def test_create_and_get_user(self, temp_db):
        """Создание и получение пользователя"""
        password = "TestP@ss123"
        password_hash = hash_password(password)
        
        # Создаём
        result = create_user(
            username='testuser',
            password_hash=password_hash,
            name='Тест',
            surname='Пользователь',
            role='user'
        )
        assert result == True
        
        # Получаем с проверкой пароля
        user = get_user('testuser', password_hash)
        assert user is not None
        assert user['username'] == 'testuser'
        assert user['name'] == 'Тест'
        assert user['role'] == 'user'
        
        # Получаем без пароля (только информация)
        user_info = get_user('testuser')
        assert user_info is not None
        assert 'password_hash' in user_info  # Но не используем его
    
    def test_duplicate_username(self, temp_db):
        """Попытка создать пользователя с существующим логином"""
        password_hash = hash_password("AnyPass123")
        
        # Первый пользователь
        create_user('duplicate', password_hash, 'First', 'User')
        
        # Второй с тем же логином - должен вернуть False
        result = create_user('duplicate', password_hash, 'Second', 'User')
        assert result == False
    
    def test_admin_role_creation(self, temp_db):
        """Создание пользователя с ролью администратора"""
        result = create_user(
            username='admin',
            password_hash=hash_password('AdminP@ss'),
            name='Админ',
            surname='Системы',
            role='admin'
        )
        assert result == True
        
        user = get_user('admin')
        assert user['role'] == 'admin'


# === ИНТЕГРАЦИОННЫЕ ТЕСТЫ ===

class TestIntegration:
    """Интеграционные тесты: модель + данные + предсказание"""
    
    def test_full_pipeline(self, sample_model, sample_signal):
        """Полный цикл: предобработка -> предсказание"""
        # Предобработка сигнала
        features = preprocess_wav_signal(sample_signal)
        
        # Добавляем batch dimension
        x = features.reshape(1, *features.shape)
        
        # Предсказание
        prediction = sample_model.predict(x)
        probas = sample_model.predict_proba(x)
        
        # Проверки
        assert len(prediction) == 1
        assert 0 <= prediction[0] < sample_model.num_classes
        assert probas.shape == (1, sample_model.num_classes)
        assert np.isclose(probas.sum(), 1.0)
    
    def test_evaluate_metrics(self, sample_model):
        """Проверка расчёта метрик оценки"""
        # Генерируем тестовые данные
        n_samples = 20
        x = np.random.randn(n_samples, 64).astype(np.float32)
        y = np.random.randint(0, 3, size=n_samples)  # 3 класса
        
        # Оценка
        metrics = sample_model.evaluate(x, y)
        
        # Проверка структуры результата
        assert 'accuracy' in metrics
        assert 'loss' in metrics
        assert 'per_class_accuracy' in metrics
        assert 'predictions' in metrics
        
        # Проверка типов и диапазонов
        assert 0 <= metrics['accuracy'] <= 1
        assert metrics['loss'] >= 0
        assert len(metrics['predictions']) == n_samples


# === ЗАПУСК ТЕСТОВ ===

if __name__ == '__main__':
    # Для запуска напрямую: python test_app.py
    pytest.main([__file__, '-v', '--tb=short'])