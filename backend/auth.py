# backend/auth.py
# Модуль для аутентификации и авторизации пользователей
# Реализует хэширование паролей и генерацию токенов сессий

import hashlib
import secrets
import time
from typing import Optional, Dict
from config import Config

# === ХЭШИРОВАНИЕ ПАРОЛЕЙ ===

def hash_password(password: str, salt: Optional[str] = None) -> str:
    """
    Хэширует пароль с использованием соли и SHA-256.
    
    :param password: Исходный пароль пользователя
    :param salt: Опциональная соль (генерируется если не передана)
    :return: Строка формата "salt$hash" для хранения в БД
    """
    # Генерируем случайную соль если не передана
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Комбинируем соль и пароль, затем хэшируем
    salted_password = f"{salt}${password}"
    password_hash = hashlib.sha256(salted_password.encode('utf-8')).hexdigest()
    
    # Возвращаем соль и хэш вместе для последующей проверки
    return f"{salt}${password_hash}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Проверяет пароль против сохранённого хэша.
    
    :param password: Пароль, введённый пользователем
    :param stored_hash: Хэш из базы данных (формат "salt$hash")
    :return: True если пароль верный
    """
    try:
        # Извлекаем соль из сохранённого хэша
        salt, original_hash = stored_hash.split('$', 1)
        
        # Хэшируем введённый пароль с той же солью
        test_hash = hash_password(password, salt)
        
        # Сравниваем хэши (constant-time сравнение для безопасности)
        return secrets.compare_digest(test_hash, f"{salt}${original_hash}")
        
    except (ValueError, AttributeError):
        # Если формат хэша неверен
        return False


# === УПРАВЛЕНИЕ ТОКЕНАМИ СЕССИЙ ===

# Временное хранилище токенов (в продакшене использовать Redis)
# Формат: {token: {'username': ..., 'role': ..., 'expires': ...}}
_active_tokens: Dict[str, Dict] = {}


def generate_token(username: str, role: str) -> str:
    """
    Генерирует новый токен сессии для пользователя.
    
    :param username: Логин пользователя
    :param role: Роль пользователя ('admin' или 'user')
    :return: Уникальный токен сессии
    """
    # Генерируем криптографически стойкий токен
    token = secrets.token_urlsafe(32)
    
    # Устанавливаем время истечения
    expires_at = time.time() + Config.TOKEN_EXPIRY_SECONDS
    
    # Сохраняем информацию о токене
    _active_tokens[token] = {
        'username': username,
        'role': role,
        'expires': expires_at,
        'created': time.time()
    }
    
    # Очищаем истёкшие токены (простая "уборка мусора")
    _cleanup_expired_tokens()
    
    return token


def verify_token(token: Optional[str], required_role: Optional[str] = None) -> bool:
    """
    Проверяет валидность токена и роль пользователя.
    
    :param token: Токен из заголовка запроса
    :param required_role: Опционально, требуемая роль для доступа
    :return: True если токен валиден и роль подходит
    """
    if not token:
        return False
    
    # Убираем префикс "Bearer " если есть
    if token.startswith('Bearer '):
        token = token[7:]
    
    # Проверяем наличие токена в хранилище
    if token not in _active_tokens:
        return False
    
    token_data = _active_tokens[token]
    
    # Проверяем время истечения
    if time.time() > token_data['expires']:
        # Удаляем истёкший токен
        del _active_tokens[token]
        return False
    
    # Проверяем роль если требуется
    if required_role and token_data['role'] != required_role:
        return False
    
    return True


def get_token_info(token: str) -> Optional[Dict]:
    """
    Получает информацию о пользователе по токену.
    
    :param token: Токен сессии
    :return: Словарь с данными пользователя или None
    """
    if verify_token(token):  # Также проверяет истечение
        # Возвращаем копию без чувствительных данных
        data = _active_tokens[token].copy()
        del data['expires']  # Не возвращаем время истечения клиенту
        return data
    return None


def revoke_token(token: str) -> bool:
    """
    Аннулирует токен сессии (для выхода из системы).
    
    :param token: Токен для аннулирования
    :return: True если токен был найден и удалён
    """
    if token.startswith('Bearer '):
        token = token[7:]
    
    if token in _active_tokens:
        del _active_tokens[token]
        return True
    return False


def _cleanup_expired_tokens() -> None:
    """Внутренняя функция: удаляет все истёкшие токены"""
    current_time = time.time()
    expired = [t for t, d in _active_tokens.items() if d['expires'] < current_time]
    for token in expired:
        del _active_tokens[token]


# === DECORATORS ДЛЯ ЗАЩИТЫ ENDPOINTS ===

def require_auth(f):
    """
    Декоратор для защиты эндпоинтов: требует валидный токен.
    
    Использование:
    @app.route('/api/protected')
    @require_auth
    def protected_endpoint():
        # Код доступен только авторизованным пользователям
        pass
    """
    from flask import request, jsonify
    import functools
    
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not verify_token(token):
            return jsonify({'error': 'Authentication required'}), 401
        
        # Добавляем информацию о пользователе в контекст
        request.user = get_token_info(token)
        return f(*args, **kwargs)
    
    return decorated_function


def require_role(role: str):
    """
    Декоратор для проверки роли пользователя.
    
    Использование:
    @app.route('/api/admin')
    @require_auth
    @require_role('admin')
    def admin_only():
        # Код доступен только администраторам
        pass
    """
    from flask import request, jsonify
    import functools
    
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(request, 'user') or request.user.get('role') != role:
                return jsonify({'error': f'{role} access required'}), 403
            return f(*args, **kwargs)
        return decorated_function
    
    return decorator