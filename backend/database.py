# backend/database.py
# Модуль для работы с базой данных пользователей
# Использует SQLite для простоты развёртывания

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict

# === Настройка путей (ДО импорта config) ===
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ===

from config import Config
# Также добавляем папку backend для локальных импортов
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Путь к БД берём из конфигурации
DB_PATH = Config.DATABASE_PATH


def get_connection() -> sqlite3.Connection:
    """
    Создаёт и возвращает подключение к базе данных.
    
    :return: sqlite3.Connection объект
    """
    # Создаём директорию для БД если не существует
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    # Возвращаем строки как словари для удобства
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Инициализирует базу данных: создаёт таблицу пользователей если не существует.
    
    Таблица users содержит:
    - username: уникальный логин (строка)
    - password_hash: хэш пароля (SHA-256)
    - name: имя пользователя (обязательно по ТЗ)
    - surname: фамилия пользователя (обязательно по ТЗ)
    - role: роль 'admin' или 'user'
    - created_at: дата создания записи
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            surname TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            CHECK(role IN ('admin', 'user'))
        )
    ''')
    
    # Создаём индекс для быстрого поиска по логину
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_username ON users(username)')
    
    conn.commit()
    conn.close()


def create_user(username: str, password_hash: str, name: str, 
                surname: str, role: str = 'user') -> bool:
    """
    Создаёт нового пользователя в базе данных.
    
    :param username: Уникальный логин пользователя
    :param password_hash: Хэшированный пароль
    :param name: Имя пользователя (обязательное поле по ТЗ)
    :param surname: Фамилия пользователя (обязательное поле по ТЗ)
    :param role: Роль пользователя ('admin' или 'user')
    :return: True если пользователь создан, False если username уже занят
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO users (username, password_hash, name, surname, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, password_hash, name, surname, role))
        conn.commit()
        return True
        
    except sqlite3.IntegrityError:
        # Username уже существует
        return False
        
    finally:
        conn.close()


def get_user(username: str, password_hash: Optional[str] = None) -> Optional[Dict]:
    """
    Получает пользователя из базы данных.
    
    :param username: Логин пользователя
    :param password_hash: Опционально, для проверки пароля
    :return: Словарь с данными пользователя или None если не найден
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if password_hash:
        # Проверка логина и пароля
        cursor.execute('''
            SELECT * FROM users 
            WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))
    else:
        # Только поиск по логину (для получения информации)
        cursor.execute('''
            SELECT * FROM users WHERE username = ?
        ''', (username,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        # Конвертируем sqlite3.Row в dict
        return dict(row)
    return None


def update_last_login(username: str) -> bool:
    """
    Обновляет время последнего входа пользователя.
    
    :param username: Логин пользователя
    :return: True если обновление успешно
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?
    ''', (username,))
    
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return success


def get_all_users() -> List[Dict]:
    """
    Получает список всех пользователей (для админ-панели).
    
    :return: Список словарей с данными пользователей (без паролей!)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Не возвращаем password_hash в списке пользователей
    cursor.execute('''
        SELECT id, username, name, surname, role, created_at, last_login
        FROM users ORDER BY created_at DESC
    ''')
    
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return users


def delete_user(username: str) -> bool:
    """
    Удаляет пользователя из базы данных.
    
    :param username: Логин пользователя для удаления
    :return: True если пользователь удалён
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM users WHERE username = ?', (username,))
    
    success = cursor.rowcount > 0
    conn.commit()
    conn.close()
    
    return success


# Создаём БД при импорте модуля (удобно для разработки)
# В продакшене лучше вызывать init_db() явно из app.py
init_db()