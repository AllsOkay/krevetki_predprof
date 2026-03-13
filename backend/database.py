"""
Универсальный модуль для работы с SQLite.
Используется всеми типами нейросетей для хранения данных.
"""
import sqlite3
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional, Any

from config import DATABASE_PATH


class PokemonDatabase:
    """
    Класс для работы с базой данных покемонов.
    
    Таблица `pokemon` хранит основные данные, полученные из PokeAPI.
    Таблица `embeddings` хранит векторные представления для ML-моделей.
    Таблица `api_cache` кэширует ответы API для уменьшения нагрузки.
    """
    
    def __init__(self, db_path: str = None):
        """
        Инициализация подключения к БД.
        
        Args:
            db_path: Путь к файлу базы данных (по умолчанию из config)
        """
        self.db_path = db_path or str(DATABASE_PATH)
        self._init_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Возвращает соединение с БД с поддержкой именованных колонок."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_tables(self):
        """
        Создаёт необходимые таблицы, если они не существуют.
        Добавляет недостающие колонки при миграции.
        
        Таблицы:
        - pokemon: основные данные покемонов из PokeAPI
        - embeddings: векторные представления для разных типов моделей
        - api_cache: кэш ответов API для оптимизации запросов
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Создаём основную таблицу если не существует
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pokemon (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                types TEXT,
                height INTEGER,
                weight INTEGER,
                hp INTEGER, attack INTEGER, defense INTEGER,
                special_attack INTEGER, special_defense INTEGER, speed INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Проверяем и добавляем недостающие колонки (миграция)
        cursor.execute("PRAGMA table_info(pokemon)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'sprite_url' not in columns:
            print("🔧 Миграция: добавляем колонку sprite_url...")
            cursor.execute('ALTER TABLE pokemon ADD COLUMN sprite_url TEXT')
        
        if 'sprite_data' not in columns:
            print("🔧 Миграция: добавляем колонку sprite_data...")
            cursor.execute('ALTER TABLE pokemon ADD COLUMN sprite_data BLOB')
        
        # Таблица для хранения эмбеддингов (векторных представлений)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pokemon_id INTEGER NOT NULL,
                model_type TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pokemon_id, model_type),
                FOREIGN KEY (pokemon_id) REFERENCES pokemon(id) ON DELETE CASCADE
            )
        ''')
        
        # Таблица для кэширования ответов PokeAPI
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                params TEXT,
                response BLOB NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(endpoint, params)
            )
        ''')
        
        # Индексы для ускорения поиска
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pokemon_name ON pokemon(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_cache_endpoint ON api_cache(endpoint)')
        
        conn.commit()
        conn.close()
        print("✅ База данных инициализирована")
    
    # ==================== Методы для работы с таблицей pokemon ====================
    
    def add_pokemon(self, data: Dict[str, Any]) -> bool:
        """
        Добавляет или обновляет данные покемона.
        
        Args:
            data: Словарь с данными покемона (должен содержать 'id' и 'name')
        
        Returns:
            bool: True если операция успешна
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO pokemon (
                    id, name, types, height, weight,
                    hp, attack, defense, special_attack, special_defense, speed,
                    sprite_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['id'],
                data['name'],
                data.get('types'),
                data.get('height'),
                data.get('weight'),
                data.get('hp'),
                data.get('attack'),
                data.get('defense'),
                data.get('special-attack'),
                data.get('special-defense'),
                data.get('speed'),
                data.get('sprite_url')
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"❌ Ошибка добавления покемона: {e}")
            return False
        finally:
            conn.close()
    
    def get_pokemon_by_name(self, name: str) -> Optional[Dict]:
        """Получает данные покемона по имени (без учета регистра)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM pokemon WHERE LOWER(name) = LOWER(?)', (name,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def get_all_pokemon(self) -> List[Dict]:
        """Возвращает список всех покемонов в БД."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM pokemon ORDER BY id')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_pokemon_with_sprites(self) -> List[Dict]:
        """
        Возвращает покемонов, у которых есть URL спрайта.
        Используется для подготовки данных к обучению CNN.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, sprite_url FROM pokemon WHERE sprite_url IS NOT NULL')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    # ==================== Методы для работы с эмбеддингами ====================
    
    def save_embedding(self, pokemon_id: int, model_type: str, embedding: Any) -> bool:
        """
        Сохраняет векторное представление покемона для указанной модели.
        
        Args:
            pokemon_id: ID покемона в таблице pokemon
            model_type: Тип модели ('cnn', 'rnn', 'mlp', 'autoencoder')
            embedding: Вектор (numpy array или список) для сериализации
        
        Returns:
            bool: True если успешно сохранено
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Сериализуем эмбеддинг в бинарный формат (pickle)
            embedding_blob = pickle.dumps(embedding)
            
            cursor.execute('''
                INSERT OR REPLACE INTO embeddings (pokemon_id, model_type, embedding)
                VALUES (?, ?, ?)
            ''', (pokemon_id, model_type, embedding_blob))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения эмбеддинга: {e}")
            return False
        finally:
            conn.close()
    
    def get_embedding(self, pokemon_id: int, model_type: str) -> Optional[Any]:
        """
        Получает векторное представление покемона для указанной модели.
        
        Returns:
            Десериализованный вектор или None если не найден
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT embedding FROM embeddings WHERE pokemon_id = ? AND model_type = ?',
            (pokemon_id, model_type)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return pickle.loads(row['embedding'])
        return None
    
    def get_all_embeddings(self, model_type: str) -> Dict[int, Any]:
        """
        Получает все эмбеддинги для указанного типа модели.
        
        Returns:
            Словарь {pokemon_id: embedding}
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT pokemon_id, embedding FROM embeddings WHERE model_type = ?',
            (model_type,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return {
            row['pokemon_id']: pickle.loads(row['embedding'])
            for row in rows
        }
    
    # ==================== Методы для API кэша ====================
    
    def cache_api_response(self, endpoint: str, params: Dict, response: Dict):
        """Кэширует ответ API для повторного использования."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO api_cache (endpoint, params, response)
                VALUES (?, ?, ?)
            ''', (
                endpoint,
                json.dumps(params, sort_keys=True),
                json.dumps(response)
            ))
            conn.commit()
        finally:
            conn.close()
    
    def get_cached_response(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Получает кэшированный ответ API если он есть."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT response FROM api_cache WHERE endpoint = ? AND params = ?',
            (endpoint, json.dumps(params, sort_keys=True))
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row['response'])
        return None
    
    # ==================== Утилиты ====================
    
    def get_pokemon_count(self) -> int:
        """Возвращает общее количество покемонов в БД."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM pokemon')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_pokemon_with_sprites_count(self) -> int:
        """Возвращает количество покемонов со спрайтами."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM pokemon WHERE sprite_url IS NOT NULL')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def clear_all(self):
        """Очищает все данные из всех таблиц."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM pokemon')
            cursor.execute('DELETE FROM embeddings')
            cursor.execute('DELETE FROM api_cache')
            conn.commit()
            print("🗑️ Все данные очищены")
        except Exception as e:
            print(f"❌ Ошибка очистки: {e}")
            conn.rollback()
        finally:
            conn.close()

        # Добавить в конец класса PokemonDatabase:

    def save_model_result(self, pokemon_id: int, model_type: str, 
                        result_data: Dict, confidence: float = None) -> bool:
        """
        Сохраняет результат работы модели для покемона.
        
        Args:
            pokemon_id: ID покемона
            model_type: Тип модели ('cnn', 'rnn', 'mlp', 'autoencoder')
            result_data: Словарь с результатами (сериализуется в JSON)
            confidence: Уверенность модели (0.0 - 1.0)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pokemon_id INTEGER NOT NULL,
                    model_type TEXT NOT NULL,
                    result_data TEXT NOT NULL,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pokemon_id) REFERENCES pokemon(id) ON DELETE CASCADE
                )
            ''')
            
            cursor.execute('''
                INSERT OR REPLACE INTO model_results 
                (pokemon_id, model_type, result_data, confidence)
                VALUES (?, ?, ?, ?)
            ''', (
                pokemon_id,
                model_type,
                json.dumps(result_data),
                confidence
            ))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения результата модели: {e}")
            return False
        finally:
            conn.close()

    def get_model_result(self, pokemon_id: int, model_type: str) -> Optional[Dict]:
        """
        Получает результат работы модели для покемона.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT result_data FROM model_results WHERE pokemon_id = ? AND model_type = ?',
            (pokemon_id, model_type)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row['result_data'])
        return None

    def get_all_model_results(self, model_type: str) -> Dict[int, Dict]:
        """
        Получает все результаты для указанного типа модели.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT pokemon_id, result_data FROM model_results WHERE model_type = ?',
            (model_type,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return {
            row['pokemon_id']: json.loads(row['result_data'])
            for row in rows
        }