"""
Универсальный модуль для работы с SQLite базой данных.
Используется всеми типами нейросетей (CNN, RNN, Autoencoder, MLP) 
для хранения данных покемонов, эмбеддингов, метрик обучения и результатов предсказаний.

Функционал:
- Хранение основных данных покемонов из PokeAPI
- Хранение векторных представлений (эмбеддингов) для ML-моделей
- Хранение метрик обучения для визуализации графиков
- Хранение результатов предсказаний моделей
- Кэширование ответов API для оптимизации запросов
- Автоматическая миграция схемы БД при добавлении новых колонок
"""
import sqlite3
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from config import DATABASE_PATH


class PokemonDatabase:
    """
    Класс для работы с базой данных покемонов.
    
    Таблицы:
    - pokemon: основные данные покемонов из PokeAPI
    - embeddings: векторные представления для разных типов моделей (CNN, RNN, etc.)
    - model_results: результаты предсказаний моделей (классификация, вероятность победы, etc.)
    - training_metrics: метрики обучения для визуализации графиков
    - api_cache: кэш ответов API для оптимизации запросов
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
        """
        Возвращает соединение с БД с поддержкой именованных колонок.
        
        Returns:
            sqlite3.Connection: Соединение с базой данных
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Включаем поддержку внешних ключей
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def _init_tables(self):
        """
        Создаёт необходимые таблицы, если они не существуют.
        Добавляет недостающие колонки при миграции.
        
        Таблицы:
        - pokemon: основные данные покемонов из PokeAPI
        - embeddings: векторные представления для разных типов моделей
        - model_results: результаты предсказаний моделей
        - training_metrics: метрики обучения для визуализации
        - api_cache: кэш ответов API
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # ==================== Таблица pokemon ====================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pokemon (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                types TEXT,
                height INTEGER,
                weight INTEGER,
                hp INTEGER, 
                attack INTEGER, 
                defense INTEGER,
                special_attack INTEGER, 
                special_defense INTEGER, 
                speed INTEGER,
                sprite_url TEXT,
                sprite_data BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Миграция: добавляем недостающие колонки
        cursor.execute("PRAGMA table_info(pokemon)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'sprite_url' not in columns:
            print("🔧 Миграция: добавляем колонку sprite_url...")
            cursor.execute('ALTER TABLE pokemon ADD COLUMN sprite_url TEXT')
        
        if 'sprite_data' not in columns:
            print("🔧 Миграция: добавляем колонку sprite_data...")
            cursor.execute('ALTER TABLE pokemon ADD COLUMN sprite_data BLOB')
        
        if 'special_attack' not in columns:
            print("🔧 Миграция: добавляем колонку special_attack...")
            cursor.execute('ALTER TABLE pokemon ADD COLUMN special_attack INTEGER')
        
        if 'special_defense' not in columns:
            print("🔧 Миграция: добавляем колонку special_defense...")
            cursor.execute('ALTER TABLE pokemon ADD COLUMN special_defense INTEGER')
        
        if 'updated_at' not in columns:
            print("🔧 Миграция: добавляем колонку updated_at...")
            cursor.execute('ALTER TABLE pokemon ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
        
        # ==================== Таблица embeddings ====================
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
        
        # ==================== Таблица model_results ====================
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
        
        # ==================== Таблица training_metrics ====================
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                loss REAL,
                accuracy REAL,
                metric_name TEXT,
                metric_value REAL,
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_type, epoch, metric_name)
            )
        ''')
        
        # ==================== Таблица api_cache ====================
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
        
        # ==================== Индексы для ускорения поиска ====================
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pokemon_name ON pokemon(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pokemon_types ON pokemon(types)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_pokemon ON embeddings(pokemon_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_results_model ON model_results(model_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_results_pokemon ON model_results(pokemon_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_metrics_model ON training_metrics(model_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_metrics_epoch ON training_metrics(epoch)')
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
                 Ожидаемые ключи: id, name, types, height, weight, hp, attack, 
                 defense, special-attack, special-defense, speed, sprite_url
        
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
                    sprite_url, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                data.get('sprite_url'),
                datetime.now().isoformat()
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"❌ Ошибка добавления покемона: {e}")
            return False
        finally:
            conn.close()
    
    def get_pokemon_by_name(self, name: str) -> Optional[Dict]:
        """
        Получает данные покемона по имени (без учета регистра).
        
        Args:
            name: Имя покемона
        
        Returns:
            Dict: Данные покемона или None если не найден
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM pokemon WHERE LOWER(name) = LOWER(?)', (name,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def get_pokemon_by_id(self, pokemon_id: int) -> Optional[Dict]:
        """
        Получает данные покемона по ID.
        
        Args:
            pokemon_id: ID покемона
        
        Returns:
            Dict: Данные покемона или None если не найден
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM pokemon WHERE id = ?', (pokemon_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    def get_all_pokemon(self) -> List[Dict]:
        """
        Возвращает список всех покемонов в БД.
        
        Returns:
            List[Dict]: Список словарей с данными покемонов
        """
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
        
        Returns:
            List[Dict]: Список покемонов со спрайтами
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, sprite_url FROM pokemon WHERE sprite_url IS NOT NULL')
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_pokemon_count(self) -> int:
        """
        Возвращает общее количество покемонов в БД.
        
        Returns:
            int: Количество покемонов
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM pokemon')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_pokemon_with_sprites_count(self) -> int:
        """
        Возвращает количество покемонов со спрайтами.
        
        Returns:
            int: Количество покемонов со спрайтами
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM pokemon WHERE sprite_url IS NOT NULL')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def delete_pokemon(self, pokemon_id: int) -> bool:
        """
        Удаляет покемона по ID.
        
        Args:
            pokemon_id: ID покемона для удаления
        
        Returns:
            bool: True если успешно удалён
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('DELETE FROM pokemon WHERE id = ?', (pokemon_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"❌ Ошибка удаления покемона: {e}")
            return False
        finally:
            conn.close()
    
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
        
        Args:
            pokemon_id: ID покемона
            model_type: Тип модели
        
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
        
        Args:
            model_type: Тип модели ('cnn', 'rnn', 'mlp', 'autoencoder')
        
        Returns:
            Dict: Словарь {pokemon_id: embedding}
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
    
    def delete_embedding(self, pokemon_id: int, model_type: str) -> bool:
        """
        Удаляет эмбеддинг для указанного покемона и модели.
        
        Args:
            pokemon_id: ID покемона
            model_type: Тип модели
        
        Returns:
            bool: True если успешно удалён
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'DELETE FROM embeddings WHERE pokemon_id = ? AND model_type = ?',
                (pokemon_id, model_type)
            )
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"❌ Ошибка удаления эмбеддинга: {e}")
            return False
        finally:
            conn.close()
    
    # ==================== Методы для работы с результатами моделей ====================
    
    def save_model_result(self, pokemon_id: int, model_type: str, 
                          result_data: Dict, confidence: float = None) -> bool:
        """
        Сохраняет результат работы модели для покемона.
        
        Args:
            pokemon_id: ID покемона
            model_type: Тип модели ('cnn', 'rnn', 'mlp', 'autoencoder')
            result_data: Словарь с результатами (сериализуется в JSON)
            confidence: Уверенность модели (0.0 - 1.0)
        
        Returns:
            bool: True если успешно сохранено
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO model_results 
                (pokemon_id, model_type, result_data, confidence)
                VALUES (?, ?, ?, ?)
            ''', (
                pokemon_id,
                model_type,
                json.dumps(result_data, ensure_ascii=False),
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
        
        Args:
            pokemon_id: ID покемона
            model_type: Тип модели
        
        Returns:
            Dict: Результат или None если не найден
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT result_data, confidence FROM model_results WHERE pokemon_id = ? AND model_type = ?',
            (pokemon_id, model_type)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            result = json.loads(row['result_data'])
            result['confidence'] = row['confidence']
            return result
        return None
    
    def get_all_model_results(self, model_type: str) -> Dict[int, Dict]:
        """
        Получает все результаты для указанного типа модели.
        
        Args:
            model_type: Тип модели
        
        Returns:
            Dict: Словарь {pokemon_id: result_data}
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT pokemon_id, result_data, confidence FROM model_results WHERE model_type = ?',
            (model_type,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return {
            row['pokemon_id']: {
                **json.loads(row['result_data']),
                'confidence': row['confidence']
            }
            for row in rows
        }
    
    def delete_model_result(self, pokemon_id: int, model_type: str) -> bool:
        """
        Удаляет результат модели для указанного покемона.
        
        Args:
            pokemon_id: ID покемона
            model_type: Тип модели
        
        Returns:
            bool: True если успешно удалён
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'DELETE FROM model_results WHERE pokemon_id = ? AND model_type = ?',
                (pokemon_id, model_type)
            )
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"❌ Ошибка удаления результата модели: {e}")
            return False
        finally:
            conn.close()
    
    # ==================== Методы для работы с метриками обучения ====================
    
    def save_training_metrics(self, model_type: str, metrics: List[Dict]) -> bool:
        """
        Сохраняет метрики обучения модели в БД.
        
        Args:
            model_type: Тип модели ('cnn', 'rnn', 'autoencoder', 'mlp')
            metrics: Список словарей с метриками для каждой эпохи
                    [{'epoch': 1, 'loss': 0.5, 'accuracy': 0.85}, ...]
        
        Returns:
            bool: True если успешно сохранено
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            for m in metrics:
                cursor.execute('''
                    INSERT OR REPLACE INTO training_metrics 
                    (model_type, epoch, loss, accuracy, metric_name, metric_value)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    model_type,
                    m.get('epoch'),
                    m.get('loss'),
                    m.get('accuracy'),
                    m.get('metric_name', 'primary'),
                    m.get('metric_value', m.get('loss'))
                ))
            conn.commit()
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения метрик: {e}")
            return False
        finally:
            conn.close()
    
    def get_training_metrics(self, model_type: str) -> List[Dict]:
        """
        Получает метрики обучения для указанной модели.
        
        Args:
            model_type: Тип модели
        
        Returns:
            List[Dict]: Список метрик по эпохам
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT epoch, loss, accuracy, metric_name, metric_value, trained_at
            FROM training_metrics 
            WHERE model_type = ? 
            ORDER BY epoch
        ''', (model_type,))
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'epoch': row['epoch'],
                'loss': row['loss'],
                'accuracy': row['accuracy'],
                'metric_name': row['metric_name'],
                'metric_value': row['metric_value'],
                'trained_at': row['trained_at']
            }
            for row in rows
        ]
    
    def clear_training_metrics(self, model_type: str = None):
        """
        Очищает метрики обучения.
        
        Args:
            model_type: Если указан — очищает только для этой модели
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            if model_type:
                cursor.execute('DELETE FROM training_metrics WHERE model_type = ?', (model_type,))
            else:
                cursor.execute('DELETE FROM training_metrics')
            
            conn.commit()
            print(f"🗑️ Метрики обучения очищены{' для ' + model_type if model_type else ''}")
        except Exception as e:
            print(f"❌ Ошибка очистки метрик: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    # ==================== Методы для API кэша ====================
    
    def cache_api_response(self, endpoint: str, params: Dict, response: Dict):
        """
        Кэширует ответ API для повторного использования.
        
        Args:
            endpoint: URL endpoint API
            params: Параметры запроса
            response: Ответ от API
        """
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
        except Exception as e:
            print(f"❌ Ошибка кэширования API ответа: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_cached_response(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Получает кэшированный ответ API если он есть.
        
        Args:
            endpoint: URL endpoint API
            params: Параметры запроса
        
        Returns:
            Dict: Кэшированный ответ или None
        """
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
    
    def clear_api_cache(self, endpoint: str = None):
        """
        Очищает кэш API.
        
        Args:
            endpoint: Если указан — очищает только для этого endpoint
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            if endpoint:
                cursor.execute('DELETE FROM api_cache WHERE endpoint = ?', (endpoint,))
            else:
                cursor.execute('DELETE FROM api_cache')
            
            conn.commit()
            print(f"🗑️ Кэш API очищен{' для ' + endpoint if endpoint else ''}")
        except Exception as e:
            print(f"❌ Ошибка очистки кэша API: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    # ==================== Утилиты ====================
    
    def clear_all(self):
        """
        Очищает все данные из всех таблиц.
        Используется при сбросе приложения.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Очищаем в правильном порядке из-за внешних ключей
            cursor.execute('DELETE FROM training_metrics')
            cursor.execute('DELETE FROM model_results')
            cursor.execute('DELETE FROM embeddings')
            cursor.execute('DELETE FROM api_cache')
            cursor.execute('DELETE FROM pokemon')
            
            conn.commit()
            print("🗑️ Все данные очищены")
        except Exception as e:
            print(f"❌ Ошибка очистки: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def get_database_stats(self) -> Dict:
        """
        Возвращает статистику по базе данных.
        
        Returns:
            Dict: Статистика по всем таблицам
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Количество покемонов
        cursor.execute('SELECT COUNT(*) FROM pokemon')
        stats['pokemon_count'] = cursor.fetchone()[0]
        
        # Количество покемонов со спрайтами
        cursor.execute('SELECT COUNT(*) FROM pokemon WHERE sprite_url IS NOT NULL')
        stats['pokemon_with_sprites'] = cursor.fetchone()[0]
        
        # Количество эмбеддингов по типам моделей
        for model_type in ['cnn', 'rnn', 'mlp', 'autoencoder']:
            cursor.execute('SELECT COUNT(*) FROM embeddings WHERE model_type = ?', (model_type,))
            stats[f'{model_type}_embeddings'] = cursor.fetchone()[0]
        
        # Количество результатов моделей
        for model_type in ['cnn', 'rnn', 'mlp', 'autoencoder']:
            cursor.execute('SELECT COUNT(*) FROM model_results WHERE model_type = ?', (model_type,))
            stats[f'{model_type}_results'] = cursor.fetchone()[0]
        
        # Количество метрик обучения
        for model_type in ['cnn', 'rnn', 'mlp', 'autoencoder']:
            cursor.execute('SELECT COUNT(*) FROM training_metrics WHERE model_type = ?', (model_type,))
            stats[f'{model_type}_metrics'] = cursor.fetchone()[0]
        
        # Размер кэша API
        cursor.execute('SELECT COUNT(*) FROM api_cache')
        stats['api_cache_count'] = cursor.fetchone()[0]
        
        conn.close()
        
        return stats
    
    def export_to_json(self, output_path: str) -> bool:
        """
        Экспортирует все данные из БД в JSON файл.
        
        Args:
            output_path: Путь к выходному JSON файлу
        
        Returns:
            bool: True если успешно экспортировано
        """
        try:
            data = {
                'pokemon': self.get_all_pokemon(),
                'stats': self.get_database_stats(),
                'exported_at': datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"📦 Данные экспортированы в {output_path}")
            return True
        except Exception as e:
            print(f"❌ Ошибка экспорта: {e}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Создаёт резервную копию базы данных.
        
        Args:
            backup_path: Путь к файлу резервной копии
        
        Returns:
            bool: True если успешно создано
        """
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            print(f"💾 Резервная копия создана: {backup_path}")
            return True
        except Exception as e:
            print(f"❌ Ошибка создания резервной копии: {e}")
            return False
    
    def __del__(self):
        """Деструктор для гарантированного закрытия соединений."""
        pass  # Соединения закрываются автоматически после каждого запроса