"""
Модуль для загрузки и подготовки данных о покемонах из SQLite.
Все модели используют единый интерфейс получения данных.
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

# Список всех типов покемонов для one-hot encoding
POKEMON_TYPES = [
    "normal", "fighting", "flying", "poison", "ground", "bug", "ghost", 
    "steel", "fire", "water", "grass", "electric", "psychic", "ice", 
    "dragon", "fairy", "dark", "rock"
]

# Числовые характеристики для нормализации
NUMERIC_FEATURES = [
    'hp', 'attack', 'defense', 'special-attack', 
    'special-defense', 'speed', 'height', 'weight'
]


def get_db_connection(db_path: str = 'pokemon.db') -> sqlite3.Connection:
    """
    Создаёт соединение с SQLite базой данных.
    
    Args:
        db_path: Путь к файлу базы данных
        
    Returns:
        sqlite3.Connection объект с row_factory для доступа по имени колонки
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_pokemon_data(db_path: str = 'pokemon.db') -> pd.DataFrame:
    """
    Загружает все данные о покемонах из базы данных в DataFrame.
    
    Returns:
        pd.DataFrame с колонками: id, name, types, stats...
    """
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM pokemon", conn)
    conn.close()
    return df


def extract_numeric_features(df: pd.DataFrame) -> np.ndarray:
    """
    Извлекает и нормализует числовые характеристики покемонов.
    Использует Min-Max нормализацию [0, 1] для стабильности обучения нейросетей.
    
    Args:
        df: DataFrame с данными покемонов
        
    Returns:
        np.ndarrayshape (n_samples, n_features) с нормализованными данными
    """
    # Извлекаем числовые колонки
    numeric_data = df[NUMERIC_FEATURES].values.astype(np.float32)
    
    # Min-Max нормализация: (x - min) / (max - min)
    min_vals = numeric_data.min(axis=0)
    max_vals = numeric_data.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Защита от деления на 0
    
    normalized = (numeric_data - min_vals) / range_vals
    
    return normalized, min_vals, max_vals, range_vals


def encode_types(types_str: str) -> np.ndarray:
    """
    Преобразует строку типов (напр. "fire, flying") в one-hot вектор.
    
    Args:
        types_str: Строка с типами через запятую
        
    Returns:
        np.ndarrayshape (n_types,) с бинарными значениями
    """
    vector = np.zeros(len(POKEMON_TYPES), dtype=np.float32)
    if pd.isna(types_str):
        return vector
    
    types_list = [t.strip().lower() for t in types_str.split(',')]
    for t in types_list:
        if t in POKEMON_TYPES:
            vector[POKEMON_TYPES.index(t)] = 1.0
    return vector


def encode_all_types(df: pd.DataFrame) -> np.ndarray:
    """
    Применяет one-hot encoding ко всем покемонам в DataFrame.
    
    Returns:
        np.ndarrayshape (n_samples, n_types)
    """
    return np.array([encode_types(t) for t in df['types']])


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    """
    Полный пайплайн подготовки признаков для нейросетей.
    Объединяет нормализованные статы и one-hot типы.
    
    Args:
        df: DataFrame с данными
        
    Returns:
        Tuple:
            - np.ndarray: готовый фич-вектор (n_samples, n_features)
            - Dict: параметры нормализации для будущего использования
    """
    # 1. Нормализуем числовые признаки
    numeric, min_v, max_v, range_v = extract_numeric_features(df)
    
    # 2. One-hot кодируем типы
    types = encode_all_types(df)
    
    # 3. Объединяем в один вектор
    X = np.hstack([numeric, types])
    
    scaler_params = {
        'min': min_v,
        'max': max_v,
        'range': range_v,
        'numeric_cols': NUMERIC_TYPES,
        'type_cols': POKEMON_TYPES
    }
    
    return X, scaler_params


def get_pokemon_by_name(name: str, db_path: str = 'pokemon.db') -> Optional[pd.DataFrame]:
    """
    Получает данные одного покемона по имени (регистронезависимо).
    
    Args:
        name: Имя покемона
        
    Returns:
        pd.DataFrame с одной строкой или None если не найден
    """
    conn = get_db_connection(db_path)
    query = "SELECT * FROM pokemon WHERE LOWER(name) = LOWER(?)"
    df = pd.read_sql_query(query, conn, params=(name,))
    conn.close()
    return df if len(df) > 0 else None