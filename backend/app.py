"""
Pokémon AI - Backend сервер на Flask
Поддерживает 5 типов нейросетей: MLP, CNN, RNN, Autoencoder, Siamese
Работает с SQLite для хранения данных покемонов и результатов запросов
"""

from flask import Flask, request, jsonify, send_from_directory
import sqlite3
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# === Импорты утилит ===
from utils.data_loader import (
    load_pokemon_data, prepare_features, get_pokemon_by_name,
    NUMERIC_FEATURES, POKEMON_TYPES
)
from utils.preprocessing import (
    prepare_for_mlp, prepare_for_cnn, prepare_for_rnn,
    prepare_for_autoencoder, prepare_for_siamese,
    prepare_single_sequence_for_rnn # <-- Добавлено
)

# === Импорты моделей ===
from models.mlp.model import PokemonMLP
from models.mlp.trainer import MLPTrainer
from models.cnn.model import PokemonCNN
from models.cnn.trainer import CNNTrainer
from models.rnn.model import PokemonRNN
from models.rnn.trainer import RNNTrainer
from models.autoencoder.model import PokemonAutoencoder
from models.autoencoder.trainer import AutoencoderTrainer
from models.siamese.model import PokemonSiamese
from models.siamese.trainer import SiameseTrainer

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# === Глобальные переменные ===
loaded_models: Dict[str, any] = {}
scaler_params: Optional[Dict] = None
DB_PATH = 'pokemon.db'

# === Инициализация БД ===
def init_db():
    """Создаёт таблицу pokemon если не существует"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
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
            "special-attack" INTEGER,
            "special-defense" INTEGER,
            speed INTEGER
        )
    ''')
    # Таблица для логирования API запросов (требование задания)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            request_data TEXT,
            response_data TEXT,
            model_used TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print(f"✅ База данных '{DB_PATH}' инициализирована")

def get_db_connection():
    """Возвращает соединение с БД"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def log_api_request(endpoint: str, request_data: dict, response_data: dict, model_used: str = None):
    """Сохраняет информацию об API запросе в БД (требование задания)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO api_requests (timestamp, endpoint, request_data, response_data, model_used)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            endpoint,
            json.dumps(request_data),
            json.dumps(response_data),
            model_used
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Не удалось залогировать запрос: {e}")

# === Загрузка моделей ===
def load_all_models():
    """Загружает все обученные модели при старте сервера"""
    global loaded_models, scaler_params
    
    model_dir = Path('models')
    
    # Загрузка MLP
    mlp_path = model_dir / 'mlp' / 'mlp_model.pth'
    if mlp_path.exists():
        try:
            mlp = PokemonMLP(task='classification', output_dim=18)
            mlp.load_state_dict(torch.load(mlp_path, map_location='cpu'))
            mlp.eval()
            loaded_models['mlp'] = mlp
            print("✅ MLP загружен")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки MLP: {e}")
    
    # Загрузка CNN
    cnn_path = model_dir / 'cnn' / 'cnn_model.pth'
    if cnn_path.exists():
        try:
            cnn = PokemonCNN(num_classes=18)
            cnn.load_state_dict(torch.load(cnn_path, map_location='cpu'))
            cnn.eval()
            loaded_models['cnn'] = cnn
            print("✅ CNN загружен")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки CNN: {e}")
    
    # Загрузка RNN
    rnn_path = model_dir / 'rnn' / 'rnn_model.pth'
    if rnn_path.exists():
        try:
            # Проверяем, какая архитектура была у модели при обучении
            # Попробуем загрузить state_dict, чтобы получить параметры
            state_dict = torch.load(rnn_path, map_location='cpu')
            # Получим размерность последнего слоя (output_dim)
            # Имя ключа может отличаться в зависимости от реализации модели
            # Для PokemonRNN, последний слой - это fc (Sequential), а последний модуль - Linear
            # Ключи будут fc.2.weight, fc.2.bias (если 2 слоя ReLU и 1 Linear в конце)
            # Найдём ключ, содержащий '.weight' и относящийся к последнему слою fc
            last_linear_weight_key = None
            for key in sorted(state_dict.keys(), reverse=True):
                if 'fc' in key and 'weight' in key:
                    last_linear_weight_key = key
                    break
            if last_linear_weight_key:
                output_dim = state_dict[last_linear_weight_key].shape[0]
                print(f"   📏 Обнаружен output_dim RNN: {output_dim}")
            else:
                print(f"   ⚠️ Не удалось определить output_dim RNN из state_dict, использую 8")
                output_dim = 8 # Значение по умолчанию, если не найдено
            
            rnn = PokemonRNN(input_dim=26, output_dim=output_dim)  # Прогноз N числовых признаков
            rnn.load_state_dict(torch.load(rnn_path, map_location='cpu'))
            rnn.eval()
            loaded_models['rnn'] = rnn
            print("✅ RNN загружен")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки RNN: {e}")
    
    # Загрузка Autoencoder (ключевая модель для поиска похожих)
    ae_path = model_dir / 'autoencoder' / 'ae_model.pth'
    if ae_path.exists():
        try:
            ae = PokemonAutoencoder(latent_dim=8)
            ae.load_state_dict(torch.load(ae_path, map_location='cpu'))
            ae.eval()
            loaded_models['autoencoder'] = ae
            print("✅ Autoencoder загружен")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки Autoencoder: {e}")
    
    # Загрузка Siamese
    siamese_path = model_dir / 'siamese' / 'siamese_model.pth'
    if siamese_path.exists():
        try:
            siamese = PokemonSiamese(embedding_dim=16)
            siamese.load_state_dict(torch.load(siamese_path, map_location='cpu'))
            siamese.eval()
            loaded_models['siamese'] = siamese
            print("✅ Siamese загружен")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки Siamese: {e}")
    
    # Загрузка параметров нормализации
    scaler_path = Path('utils/scaler_params.npy')
    if scaler_path.exists():
        try:
            scaler_params = np.load(scaler_path, allow_pickle=True).item()
            print("✅ Параметры нормализации загружены") # Отладка
            print(f"   scaler_params keys: {list(scaler_params.keys())}") # Отладка
        except Exception as e:
            print(f"⚠️ Ошибка загрузки scaler: {e}") # Отладка
            scaler_params = None # Убедиться, что устанавливается в None при ошибке
    else:
        print("⚠️ Файл scaler_params.npy")
        scaler_params = None

# === Frontend Routes ===
@app.route('/')
def index():
    """Отдаёт главную HTML-страницу"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/css.css')
def css():
    """Отдаёт файл стилей"""
    return send_from_directory(app.static_folder, 'css.css')

@app.route('/script.js')
def js():
    """Отдаёт JavaScript-файл"""
    return send_from_directory(app.static_folder, 'script.js')

# === CRUD API для покемонов ===
@app.route('/api/pokemon', methods=['GET'])
def get_pokemon():
    """Получение всех покемонов из БД"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM pokemon ORDER BY id')
        rows = cursor.fetchall()
        conn.close()
        
        result = [dict(row) for row in rows]
        log_api_request('/api/pokemon', {}, {'count': len(result)})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pokemon', methods=['POST'])
def add_pokemon():
    """Добавление покемона в БД"""
    try:
        data = request.get_json()
        if not data or 'id' not in data or 'name' not in data:
            return jsonify({'error': 'Требуется id и name'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO pokemon 
            (id, name, types, height, weight, hp, attack, defense, 
             "special-attack", "special-defense", speed) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['id'], data['name'], data.get('types'), data.get('height'),
            data.get('weight'), data.get('hp'), data.get('attack'),
            data.get('defense'), data.get('special-attack'),
            data.get('special-defense'), data.get('speed')
        ))
        
        conn.commit()
        conn.close()
        
        log_api_request('/api/pokemon', data, {'status': 'added'})
        return jsonify({'status': 'ok', 'id': data['id']}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pokemon', methods=['DELETE'])
def clear_pokemon():
    """Очистка всех покемонов из БД"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM pokemon')
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        # Удаляем модели т.к. данные изменились
        for model_name in ['mlp', 'cnn', 'rnn', 'autoencoder', 'siamese']:
            model_path = Path(f'models/{model_name}/{model_name}_model.pth')
            if model_path.exists():
                model_path.unlink()
        if Path('utils/scaler_params.npy').exists():
            Path('utils/scaler_params.npy').unlink()
        
        loaded_models.clear()
        # Исправленная строка: вызов log_api_request с правильными аргументами
        log_api_request('/api/pokemon', {}, {'status': 'cleared', 'deleted': deleted})
        # Возврат правильного JSON-ответа
        return jsonify({'status': 'cleared', 'deleted': deleted})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === API для обучения моделей ===
@app.route('/api/train/<model_type>', methods=['POST'])
def train_model_api(model_type: str):
    """
    Обучение нейросети.
    model_type: 'mlp', 'cnn', 'rnn', 'autoencoder', 'siamese'
    """
    if model_type not in ['mlp', 'cnn', 'rnn', 'autoencoder', 'siamese']:
        return jsonify({'error': f'Неизвестный тип модели: {model_type}'}), 400
    
    try:
        # Загружаем данные
        df = load_pokemon_data()
        if len(df) < 30:
            return jsonify({'error': 'Недостаточно данных (минимум 30 покемонов)'}), 400
        
        X, scaler = prepare_features(df)
        
        # Сохраняем параметры нормализации
        Path('utils').mkdir(exist_ok=True)
        np.save('utils/scaler_params.npy', scaler, allow_pickle=True)
        
        history = {}
        
        if model_type == 'mlp':
            # Задача: классификация основного типа покемона
            y = np.array([
                POKEMON_TYPES.index(t.split(',')[0].strip()) 
                for t in df['types'] if pd.notna(t) and t.split(',')[0].strip() in POKEMON_TYPES
            ])
            
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = PokemonMLP(task='classification', output_dim=18)
            trainer = MLPTrainer(model, task='classification')
            history = trainer.fit(X_tr, y_tr, X_val, y_val, epochs=50, batch_size=16)
            
            # Сохранение
            Path('models/mlp').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), 'models/mlp/mlp_model.pth')
            loaded_models['mlp'] = model
            
        elif model_type == 'cnn':
            # Подготовка данных для CNN: reshape в "изображение"
            X_cnn = prepare_for_cnn(X)
            y = np.array([
                POKEMON_TYPES.index(t.split(',')[0].strip()) 
                for t in df['types'] if pd.notna(t) and t.split(',')[0].strip() in POKEMON_TYPES
            ])
            
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(X_cnn.numpy(), y, test_size=0.2, random_state=42)
            
            model = PokemonCNN(num_classes=18)
            trainer = CNNTrainer(model)
            history = trainer.fit(X_tr, y_tr, X_val, y_val, epochs=30, batch_size=16)
            
            Path('models/cnn').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), 'models/cnn/cnn_model.pth')
            loaded_models['cnn'] = model
            
        elif model_type == 'rnn':
            # Подготовка последовательностей для RNN
            X_rnn = prepare_for_rnn(X, sequence_length=5)
            # Целевая переменная: предсказание следующего покемона в последовательности
            y_rnn = X[5:]  # Сдвиг на 1
            
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_rnn.numpy(), y_rnn, test_size=0.2, random_state=42
            )
            
            model = PokemonRNN(input_dim=26, output_dim=26) # Обучаем на все 26 признаков
            trainer = RNNTrainer(model)
            history = trainer.fit(X_tr, y_tr, X_val, y_val, epochs=30, batch_size=8)
            
            Path('models/rnn').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), 'models/rnn/rnn_model.pth')
            loaded_models['rnn'] = model
            
        elif model_type == 'autoencoder':
            # Автоэнкодер: вход = выход
            model = PokemonAutoencoder(latent_dim=8)
            trainer = AutoencoderTrainer(model)
            history = trainer.fit(X, epochs=100, batch_size=32)
            
            Path('models/autoencoder').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), 'models/autoencoder/ae_model.pth')
            loaded_models['autoencoder'] = model
            
        elif model_type == 'siamese':
            # Сиамская сеть: создаём пары для обучения
            from itertools import combinations
            n = len(X)
            # Создаём пары: похожие (одинаковый тип) и непохожие
            pairs = []
            labels = []
            
            # Группируем по типам
            type_groups = {}
            for i, types_str in enumerate(df['types']):
                if pd.notna(types_str):
                    main_type = types_str.split(',')[0].strip()
                    if main_type not in type_groups:
                        type_groups[main_type] = []
                    type_groups[main_type].append(i)
            
            # Положительные пары (один тип)
            for type_name, indices in type_groups.items():
                if len(indices) >= 2:
                    for i, j in combinations(indices, 2):
                        pairs.append((i, j))
                        labels.append(1)
            
            # Отрицательные пары (разные типы)
            all_indices = list(range(n))
            neg_count = 0
            for i in range(n):
                for j in range(i+1, n):
                    if neg_count >= len(labels) // 2:
                        break
                    type_i = df.iloc[i]['types'].split(',')[0].strip() if pd.notna(df.iloc[i]['types']) else ''
                    type_j = df.iloc[j]['types'].split(',')[0].strip() if pd.notna(df.iloc[j]['types']) else ''
                    if type_i != type_j:
                        pairs.append((i, j))
                        labels.append(0)
                        neg_count += 1
            
            if len(pairs) < 20:
                return jsonify({'error': 'Недостаточно данных для обучения сиамской сети'}), 400
            
            pair_data, pair_labels = prepare_for_siamese(X, pairs)
            
            model = PokemonSiamese(embedding_dim=16)
            trainer = SiameseTrainer(model)
            history = trainer.fit(pair_data, pair_labels, epochs=50, batch_size=16)
            
            Path('models/siamese').mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), 'models/siamese/siamese_model.pth')
            loaded_models['siamese'] = model
        
        log_api_request(f'/api/train/{model_type}', {}, {'status': 'trained', 'history': {k: v[-5:] for k, v in history.items()}})
        return jsonify({'status': 'trained', 'history': {k: v[-5:] for k, v in history.items()}})
        
    except Exception as e:
        import traceback
        print(f"❌ Ошибка обучения {model_type}: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# === API для предсказания ===
@app.route('/api/predict/<model_type>', methods=['POST'])
def predict_api(model_type: str):
    """Предсказание с использованием обученной модели"""
    
    data = request.get_json()
    pokemon_name = data.get('pokemon_name')
    
    if not pokemon_name:
        return jsonify({'error': 'Требуется имя покемона'}), 400
    
    # Получаем данные покемона
    df = get_pokemon_by_name(pokemon_name)
    if df is None:
        return jsonify({'error': f'Покемон "{pokemon_name}" не найден в БД'}), 404
    
    # Подготовка признаков
    X, _ = prepare_features(df)
    
    try:
        if model_type == 'mlp' and 'mlp' in loaded_models:
            model = loaded_models['mlp']
            X_tensor = prepare_for_mlp(X)[0]
            
            with torch.no_grad():
                output = model(X_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                pred_type = POKEMON_TYPES[pred_idx]
            
            result = {
                'pokemon': pokemon_name,
                'predicted_type': pred_type,
                'confidence': float(output[0][pred_idx]),
                'all_probabilities': {
                    POKEMON_TYPES[i]: float(output[0][i]) for i in range(18)
                }
            }
            log_api_request(f'/api/predict/{model_type}', data, result, model_type)
            return jsonify(result)
            
        elif model_type == 'cnn' and 'cnn' in loaded_models:
            model = loaded_models['cnn']
            X_tensor = prepare_for_cnn(X)
            
            with torch.no_grad():
                output = model(X_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                pred_type = POKEMON_TYPES[pred_idx]
            
            result = {
                'pokemon': pokemon_name,
                'predicted_type': pred_type,
                'confidence': float(output[0][pred_idx])
            }
            log_api_request(f'/api/predict/{model_type}', data, result, model_type)
            return jsonify(result)
            
        elif model_type == 'rnn' and 'rnn' in loaded_models:
            model = loaded_models['rnn']
            # --- ИСПРАВЛЕНАЯ ЛОГИКА ДЛЯ RNN ---
            # Загружаем всех покемонов из базы
            all_df = load_pokemon_data()
            if len(all_df) < 4:
                return jsonify({'error': 'Недостаточно данных для RNN прогноза (нужно минимум 4)'}), 400

            # Получаем данные покемона по имени (он может быть в середине или конце)
            target_df = get_pokemon_by_name(pokemon_name)
            if target_df is None:
                # Это "дубль" проверки, но на всякий случай
                return jsonify({'error': f'Покемон "{pokemon_name}" не найден в БД (индексация).'}), 404

            # Подготовим признаки для ВСЕХ покемонов
            X_all, _ = prepare_features(all_df)

            # Найдем индекс введенного покемона в датафрейме
            target_idx_in_all = all_df[all_df['name'].str.lower() == pokemon_name.lower()].index
            if target_idx_in_all.empty:
                 # Это "дубль" проверки, но на всякий случай
                 return jsonify({'error': f'Покемон "{pokemon_name}" не найден в БД (индексация).'}), 404

            target_idx_in_all = target_idx_in_all[0]

            # Создадим "псевдопоследовательность" для RNN.
            # Возьмем 4 предыдущих покемона перед target_idx_in_all (или сколько есть)
            start_idx = max(0, target_idx_in_all - 4)
            sequence_df = all_df.iloc[start_idx:target_idx_in_all + 1] # Включая target

            if len(sequence_df) < 2: # Если введенный покемон первый
                # Нужно хотя бы 2 элемента для последовательности, или использовать фиксированное начало
                # Попробуем использовать первые 5, если target - первый
                if target_idx_in_all == 0 and len(all_df) >= 5:
                     sequence_df = all_df.iloc[0:5]
                else:
                    return jsonify({'error': 'Недостаточно предшествующих данных для создания последовательности.'}), 400

            # Подготовим признаки для этой "псевдопоследовательности"
            X_seq_part, _ = prepare_features(sequence_df)

            # Подготовим последовательность для RNN
            # sequence_length = 5 - попробуем использовать длину получившейся последовательности
            effective_seq_len = len(sequence_df) # Это может быть 2, 3, 4, 5
            # Однако, модель была обучена с фиксированной sequence_length (например, 5).
            # Если длина отличается, нужно либо:
            # 1. Заполнять нулями/предыдущим значением до 5
            # 2. Использовать только последние 5 (если длина > 5)
            # 3. Использовать только первые 5 (если длина > 5) - НЕ РЕКОМЕНДУЕТСЯ
            # 4. Обучить модель на переменной длине (сложнее)

            # Попробуем вариант 1: заполнение до 5 нулями, если короче
            desired_seq_len = 5 # Должно совпадать с sequence_length, используемым при обучении
            if effective_seq_len < desired_seq_len:
                padding_needed = desired_seq_len - effective_seq_len
                # Создадим массив нулей той же размерности, что и один покемон (26)
                # ИСПОЛЬЗУЕМ 0.5 как "нейтральное" значение для паддинга
                padding = np.full((padding_needed, X_seq_part.shape[1]), 0.5, dtype=X_seq_part.dtype)
                # Добавим паддинг в начало последовательности (или в конец - зависит от логики)
                # Обычно паддинг добавляют в начало, чтобы последние "реальные" покемоны были в конце
                X_seq_padded = np.vstack([padding, X_seq_part])
            elif effective_seq_len > desired_seq_len:
                 # Если последовательность длиннее, используем последние desired_seq_len
                 X_seq_padded = X_seq_part[-desired_seq_len:]
            else:
                 # Если длина совпадает
                 X_seq_padded = X_seq_part

            # Теперь X_seq_padded имеет форму (desired_seq_len, num_features)
            # Подготовим его для RNN (batch_size, sequence_length, features)
            # X_rnn_input = prepare_for_rnn(X_seq_padded[np.newaxis, :, :], sequence_length=desired_seq_len) # Добавим batch dimension

            # Используем новую функцию для подготовки одной последовательности фиксированной длины
            # ПЕРЕДАЁМ X_seq_part (до паддинга) и desired_seq_len, и используем 0.5 для паддинга внутри функции
            # НО: prepare_single_sequence_for_rnn теперь ожидает mean_values
            # mean_values_for_padding = np.full(X_seq_part.shape[1], 0.5) # feature_dim = 26
            # X_rnn_input = prepare_single_sequence_for_rnn(X_seq_part, desired_seq_len, mean_values_for_padding)
            # НО: чтобы использовать 0.5 в prepare_single_sequence_for_rnn, нужно её изменить
            # ПЕРЕДЕЛАЕМ: в prepare_single_sequence_for_rnn используем 0.5
            # Тогда вызов: X_rnn_input = prepare_single_sequence_for_rnn(X_seq_part, desired_seq_len, 0.5) <- НЕТ, это число, а не массив
            # Новый вызов: X_rnn_input = prepare_single_sequence_for_rnn(X_seq_part, desired_seq_len, np.full(X_seq_part.shape[1], 0.5)) <- ЛУЧШЕ
            mean_values_for_padding = np.full(X_seq_part.shape[1], 0.5) # feature_dim = 26
            X_rnn_input = prepare_single_sequence_for_rnn(X_seq_part, desired_seq_len, mean_values_for_padding) # Передаём mean_values

            # Убедимся, что модель в режиме eval
            model.eval()
            with torch.no_grad():
                # X_rnn_input теперь правильной формы (1, desired_seq_len, num_features)
                prediction = model(X_rnn_input)
                # Предполагаем, что output_dim модели соответствует количеству признаков (26 или 8)
                # В оригинальном коде было [:8], предполагая 8 статов. Проверим output_dim модели.
                # В trainer.py для RNN output_dim был 26 (X[5:]) или 8 (X[5:][:8]).
                # В model.py default output_dim = 1.
                # В backend.py при обучении RNN использовался PokemonRNN(input_dim=26, output_dim=26)
                # и trainer.fit(X_tr, y_tr, ...) где y_tr = X[5:] (все 26 признаков)
                # Значит, output_dim модели 26.
                # Однако, в примере в trainer.py для output_dim=8, y_tr = X[5:][:, :8]
                # Проверим, какая архитектура была у последней обученной модели.
                # Если output_dim=26, то prediction.shape = (1, 26)
                # Если output_dim=8, то prediction.shape = (1, 8)
                # В предыдущем обсуждении, когда ты обучал RNN, ты использовал output_dim=8?
                # Проверим архитектуру модели при загрузке или захардкодим ожидание.
                # Предположим, что после исправления и переобучения, output_dim=8.
                # Лучше всего проверить output_dim самой модели.
                # model.network[-1] - последний слой Linear <- НЕПРАВИЛЬНО
                # model.fc[-1] - последний слой Linear В FC Sequential <- ПРАВИЛЬНО
                expected_output_dim = model.fc[-1].out_features # <-- ИСПРАВЛЕНО
                # print(f"DEBUG: RNN output_dim = {expected_output_dim}") # Для отладки
                if prediction.shape[1] != expected_output_dim:
                     print(f"WARNING: Predicted shape {prediction.shape[1]} does not match expected {expected_output_dim}")
                     # Возьмем столько, сколько нужно
                     take_dim = min(prediction.shape[1], expected_output_dim)
                     pred_stats_raw = prediction[0][:take_dim].numpy()
                else:
                     pred_stats_raw = prediction[0].numpy()

                # Так как модель обучалась с output_dim=26, берем все 26
                # pred_stats = pred_stats_raw[:8] # <-- Используйте это, если output_dim модели 8
                pred_stats = pred_stats_raw # <-- Используем все 26, если output_dim модели 26
                
            result = {
                'pokemon': pokemon_name, # Имя введенного покемона
                # Берём только первые 8 предсказанных значений для NUMERIC_FEATURES
                'predicted_stats': {
                    NUMERIC_FEATURES[i]: float(pred_stats[i]) for i in range(len(NUMERIC_FEATURES)) # len(NUMERIC_FEATURES) = 8
                }
            }
            log_api_request(f'/api/predict/{model_type}', data, result, model_type)
            return jsonify(result)
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
            
        elif model_type == 'autoencoder' and 'autoencoder' in loaded_models:
            model = loaded_models['autoencoder']
            X_tensor = prepare_for_autoencoder(X)
            
            # Ошибка восстановления = мера аномальности
            error = model.get_reconstruction_error(X_tensor).item()
            
            # Получаем латентный вектор для визуализации
            with torch.no_grad():
                latent = model.encode(X_tensor).numpy().flatten()
            
            result = {
                'pokemon': pokemon_name,
                'anomaly_score': float(error),
                'interpretation': 'нормальный' if error < 0.15 else 'возможно аномальный',
                'latent_vector': latent.tolist()
            }
            log_api_request(f'/api/predict/{model_type}', data, result, model_type)
            return jsonify(result)
            
        elif model_type == 'siamese' and 'siamese' in loaded_models:
            # Сиамская сеть сравнивает с эталонным покемоном
            reference_name = data.get('reference_pokemon', 'pikachu')
            ref_df = get_pokemon_by_name(reference_name)
            if ref_df is None:
                return jsonify({'error': f'Эталонный покемон "{reference_name}" не найден'}), 404
            
            X_ref, _ = prepare_features(ref_df)
            
            model = loaded_models['siamese']
            X1 = torch.FloatTensor(X)
            X2 = torch.FloatTensor(X_ref)
            
            similarity = model.predict_similarity(X1, X2)
            
            result = {
                'pokemon': pokemon_name,
                'reference': reference_name,
                'similarity': float(similarity),
                'interpretation': 'очень похожи' if similarity > 0.8 else 'похожи' if similarity > 0.5 else 'различаются'
            }
            log_api_request(f'/api/predict/{model_type}', data, result, model_type)
            return jsonify(result)
            
        else:
            return jsonify({'error': f'Модель {model_type} не загружена или не реализована'}), 400
            
    except Exception as e:
        import traceback
        print(f"❌ Ошибка предсказания: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# === API для поиска похожих (использует Autoencoder) ===
@app.route('/api/similar/<pokemon_name>', methods=['GET'])
def find_similar_api(pokemon_name: str):
    """
    Находит 5 наиболее похожих покемонов используя латентное пространство автоэнкодера.
    Алгоритм:
    1. Кодируем целевого покемона в latent space
    2. Кодируем всех остальных покемонов
    3. Находим 5 ближайших по косинусному расстоянию
    """
    if 'autoencoder' not in loaded_models or scaler_params is None:
        return jsonify({'error': 'Автоэнкодер не обучен. Сначала обучите модель.'}), 503

    try:
        # Загружаем всех покемонов
        df = load_pokemon_data()
        target_df = get_pokemon_by_name(pokemon_name)

        if target_df is None:
            return jsonify({'error': f'Покемон "{pokemon_name}" не найден'}), 404

        # Подготовка признаков
        X_all, _ = prepare_features(df)
        X_target, _ = prepare_features(target_df)

        model = loaded_models['autoencoder']

        # --- ДОБАВЛЕНО ---
        # Убедиться, что модель в режиме eval перед предсказанием
        model.eval()
        # ---------------

        with torch.no_grad(): # Рекомендуется использовать torch.no_grad() для инференса
            # Кодируем в latent space
            target_emb = model.encode(torch.FloatTensor(X_target)).numpy().flatten()
            all_embs = model.encode(torch.FloatTensor(X_all)).numpy()

        # Косинусное сходство
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([target_emb], all_embs)[0] # <-- Это numpy array

        # Исключаем самого себя из результатов
        target_idx = df[df['name'].str.lower() == pokemon_name.lower()].index[0]
        similarities[target_idx] = -1

        # Берём топ-5
        # similarities - это numpy array, так что np.argsort работает корректно
        top_5_idx = np.argsort(similarities)[::-1][:5] # <-- ОШИБКА БЫЛА ТУТ

        results = []
        for idx in top_5_idx:
            if similarities[idx] > 0:  # <-- Здесь также используется numpy array
                results.append({
                    'name': df.iloc[idx]['name'],
                    'similarity': float(similarities[idx]),
                    'types': df.iloc[idx]['types'],
                    'stats': {
                        'hp': int(df.iloc[idx]['hp']),
                        'attack': int(df.iloc[idx]['attack']),
                        'defense': int(df.iloc[idx]['defense']),
                        'speed': int(df.iloc[idx]['speed'])
                    }
                })

        response = {'similar': results}
        log_api_request(f'/api/similar/{pokemon_name}', {}, response, 'autoencoder')
        return jsonify(response)

    except Exception as e:
        import traceback
        print(f"❌ Ошибка поиска похожих: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# === API для получения статистики обучения ===
@app.route('/api/models/status', methods=['GET'])
def models_status():
    """Возвращает статус загруженных моделей"""
    status = {
        'loaded': list(loaded_models.keys()),
        'scaler_loaded': scaler_params is not None,
        'pokemon_count': len(load_pokemon_data()) if Path(DB_PATH).exists() else 0
    }
    return jsonify(status)

# === Запуск приложения ===
if __name__ == '__main__':
    # Создаём необходимые папки
    for model_name in ['mlp', 'cnn', 'rnn', 'autoencoder', 'siamese']:
        Path(f'models/{model_name}').mkdir(parents=True, exist_ok=True)
    Path('utils').mkdir(exist_ok=True)
    
    # Инициализация
    init_db()
    load_all_models()
    
    print("🚀 Backend запущен: http://127.0.0.1:5000")
    print("🧠 Доступные модели: mlp, cnn, rnn, autoencoder, siamese")
    print("📊 Статус моделей: /api/models/status")
    
    app.run(debug=True, host='127.0.0.1', port=5000)