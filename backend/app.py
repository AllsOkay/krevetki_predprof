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
    prepare_for_autoencoder, prepare_for_siamese
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
            rnn = PokemonRNN(input_dim=26, output_dim=8)  # Прогноз 8 числовых признаков
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
            print("✅ Параметры нормализации загружены")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки scaler: {e}")

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
        log_api_request('/api/pokemon', {}, {'status': 'cleared', 'deleted': deleted})
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
            
            model = PokemonRNN(input_dim=26, output_dim=26)
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
            # Для RNN нужна последовательность - берем последних 5 покемонов из БД
            all_df = load_pokemon_data()
            if len(all_df) < 5:
                return jsonify({'error': 'Недостаточно данных для RNN прогноза'}), 400
            
            last_5 = all_df.tail(5)
            X_seq, _ = prepare_features(last_5)
            X_rnn = prepare_for_rnn(X_seq, sequence_length=5)[-1:]  # Берем последнюю последовательность
            
            with torch.no_grad():
                prediction = model(X_rnn)
                # Возвращаем предсказанные числовые признаки
                pred_stats = prediction[0][:8].numpy()  # Первые 8 - числовые признаки
            
            result = {
                'pokemon': pokemon_name,
                'predicted_stats': {
                    NUMERIC_FEATURES[i]: float(pred_stats[i]) for i in range(8)
                }
            }
            log_api_request(f'/api/predict/{model_type}', data, result, model_type)
            return jsonify(result)
            
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
        
        with torch.no_grad():
            # Кодируем в latent space
            target_emb = model.encode(torch.FloatTensor(X_target)).numpy().flatten()
            all_embs = model.encode(torch.FloatTensor(X_all)).numpy()
        
        # Косинусное сходство
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([target_emb], all_embs)[0]
        
        # Исключаем самого себя из результатов
        target_idx = df[df['name'].str.lower() == pokemon_name.lower()].index[0]
        similarities[target_idx] = -1
        
        # Берём топ-5
        top_5_idx = np.argsort(similarities)[::-1][:5]
        
        results = []
        for idx in top_5_idx:
            if similarities[idx] > 0:  # Только положительные сходства
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