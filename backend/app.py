from flask import Flask, request, jsonify, send_from_directory
import sqlite3
import torch
import numpy as np
import os
import threading
from model import PokemonAutoencoder
from trainer import train_model, load_model_and_scaler, preprocess_data, get_data_from_db, TYPES_LIST

app = Flask(__name__, static_folder='../frontend', static_url_path='')

DATABASE = 'pokemon.db'
MODEL = None
SCALER = None

def init_model():
    """Пытается загрузить модель, если нет - обучает при наличии данных"""
    global MODEL, SCALER
    m, s = load_model_and_scaler()
    if m and s:
        MODEL, SCALER = m, s
        print("✅ Модель загружена из файла")
    else:
        print("⚠️ Модель не найдена. Попытка обучения...")
        # Запускаем обучение в отдельном потоке, чтобы не блокировать старт
        t = threading.Thread(target=try_auto_train)
        t.start()

def try_auto_train():
    if train_model():
        global MODEL, SCALER
        MODEL, SCALER = load_model_and_scaler()

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# --- Frontend Routes ---
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/css.css')
def css():
    return send_from_directory(app.static_folder, 'css.css')

@app.route('/script.js')
def js():
    return send_from_directory(app.static_folder, 'script.js')

# --- API Routes ---
@app.route('/api/pokemon', methods=['GET'])
def get_pokemon():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM pokemon ORDER BY id')
    rows = cur.fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])

@app.route('/api/pokemon', methods=['POST'])
def add_pokemon():
    # (Код добавления из предыдущей версии, сокращен для краткости)
    # Важно: после добавления можно предложить переобучить модель
    data = request.get_json()
    if not data or 'id' not in data:
        return jsonify({'error': 'No data'}), 400
        
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('''INSERT OR IGNORE INTO pokemon 
                       (id, name, types, height, weight, hp, attack, defense, 
                       "special-attack", "special-defense", speed) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (data['id'], data['name'], data.get('types'), data.get('height'),
                     data.get('weight'), data.get('hp'), data.get('attack'),
                     data.get('defense'), data.get('special-attack'),
                     data.get('special-defense'), data.get('speed')))
        conn.commit()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()
    return jsonify({'status': 'ok'})

@app.route('/api/pokemon', methods=['DELETE'])
def clear_pokemon():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('DELETE FROM pokemon')
    conn.commit()
    conn.close()
    # Удаляем модель, так как данные изменились кардинально
    if os.path.exists('model.pth'): os.remove('model.pth')
    if os.path.exists('scaler.pkl'): os.remove('scaler.pkl')
    return jsonify({'status': 'cleared'})

@app.route('/api/train', methods=['POST'])
def trigger_train():
    """Ручной запуск обучения"""
    success = train_model()
    global MODEL, SCALER
    if success:
        MODEL, SCALER = load_model_and_scaler()
        return jsonify({'status': 'trained'})
    return jsonify({'status': 'failed', 'reason': 'Not enough data'}), 400

@app.route('/api/similar', methods=['GET'])
def find_similar():
    """Поиск похожих покемонов"""
    global MODEL, SCALER
    
    name = request.args.get('name', '').lower().strip()
    if not name:
        return jsonify({'error': 'No name provided'}), 400
        
    if MODEL is None or SCALER is None:
        return jsonify({'error': 'Model not trained yet. Please add data and train.'}), 503
    
    # 1. Находим целевого покемона в БД
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM pokemon WHERE LOWER(name) = ?', (name,))
    target_row = cur.fetchone()
    
    if not target_row:
        conn.close()
        return jsonify({'error': 'Pokemon not found in database'}), 404
        
    target_dict = dict(target_row)
    
    # 2. Получаем всех покемонов для сравнения
    cur.execute('SELECT * FROM pokemon WHERE LOWER(name) != ?', (name,))
    all_rows = cur.fetchall()
    conn.close()
    
    if len(all_rows) == 0:
        return jsonify({'similar': []})
        
    # 3. Предобработка (временно создаем DF для одного и всех)
    import pandas as pd
    target_df = pd.DataFrame([target_dict])
    all_df = pd.DataFrame([dict(r) for r in all_rows])
    
    # Функция векторизации (упрощенная копия из trainer.py)
    def vectorize(df, scaler):
        num_cols = scaler['columns']
        num_data = df[num_cols].values.astype(float)
        num_norm = (num_data - scaler['min']) / scaler['range']
        
        types_matrix = np.zeros((len(df), len(TYPES_LIST)))
        for i, types_str in enumerate(df['types']):
            if pd.isna(types_str): continue
            p_types = [t.strip() for t in types_str.split(',')]
            for t in p_types:
                if t in TYPES_LIST:
                    types_matrix[i, TYPES_LIST.index(t)] = 1
        return np.hstack([num_norm, types_matrix])
    
    try:
        target_vec = vectorize(target_df, SCALER)
        all_vecs = vectorize(all_df, SCALER)
        
        # 4. Получаем эмбеддинги
        with torch.no_grad():
            target_tensor = torch.FloatTensor(target_vec)
            all_tensor = torch.FloatTensor(all_vecs)
            
            target_emb = MODEL.get_embedding(target_tensor).numpy().flatten()
            all_embs = MODEL.get_embedding(all_tensor).numpy()
            
        # 5. Косинусное сходство
        # norm(target) * norm(all)
        target_norm = np.linalg.norm(target_emb)
        all_norms = np.linalg.norm(all_embs, axis=1)
        
        # Избегаем деления на 0
        all_norms[all_norms == 0] = 1e-9
        
        similarities = np.dot(all_embs, target_emb) / (all_norms * target_norm)
        
        # 6. Топ 5
        top_indices = np.argsort(similarities)[::-1][:5]
        
        results = []
        for idx in top_indices:
            results.append({
                'name': all_df.iloc[idx]['name'],
                'similarity': float(similarities[idx]),
                'types': all_df.iloc[idx]['types']
            })
            
        return jsonify({'similar': results})
        
    except Exception as e:
        print(f"Error in similarity: {e}")
        return jsonify({'error': 'Processing error'}), 500

if __name__ == '__main__':
    # Переходим в директорию backend для корректных путей к файлам модели
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    init_model()
    print("🚀 Backend запущен: http://localhost:5000")
    app.run(debug=True, port=5000)