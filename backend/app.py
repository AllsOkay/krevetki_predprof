"""
Flask сервер с поддержкой модульной архитектуры нейросетей.
"""
from flask import Flask, request, jsonify, send_from_directory
import os
import sys
from pathlib import Path

# Добавляем директорию backend в sys.path для корректных импортов
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Импорты без префикса 'backend.'
from database import PokemonDatabase
from models.cnn.predictor import CNNPredictor
from config import CNN_CONFIG, TRAINING_CONFIG

app = Flask(__name__, static_folder='../frontend', static_url_path='')

# Инициализация компонентов
db = PokemonDatabase()
cnn_predictor = CNNPredictor(db)

# ==================== Frontend Routes ====================

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/css.css')
def css():
    return send_from_directory(app.static_folder, 'css.css')

@app.route('/script.js')
def js():
    return send_from_directory(app.static_folder, 'script.js')

# ==================== Pokemon API Routes ====================

@app.route('/api/pokemon', methods=['GET'])
def get_pokemon():
    """Получить всех покемонов из БД."""
    return jsonify(db.get_all_pokemon())

@app.route('/api/pokemon', methods=['POST'])
def add_pokemon():
    """Добавить покемона в БД (с URL спрайта)."""
    data = request.get_json()
    if not data or 'id' not in data or 'name' not in data:
        return jsonify({'error': 'Missing id or name'}), 400
    
    # Добавляем URL спрайта если есть
    if 'sprite_url' not in data:
        import requests
        try:
            resp = requests.get(f"https://pokeapi.co/api/v2/pokemon/{data['id']}")
            if resp.ok:
                poke_data = resp.json()
                data['sprite_url'] = poke_data.get('sprites', {}).get('front_default')
        except:
            pass
    
    success = db.add_pokemon(data)
    return jsonify({'status': 'ok' if success else 'failed'}), 200 if success else 500

@app.route('/api/pokemon', methods=['DELETE'])
def clear_pokemon():
    """Очистить все данные."""
    db.clear_all()
    
    if CNN_CONFIG['model_path'].exists():
        CNN_CONFIG['model_path'].unlink()
    
    return jsonify({'status': 'cleared'})

# ==================== CNN Module Routes ====================

@app.route('/api/cnn/train', methods=['POST'])
def train_cnn():
    """Запустить обучение CNN модели."""
    from models.cnn.trainer import CNNTrainer
    
    trainer = CNNTrainer(db)
    result = trainer.train(verbose=True)
    
    return jsonify(result)

@app.route('/api/cnn/similar', methods=['GET'])
def find_similar_cnn():
    """
    Найти визуально похожих покемонов.
    
    Query params:
        name: имя покемона (обязательно)
        k: количество результатов (по умолчанию 5)
    """
    name = request.args.get('name', '').strip().lower()
    top_k = int(request.args.get('k', 5))
    
    if not name:
        return jsonify({'error': 'Parameter "name" is required'}), 400
    
    target = db.get_pokemon_by_name(name)
    if not target:
        return jsonify({'error': f'Pokemon "{name}" not found in database'}), 404
    
    similar = cnn_predictor.find_similar(name, top_k=top_k)
    
    for item in similar:
        poke = db.get_pokemon_by_name(item['name'])
        if poke:
            item['sprite_url'] = poke.get('sprite_url')
    
    return jsonify({
        'target': {
            'name': target['name'],
            'sprite_url': target.get('sprite_url')
        },
        'similar': similar
    })

    # Добавить после CNN routes:

# ==================== RNN Module Routes ====================


@app.route('/api/rnn/train', methods=['POST'])
def train_rnn():
    """Запустить обучение RNN модели."""
    # ✅ Импорты без префикса 'backend.'
    from models.rnn.trainer import RNNTrainer
    
    trainer = RNNTrainer(db)
    result = trainer.train(verbose=True)
    
    return jsonify(result)

@app.route('/api/rnn/classify', methods=['GET'])
def classify_pokemon_rnn():
    """
    Классифицировать покемона по категории силы.
    """
    name = request.args.get('name', '').strip().lower()
    
    if not name:
        return jsonify({'error': 'Parameter "name" is required'}), 400
    
    # ✅ Импорты без префикса 'backend.'
    from models.rnn.predictor import RNNPredictor
    predictor = RNNPredictor(db)
    
    result = predictor.classify(name)
    
    if not result:
        return jsonify({'error': f'Pokemon "{name}" not found'}), 404
    
    pokemon = db.get_pokemon_by_name(name)
    
    return jsonify({
        'pokemon': {
            'name': pokemon['name'],
            'bst': result.get('bst', 0),
            'sprite_url': pokemon.get('sprite_url')
        },
        'result': result
    })

@app.route('/api/rnn/distribution', methods=['GET'])
def get_rnn_distribution():
    """Получить распределение покемонов по классам силы."""
    # ✅ Импорты без префикса 'backend.'
    from models.rnn.predictor import RNNPredictor
    predictor = RNNPredictor(db)
    
    distribution = predictor.get_distribution()
    
    return jsonify(distribution)


# Добавить после RNN routes:

# ==================== Autoencoder Module Routes ====================

@app.route('/api/autoencoder/train', methods=['POST'])
def train_autoencoder():
    """Запустить обучение автоэнкодера."""
    from models.autoencoder.trainer import AETrainer
    
    trainer = AETrainer(db)
    result = trainer.train(verbose=True)
    
    return jsonify(result)

@app.route('/api/autoencoder/ordinariness', methods=['GET'])
def get_ordinariness():
    """
    Получить "обычность" покемона в процентах.
    
    Query params:
        name: имя покемона (обязательно)
    
    Response:
    {
        "pokemon": {"name": "pikachu", "sprite_url": "..."},
        "result": {
            "ordinariness": 75.3,
            "reconstruction_error": 0.0234,
            "description": "🔵 Обычный (типичные характеристики)"
        }
    }
    """
    name = request.args.get('name', '').strip().lower()
    
    if not name:
        return jsonify({'error': 'Parameter "name" is required'}), 400
    
    from models.autoencoder.predictor import AEPredictor
    predictor = AEPredictor(db)
    
    result = predictor.get_ordinariness(name)
    
    if not result:
        return jsonify({'error': f'Pokemon "{name}" not found'}), 404
    
    pokemon = db.get_pokemon_by_name(name)
    
    return jsonify({
        'pokemon': {
            'name': pokemon['name'],
            'sprite_url': pokemon.get('sprite_url')
        },
        'result': result
    })

@app.route('/api/autoencoder/statistics', methods=['GET'])
def get_ae_statistics():
    """Получить статистику по обычности всех покемонов."""
    from models.autoencoder.predictor import AEPredictor
    predictor = AEPredictor(db)
    
    stats = predictor.get_statistics()
    
    return jsonify(stats)


# Добавить после Autoencoder routes:

# ==================== MLP Module Routes ====================

@app.route('/api/mlp/train', methods=['POST'])
def train_mlp():
    """Запустить обучение MLP модели."""
    from models.mlp.trainer import MLPTrainer
    
    trainer = MLPTrainer(db)
    result = trainer.train(verbose=True)
    
    return jsonify(result)

@app.route('/api/mlp/predict', methods=['GET'])
def predict_win_probability():
    """
    Предсказать вероятность победы покемона.
    
    Query params:
        hp: HP характеристика (обязательно)
        atk: ATK характеристика (обязательно)
        def: DEF характеристика (обязательно)
        spd: SPD характеристика (обязательно)
    
    Response:
    {
        "input": {"hp": 50, "atk": 60, "def": 50, "spd": 70},
        "result": {
            "win_probability": 65.3,
            "bst": 230,
            "classification": "🔵 Выше среднего",
            "description": "Хорошие шансы..."
        }
    }
    """
    try:
        hp = float(request.args.get('hp', 0))
        atk = float(request.args.get('atk', 0))
        def_ = float(request.args.get('def', 0))
        spd = float(request.args.get('spd', 0))
    except ValueError:
        return jsonify({'error': 'Invalid stat values'}), 400
    
    if hp < 0 or atk < 0 or def_ < 0 or spd < 0:
        return jsonify({'error': 'Stats must be non-negative'}), 400
    
    from models.mlp.predictor import MLPPredictor
    predictor = MLPPredictor(db)
    
    result = predictor.predict_win_probability(hp, atk, def_, spd)
    
    if not result:
        return jsonify({'error': 'Prediction failed'}), 500
    
    return jsonify({
        'input': {
            'hp': hp,
            'atk': atk,
            'def': def_,
            'spd': spd
        },
        'result': result
    })

@app.route('/api/mlp/stats', methods=['GET'])
def get_mlp_stats():
    """Получить статистику по предсказаниям MLP."""
    from models.mlp.predictor import MLPPredictor
    predictor = MLPPredictor(db)
    
    # Пример статистики
    return jsonify({
        'model_loaded': predictor.model_loaded,
        'input_features': ['HP', 'ATK', 'DEF', 'SPD'],
        'output': 'Win Probability (0-100%)'
    })

# ==================== Health Check ====================

@app.route('/api/health')
def health():
    """Проверка работоспособности сервера."""
    return jsonify({
        'status': 'ok',
        'database': 'connected',
        'cnn_model': 'loaded' if cnn_predictor.model_loaded else 'not_loaded',
        'pokemon_count': db.get_pokemon_count(),
        'pokemon_with_sprites': db.get_pokemon_with_sprites_count(),
        'min_required_for_cnn': CNN_CONFIG['min_samples']
    })

if __name__ == '__main__':
    from config import MODELS_DIR
    for model_type in ['cnn', 'rnn', 'mlp', 'autoencoder']:
        (MODELS_DIR / model_type).mkdir(parents=True, exist_ok=True)
    
    print("🚀 Backend запущен: http://localhost:5000")
    print("📊 Health check: http://localhost:5000/api/health")
    print("🖼️  CNN similar: http://localhost:5000/api/cnn/similar?name=pikachu")
    
    app.run(debug=True, port=5000)