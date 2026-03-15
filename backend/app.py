# backend/app.py
# Основной Flask-сервер для классификатора инопланетных сигналов

# === 🚀 НАСТРОЙКА ПУТЕЙ — ДО ВСЕХ ИМПОРТОВ! ===
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
# === КОНЕЦ НАСТРОЙКИ ПУТЕЙ ===

# Стандартные импорты
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import hashlib
import numpy as np
from datetime import datetime

# Наши модули (✅ УБРАЛИ init_db из импорта)
from database import get_user, create_user, get_all_users, delete_user, update_last_login
from auth import generate_token, verify_token, hash_password
from model.neural_net import AlienSignalNet
from model.trainer import ModelTrainer
from model.predictor import ModelPredictor
from model.data_utils import load_dataset, restore_class_labels
from config import Config

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Глобальные объекты
trainer = None
predictor = None
model = None
_initialized = False  # Флаг для однократной инициализации


# === ✅ ИНИЦИАЛИЗАЦИЯ — БЕЗ @app.before_first_request ===
def _initialize_app():
    """Внутренняя функция инициализации (вызывается один раз)"""
    global trainer, predictor, model, _initialized
    
    if _initialized:
        return
    _initialized = True
    
    try:
        # ✅ init_db() больше не нужен — таблица создаётся автоматически при импорте database.py
        
        if not os.path.exists(Config.TRAIN_DATA_PATH):
            print(f"⚠️ Нет файла данных: {Config.TRAIN_DATA_PATH}")
            print("💡 Загрузите train.npz или обучите модель через /api/train")
            return
        
        train_data = load_dataset(Config.TRAIN_DATA_PATH)
        train_data['y'] = restore_class_labels(train_data['y'])
        
        model = AlienSignalNet(
            input_shape=train_data['x'].shape[1:],
            num_classes=len(np.unique(train_data['y']))
        )
        trainer = ModelTrainer(model, config=Config.TRAIN_CONFIG)
        predictor = ModelPredictor(model)
        print("✅ Модель инициализирована")
        
    except Exception as e:
        print(f"⚠️ Ошибка инициализации: {e}")


@app.before_request
def ensure_initialized():
    """Проверяет инициализацию перед каждым запросом"""
    if not _initialized:
        _initialize_app()


# ==================== СТАТИКА И СТРАНИЦЫ ====================

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/dashboard')
def dashboard():
    return send_from_directory('../frontend', 'dashboard.html')

@app.route('/admin')
def admin_panel():
    return send_from_directory('../frontend', 'admin.html')

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('../frontend/css', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('../frontend/js', filename)


# ==================== API: АУТЕНТИФИКАЦИЯ ====================

@app.route('/api/login', methods=['POST'])
def login():
    """
    Авторизация пользователя.
    ✅ Принимает данные из JSON тела запроса
    """
    data = request.get_json()
    print(data)
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Missing username or password'}), 400
    
    # Хэшируем пароль для сравнения с БД
    password_hash = hash_password(data['password'])
    
    # Ищем пользователя в БД
    user = get_user(data['username'], password_hash)
    
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # ✅ Обновляем last_login
    update_last_login(user['username'])
    
    # Генерируем токен сессии
    token = generate_token(user['username'], user['role'])
    
    return jsonify({
        'token': token,
        'role': user['role'],
        'name': user['name'],
        'surname': user['surname']
    })

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not verify_token(data.get('token'), required_role='admin'):
        return jsonify({'error': 'Admin access required'}), 403
    
    if not all(k in data for k in ['username', 'password', 'name', 'surname']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    success = create_user(
        username=data['username'],
        password_hash=hash_password(data['password']),
        name=data['name'],
        surname=data['surname'],
        role='user'
    )
    
    if success:
        return jsonify({'success': True, 'message': 'User created successfully'})
    else:
        return jsonify({'error': 'Username already exists'}), 409


# ==================== API: ОБУЧЕНИЕ МОДЕЛИ ====================

@app.route('/api/train', methods=['POST'])
def train_model():
    if not verify_token(request.headers.get('Authorization')):
        return jsonify({'error': 'Authentication required'}), 401
    
    global trainer, model
    
    if trainer is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    config = request.get_json() or {}
    epochs = config.get('epochs', Config.TRAIN_CONFIG['epochs'])
    batch_size = config.get('batch_size', Config.TRAIN_CONFIG['batch_size'])
    
    history = trainer.train(epochs=epochs, batch_size=batch_size, validation_split=0.25)
    model.save(Config.MODEL_SAVE_PATH)
    
    return jsonify({'history': history, 'message': f'Training completed: {epochs} epochs'})


# ==================== API: ПРОГНОЗИРОВАНИЕ И АНАЛИТИКА ====================

@app.route('/api/predict', methods=['POST'])
def predict():
    if not verify_token(request.headers.get('Authorization')):
        return jsonify({'error': 'Authentication required'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    temp_path = os.path.join(Config.TEMP_DIR, file.filename)
    file.save(temp_path)
    
    test_data = load_dataset(temp_path)
    test_data['y'] = restore_class_labels(test_data.get('y', None))
    
    metrics = predictor.evaluate(test_data['x'], test_data['y'])
    os.remove(temp_path)
    
    return jsonify(metrics)


@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    if not verify_token(request.headers.get('Authorization')):
        return jsonify({'error': 'Authentication required'}), 401
    
    global trainer, predictor
    analytics = {}
    
    # Демо-данные если модель не обучена
    if not trainer or not trainer.history:
        return jsonify({
            'accuracy_vs_epochs': {'epochs': [1,2,3], 'accuracy': [0.5, 0.7, 0.85], 'val_accuracy': [0.45, 0.65, 0.8]},
            'class_distribution': {'classes': [0,1,2,3,4], 'counts': [240, 240, 240, 240, 240]},
            'per_record_accuracy': [1,0,1,1,0],
            'top5_classes': {'classes': [0,1,2,3,4], 'counts': [100, 95, 90, 85, 80]}
        })
    
    if trainer.history:
        analytics['accuracy_vs_epochs'] = {
            'epochs': list(range(len(trainer.history['accuracy']))),
            'accuracy': trainer.history['accuracy'],
            'val_accuracy': trainer.history.get('val_accuracy', [])
        }
    
    if trainer.train_data:
        classes, counts = np.unique(trainer.train_data['y'], return_counts=True)
        analytics['class_distribution'] = {
            'classes': classes.tolist(),
            'counts': counts.tolist()
        }
    
    if predictor and predictor.last_predictions is not None:
        analytics['per_record_accuracy'] = predictor.last_predictions.tolist()
    
    if trainer.val_data:
        val_classes, val_counts = np.unique(trainer.val_data['y'], return_counts=True)
        top5_idx = np.argsort(val_counts)[-5:][::-1]
        analytics['top5_classes'] = {
            'classes': val_classes[top5_idx].tolist(),
            'counts': val_counts[top5_idx].tolist()
        }
    
    return jsonify(analytics)


# ==================== API: ПОЛЬЗОВАТЕЛИ ====================

@app.route('/api/users', methods=['GET'])
def list_users():
    if not verify_token(request.headers.get('Authorization'), required_role='admin'):
        return jsonify({'error': 'Admin access required'}), 403
    
    users = get_all_users()
    return jsonify([{
        'username': u['username'],
        'name': u['name'],
        'surname': u['surname'],
        'role': u['role'],
        'created_at': u['created_at'].isoformat() if u['created_at'] else None
    } for u in users])


# ==================== HEALTH CHECK ====================

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'database': 'connected',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/model/info')
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    return jsonify({
        'input_shape': model.input_shape,
        'num_classes': model.num_classes,
        'total_params': model.count_params(),
        'is_trained': model.is_trained
    })


# ==================== ЗАПУСК ====================

# ✅ Инициализируем при импорте модуля (без декоратора!)
_initialize_app()

if __name__ == '__main__':
    app.run(host=Config.SERVER_HOST, port=Config.SERVER_PORT, debug=Config.DEBUG)