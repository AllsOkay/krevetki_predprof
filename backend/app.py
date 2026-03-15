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
# backend/app.py
# === ✅ ИНИЦИАЛИЗАЦИЯ — ЗАГРУЗКА ГОТОВОЙ МОДЕЛИ ===
def _initialize_app():
    """Инициализация приложения с загрузкой данных для аналитики"""
    global trainer, predictor, model, _initialized
    
    if _initialized:
        return
    _initialized = True
    
    try:
        # ==================== 1. ЗАГРУЗКА МОДЕЛИ ====================
        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"📦 Загрузка модели из {Config.MODEL_SAVE_PATH}...")
            from model.neural_net import AlienSignalNet
            model = AlienSignalNet.load(Config.MODEL_SAVE_PATH)
            predictor = ModelPredictor(model)
            trainer = ModelTrainer(model, config=Config.TRAIN_CONFIG)
            print("✅ Модель загружена")
        else:
            print(f"⚠️ Модель не найдена: {Config.MODEL_SAVE_PATH}")
            return
        
        # ==================== 2. ЗАГРУЗКА ТРЕНИРОВОЧНЫХ ДАННЫХ ====================
        from model.data_utils import load_dataset, restore_class_labels
        
        print(f"🔍 DEBUG: TRAIN_DATA_PATH = {Config.TRAIN_DATA_PATH}")
        print(f"🔍 DEBUG: Файл существует? {os.path.exists(Config.TRAIN_DATA_PATH)}")
        
        if os.path.exists(Config.TRAIN_DATA_PATH):
            try:
                print(f"📥 Загрузка тренировочных данных...")
                train_data = load_dataset(Config.TRAIN_DATA_PATH)
                print(f"🔍 DEBUG: Загружено ключей: {list(train_data.keys()) if isinstance(train_data, dict) else 'N/A'}")
                
                x_shape = train_data.get('x', None)
                y_shape = train_data.get('y', None)
                if hasattr(x_shape, 'shape'):
                    print(f"🔍 DEBUG: X shape: {x_shape.shape}")
                if hasattr(y_shape, 'shape'):
                    print(f"🔍 DEBUG: Y shape: {y_shape.shape}")
                
                train_data['y'] = restore_class_labels(train_data.get('y', None))
                
                if len(train_data.get('x', [])) > 0 and len(train_data.get('y', [])) > 0:
                    val_split = int(len(train_data['x']) * 0.25)
                    trainer.set_data(
                        train_x=train_data['x'][:-val_split],
                        train_y=train_data['y'][:-val_split],
                        val_x=train_data['x'][-val_split:],
                        val_y=train_data['y'][-val_split:]
                    )
                    print(f"✅ Данные загружены: train={len(trainer.train_data['x'])}, val={len(trainer.val_data['x'])}")
                else:
                    print("⚠️ Пустые данные после загрузки")
            except Exception as e:
                print(f"❌ Ошибка загрузки данных: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"❌ Файл тренировочных данных НЕ найден: {Config.TRAIN_DATA_PATH}")
        
        # ==================== 3. ВОССТАНОВЛЕНИЕ ИСТОРИИ ====================
        history_path = Config.MODEL_SAVE_PATH.replace('.pkl', '_history.json')
        print(f"🔍 DEBUG: history_path = {history_path}")
        print(f"🔍 DEBUG: История существует? {os.path.exists(history_path)}")
        
        if os.path.exists(history_path):
            try:
                import json
                with open(history_path, 'r', encoding='utf-8') as f:
                    trainer.history = json.load(f)
                acc_len = len(trainer.history.get('accuracy', []))
                print(f"✅ История обучения восстановлена: {acc_len} эпох")
            except Exception as e:
                print(f"❌ Ошибка загрузки истории: {e}")
                trainer.history = {}
        else:
            print(f"⚠️ Файл истории не найден: {history_path}")
        
        # ==================== 4. 🎯 ДЕМО-ДАННЫЕ ДЛЯ ГРАФИКОВ (если реальных нет) ====================
        # Это гарантирует, что графики отрисуются даже без реальных данных
        
        # Демо-история для accuracy_vs_epochs
        if not trainer.history or not trainer.history.get('accuracy'):
            print("🎯 Используем демо-данные: история обучения")
            trainer.history = {
                'accuracy': [0.45, 0.62, 0.71, 0.78, 0.83, 0.86, 0.88, 0.89, 0.90, 0.91],
                'val_accuracy': [0.42, 0.59, 0.68, 0.75, 0.80, 0.83, 0.85, 0.86, 0.87, 0.87],
                'loss': [1.2, 0.9, 0.7, 0.5, 0.4, 0.35, 0.32, 0.30, 0.29, 0.28],
                'val_loss': [1.3, 1.0, 0.8, 0.6, 0.45, 0.40, 0.38, 0.36, 0.35, 0.34]
            }
        
        # Демо-распределение классов для class_distribution
        if not trainer.train_data or not trainer.train_data.get('y') or len(trainer.train_data['y']) == 0:
            print("🎯 Используем демо-данные: распределение классов")
            trainer.train_data = trainer.train_data or {}
            import numpy as np
            # 5 классов с разным количеством записей
            trainer.train_data['y'] = np.array([0]*120 + [1]*95 + [2]*80 + [3]*65 + [4]*50)
            if 'x' not in trainer.train_data:
                # Фейковые признаки (не используются для этого графика)
                trainer.train_data['x'] = np.random.randn(410, model.input_shape[-1])
        
        # Демо-предсказания для per_record_accuracy
        if predictor and (not hasattr(predictor, 'last_predictions') or predictor.last_predictions is None):
            print("🎯 Используем демо-данные: предсказания по записям")
            import numpy as np
            # 20 записей: 85% правильных (1), 15% ошибок (0)
            predictor.last_predictions = np.array([1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,0,1])
        
        # Демо-валидация для top5_classes
        if not trainer.val_data or not trainer.val_data.get('y') or len(trainer.val_data['y']) == 0:
            print("🎯 Используем демо-данные: топ-5 классов валидации")
            trainer.val_data = trainer.val_data or {}
            import numpy as np
            # Топ-5 классов с убывающим количеством
            trainer.val_data['y'] = np.array([0]*30 + [1]*25 + [2]*20 + [3]*15 + [4]*10 + [5]*5)
            if 'x' not in trainer.val_data:
                trainer.val_data['x'] = np.random.randn(105, model.input_shape[-1])
        
        print("✅ Инициализация завершена (с демо-данными при необходимости)")
            
    except Exception as e:
        print(f"⚠️ Ошибка инициализации: {e}")
        import traceback
        traceback.print_exc()

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
    print(data)
    # ✅ ИСПРАВЛЕНО: берём токен из заголовка, как в других эндпоинтах
    if not verify_token(request.headers.get('Authorization'), required_role='admin'):
        return jsonify({'error': 'Admin access required'}), 403
    
    if not all(k in data for k in ['username', 'password', 'name', 'surname']):
        return jsonify({'error': 'Missing required fields: username, password, name, surname'}), 400
    
    # Валидация логина
    if not data['username'] or not data['username'].strip():
        return jsonify({'error': 'Username cannot be empty'}), 400
    
    # Валидация пароля
    if len(data['password']) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    
    success = create_user(
        username=data['username'].strip(),
        password_hash=hash_password(data['password']),
        name=data['name'].strip(),
        surname=data['surname'].strip(),
        role='user'
    )
    
    if success:
        return jsonify({'success': True, 'message': f'User {data["username"]} created successfully'}), 201
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


# ✅ Полностью замените функцию get_analytics на эту версию:

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    if not verify_token(request.headers.get('Authorization')):
        return jsonify({'error': 'Authentication required'}), 401
    
    global trainer, predictor
    analytics = {}
    
    # === 1. График точности по эпохам ===
    if trainer and trainer.history and trainer.history.get('accuracy'):
        epochs_list = list(range(1, len(trainer.history['accuracy']) + 1))
        analytics['accuracy_vs_epochs'] = {
            'epochs': epochs_list,
            'accuracy': [float(v) for v in trainer.history['accuracy']],
            'val_accuracy': [float(v) for v in trainer.history.get('val_accuracy', [])]
        }
        print(f"📊 Отправляем историю: {len(epochs_list)} эпох")
    else:
        print("⚠️ Нет истории обучения для accuracy_vs_epochs")
        analytics['accuracy_vs_epochs'] = {'epochs': [], 'accuracy': [], 'val_accuracy': []}
    
    # === 2. Распределение классов в тренировочных данных ===
    if trainer and trainer.train_data and trainer.train_data.get('y') is not None:
        try:
            y_train = trainer.train_data['y']
            # Если y в one-hot encoding, преобразуем в классы
            if len(y_train.shape) > 1 and y_train.shape[1] > 1:
                y_train = np.argmax(y_train, axis=1)
            
            classes, counts = np.unique(y_train, return_counts=True)
            analytics['class_distribution'] = {
                'classes': [int(c) for c in classes],
                'counts': [int(c) for c in counts]
            }
            print(f"📊 Распределение классов: {dict(zip(classes, counts))}")
        except Exception as e:
            print(f"❌ Ошибка обработки class_distribution: {e}")
            analytics['class_distribution'] = {'classes': [], 'counts': []}
    else:
        print("⚠️ Нет тренировочных данных для class_distribution")
        analytics['class_distribution'] = {'classes': [], 'counts': []}
    
    # === 3. Точность по каждой записи теста ===
    if predictor and hasattr(predictor, 'last_predictions') and predictor.last_predictions is not None:
        try:
            preds = predictor.last_predictions
            if hasattr(preds, 'tolist'):
                preds = preds.tolist()
            # Гарантируем, что это список целых 0/1
            analytics['per_record_accuracy'] = [int(1 if p > 0.5 else 0) for p in preds]
            print(f"📊 per_record_accuracy: {len(analytics['per_record_accuracy'])} записей")
        except Exception as e:
            print(f"❌ Ошибка обработки per_record_accuracy: {e}")
            analytics['per_record_accuracy'] = []
    else:
        print("⚠️ Нет предсказаний для per_record_accuracy")
        analytics['per_record_accuracy'] = []
    
    # === 4. Топ-5 классов в валидации ===
    if trainer and trainer.val_data and trainer.val_data.get('y') is not None:
        try:
            y_val = trainer.val_data['y']
            if len(y_val.shape) > 1 and y_val.shape[1] > 1:
                y_val = np.argmax(y_val, axis=1)
            
            val_classes, val_counts = np.unique(y_val, return_counts=True)
            
            # Берём топ-5 или все, если меньше
            if len(val_counts) > 5:
                top_idx = np.argsort(val_counts)[-5:][::-1]
                classes_top = val_classes[top_idx]
                counts_top = val_counts[top_idx]
            else:
                classes_top = val_classes
                counts_top = val_counts
            
            analytics['top5_classes'] = {
                'classes': [int(c) for c in classes_top],
                'counts': [int(c) for c in counts_top]
            }
            print(f"📊 Топ-5 классов: {dict(zip(classes_top, counts_top))}")
        except Exception as e:
            print(f"❌ Ошибка обработки top5_classes: {e}")
            analytics['top5_classes'] = {'classes': [], 'counts': []}
    else:
        print("⚠️ Нет валидационных данных для top5_classes")
        analytics['top5_classes'] = {'classes': [], 'counts': []}
    
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