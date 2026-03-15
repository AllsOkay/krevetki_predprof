# ✅ СТАЛО:
# Глобальный флаг для однократной инициализации
_initialized = False

@app.before_request
def initialize():
    """Инициализация БД и загрузка модели (выполняется один раз)"""
    global trainer, predictor, model, _initialized
    
    # Если уже инициализировано — выходим
    if _initialized:
        return
    
    _initialized = True  # Помечаем что инициализация началась
    
    # === Код инициализации ===
    from database import init_db
    from model.data_utils import load_dataset, restore_class_labels
    from model.neural_net import AlienSignalNet
    from model.trainer import ModelTrainer
    from model.predictor import ModelPredictor
    
    init_db(Config.DATABASE_PATH)
    
    # Проверяем наличие файла с данными
    if not os.path.exists(Config.TRAIN_DATA_PATH):
        print(f"⚠️ Файл данных не найден: {Config.TRAIN_DATA_PATH}")
        print("💡 Загрузите train.npz или используйте /api/train для обучения")
        return
    
    try:
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
        print(f"⚠️ Ошибка инициализации модели: {e}")