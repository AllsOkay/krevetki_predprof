# backend/model/trainer.py
# Модуль для обучения нейросети
# Реализует цикл обучения с валидацией и сохранением лучшей модели

import numpy as np
from typing import Dict, List, Optional, Tuple
from config import Config
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Трейнер для обучения нейросети классификации сигналов.
    
    Особенности:
    - Поддержка early stopping для предотвращения переобучения
    - Сохранение лучшей модели по валидационной точности
    - Логирование метрик после каждой эпохи
    - Мини-батч обработка для эффективного использования памяти
    """
    
    def __init__(self, model, config: Optional[Dict] = None):
        """
        Инициализация трейнера.
        
        :param model: Экземпляр нейросети (AlienSignalNet)
        :param config: Словарь с параметрами обучения
        """
        self.model = model
        self.config = config or Config.TRAIN_CONFIG.copy()
        
        # История обучения для аналитики
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Данные для обучения (загружаются отдельно)
        self.train_data = None
        self.val_data = None
        
        # Для early stopping
        self.best_val_accuracy = 0
        self.patience_counter = 0
        
    def set_data(self, train_x: np.ndarray, train_y: np.ndarray,
                 val_x: Optional[np.ndarray] = None, 
                 val_y: Optional[np.ndarray] = None) -> None:
        """
        Устанавливает данные для обучения и валидации.
        
        :param train_x: Признаки обучающей выборки
        :param train_y: Метки обучающей выборки
        :param val_x: Признаки валидационной выборки (опционально)
        :param val_y: Метки валидационной выборки (опционально)
        """
        self.train_data = {'x': train_x, 'y': train_y}
        
        if val_x is not None and val_y is not None:
            self.val_data = {'x': val_x, 'y': val_y}
            logger.info(f"✓ Валидационные данные: {len(val_x)} образцов")
    
    def _create_batches(self, x: np.ndarray, y: np.ndarray, 
                        batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Создаёт мини-батчи из данных.
        
        :param x: Признаки
        :param y: Метки
        :param batch_size: Размер одного батча
        :return: Список кортежей (batch_x, batch_y)
        """
        n_samples = len(x)
        batches = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batches.append((x[start_idx:end_idx], y[start_idx:end_idx]))
        
        return batches
    
    def _train_epoch(self, batches: List[Tuple[np.ndarray, np.ndarray]], 
                     shuffle: bool = True) -> Dict[str, float]:
        """
        Проводит одну эпоху обучения.
        
        :param batches: Список мини-батчей
        :param shuffle: Перемешивать ли батчи
        :return: Словарь с метриками за эпоху
        """
        if shuffle:
            np.random.shuffle(batches)
        
        epoch_losses = []
        epoch_accuracies = []
        
        for batch_x, batch_y in batches:
            # Один шаг градиентного спуска
            loss, acc = self.model.train_step(batch_x, batch_y)
            epoch_losses.append(loss)
            epoch_accuracies.append(acc)
        
        return {
            'loss': np.mean(epoch_losses),
            'accuracy': np.mean(epoch_accuracies)
        }
    
    def _evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Оценивает модель на данных (без обучения).
        
        :param x: Признаки
        :param y: Истинные метки
        :return: Словарь с метриками
        """
        return self.model.evaluate(x, y)
    
    def train(self, epochs: Optional[int] = None, 
              batch_size: Optional[int] = None,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Запускает полный цикл обучения модели.
        
        :param epochs: Количество эпох (по умолчанию из config)
        :param batch_size: Размер мини-батча (по умолчанию из config)
        :param verbose: Выводить ли прогресс в консоль
        :return: История обучения (self.history)
        """
        epochs = epochs or self.config['epochs']
        batch_size = batch_size or self.config['batch_size']
        patience = self.config.get('early_stopping_patience', 5)
        save_best = self.config.get('save_best_only', True)
        
        if self.train_data is None:
            raise ValueError("Training data not set. Call set_data() first.")
        
        # Создаём батчи для обучения
        train_batches = self._create_batches(
            self.train_data['x'], 
            self.train_data['y'], 
            batch_size
        )
        
        logger.info(f"🚀 Начало обучения: {epochs} эпох, {len(train_batches)} батчей/эпоху")
        
        for epoch in range(epochs):
            # === ОБУЧЕНИЕ ===
            train_metrics = self._train_epoch(train_batches)
            
            # === ВАЛИДАЦИЯ ===
            val_metrics = {}
            if self.val_data is not None:
                val_metrics = self._evaluate(
                    self.val_data['x'], 
                    self.val_data['y']
                )
            
            # === СОХРАНЕНИЕ ИСТОРИИ ===
            self.history['loss'].append(train_metrics['loss'])
            self.history['accuracy'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics.get('loss', 0))
            self.history['val_accuracy'].append(val_metrics.get('accuracy', 0))
            
            # === ЛОГИРОВАНИЕ ===
            if verbose:
                val_info = ""
                if val_metrics:
                    val_info = f" | Val: {val_metrics['accuracy']:.4f} loss: {val_metrics['loss']:.4f}"
                print(f"Epoch {epoch+1}/{epochs} "
                      f"| Train acc: {train_metrics['accuracy']:.4f} loss: {train_metrics['loss']:.4f}"
                      f"{val_info}")
            
            # === EARLY STOPPING ===
            if self.val_data is not None and save_best:
                current_val_acc = val_metrics.get('accuracy', 0)
                
                if current_val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = current_val_acc
                    self.patience_counter = 0
                    # Сохраняем лучшую модель
                    if verbose:
                        print(f"  💾 Новая лучшая модель! Сохраняем...")
                    self.model.save(Config.MODEL_SAVE_PATH)
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= patience:
                        if verbose:
                            print(f"  ⏹ Early stopping: нет улучшений {patience} эпох")
                        break
        
        # Сохраняем финальную модель если не сохраняли лучшую
        if not save_best or self.val_data is None:
            self.model.save(Config.MODEL_SAVE_PATH)
            logger.info(f"✓ Модель сохранена: {Config.MODEL_SAVE_PATH}")
        
        self.model.is_trained = True
        logger.info(f"✅ Обучение завершено! Лучшая вал. точность: {self.best_val_accuracy:.4f}")
        # В конце метода train() добавьте:
        print(f"✅ Обучение завершено. История: {len(self.history['accuracy'])} эпох")
        print(f"📊 Финальная точность: {self.history['accuracy'][-1]:.4f}")
        return self.history
    
    def get_training_plot_data(self) -> Dict[str, List]:
        """
        Возвращает данные для построения графика обучения.
        
        :return: Словарь с данными для графиков точности и потерь
        """
        return {
            'epochs': list(range(1, len(self.history['accuracy']) + 1)),
            'train_accuracy': self.history['accuracy'],
            'val_accuracy': self.history['val_accuracy'],
            'train_loss': self.history['loss'],
            'val_loss': self.history['val_loss']
        }