# backend/model/predictor.py
# Модуль для инференса (предсказаний) и оценки модели
# Предоставляет метрики и аналитику для отображения в интерфейсе

import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Предиктор для классификации сигналов и расчёта метрик.
    
    Функционал:
    - Предсказание класса для одного или нескольких сигналов
    - Расчёт точности, потерь, пер-класс метрик
    - Генерация данных для визуализации (диаграммы, графики)
    """
    
    def __init__(self, model):
        """
        Инициализация предиктора.
        
        :param model: Обученная модель (AlienSignalNet)
        """
        self.model = model
        self.last_predictions: Optional[np.ndarray] = None
        self.last_probabilities: Optional[np.ndarray] = None
    
    def predict(self, x: Union[np.ndarray, List]) -> np.ndarray:
        """
        Предсказывает классы для входных сигналов.
        
        :param x: Сигналы формы (n_samples, ...) или список
        :return: Массив предсказанных классов
        """
        if isinstance(x, list):
            x = np.array(x)
        
        predictions = self.model.predict(x)
        self.last_predictions = predictions
        return predictions
    
    def predict_proba(self, x: Union[np.ndarray, List]) -> np.ndarray:
        """
        Возвращает вероятности для каждого класса.
        
        :param x: Входные сигналы
        :return: Массив вероятностей формы (n_samples, n_classes)
        """
        if isinstance(x, list):
            x = np.array(x)
        
        probas = self.model.predict_proba(x)
        self.last_probabilities = probas
        return probas
    
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """
        Полная оценка модели на тестовых данных.
        
        :param x: Признаки тестовой выборки
        :param y: Истинные метки
        :return: Словарь со всеми метриками
        """
        # Получаем предсказания и вероятности
        predictions = self.predict(x)
        probabilities = self.predict_proba(x)
        
        # Базовые метрики
        if y.ndim == 1:
            # Целочисленные метки
            accuracy = np.mean(predictions == y)
            
            # Потери (кросс-энтропия)
            epsilon = 1e-15
            clipped_probs = np.clip(probabilities, epsilon, 1 - epsilon)
            loss = -np.mean(np.log(clipped_probs[np.arange(len(y)), y] + epsilon))
            
            # Per-class точность
            per_class = {}
            for cls in np.unique(y):
                mask = y == cls
                if np.sum(mask) > 0:
                    per_class[int(cls)] = float(np.mean(predictions[mask] == cls))
        else:
            # One-hot encoded метки
            y_classes = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == y_classes)
            loss = 0  # Упрощённо
            per_class = {}
        
        # Точность для каждой записи (для диаграммы)
        per_record_accuracy = (predictions == (y if y.ndim == 1 else np.argmax(y, axis=1))).astype(float)
        
        return {
            'accuracy': float(accuracy),
            'loss': float(loss),
            'per_class_accuracy': per_class,
            'per_record_accuracy': per_record_accuracy.tolist(),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'n_samples': len(x)
        }
    
    def get_class_distribution(self, y: np.ndarray) -> Dict[str, List]:
        """
        Возвращает распределение классов в данных.
        
        :param y: Метки классов
        :return: Словарь с классами и их количествами
        """
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        
        classes, counts = np.unique(y, return_counts=True)
        
        return {
            'classes': classes.tolist(),
            'counts': counts.tolist(),
            'total': int(np.sum(counts))
        }
    
    def get_top_classes(self, y: np.ndarray, top_n: int = 5) -> Dict[str, List]:
        """
        Возвращает топ-N наиболее частых классов.
        
        :param y: Метки классов
        :param top_n: Количество классов в топе
        :return: Словарь с топ-классами и их частотами
        """
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        
        classes, counts = np.unique(y, return_counts=True)
        
        # Сортируем по убыванию частоты
        sorted_idx = np.argsort(counts)[::-1]
        
        # Берём топ-N
        top_idx = sorted_idx[:min(top_n, len(sorted_idx))]
        
        return {
            'classes': classes[top_idx].tolist(),
            'counts': counts[top_idx].tolist(),
            'n_requested': top_n
        }
    
    def get_confusion_data(self, y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Dict:
        """
        Возвращает данные для матрицы ошибок (упрощённо).
        
        :param y_true: Истинные метки
        :param y_pred: Предсказанные метки
        :return: Словарь с данными для визуализации
        """
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        
        # Простая матрица ошибок
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            confusion[t, p] += 1
        
        return {
            'classes': classes.tolist(),
            'matrix': confusion.tolist(),
            'n_classes': n_classes
        }