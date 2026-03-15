# backend/model/predictor.py
# Модуль для инференса (предсказаний) и оценки модели

import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Предиктор для классификации сигналов и расчёта метрик.
    """

    def __init__(self, model):
        self.model = model
        self.last_predictions: Optional[np.ndarray] = None
        self.last_probabilities: Optional[np.ndarray] = None

    def predict(self, x: Union[np.ndarray, List]) -> np.ndarray:
        if isinstance(x, list):
            x = np.array(x)
        
        predictions = self.model.predict(x)
        self.last_predictions = predictions
        return predictions

    def predict_proba(self, x: Union[np.ndarray, List]) -> np.ndarray:
        if isinstance(x, list):
            x = np.array(x)
        
        probas = self.model.predict_proba(x)
        self.last_probabilities = probas
        return probas

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict:
        predictions = self.predict(x)
        probabilities = self.predict_proba(x)
        
        n_classes = self.model.num_classes
        
        # ✅ ИСПРАВЛЕНИЕ: Нормализация меток классов
        if y.ndim == 1:
            unique_labels = np.unique(y)
            
            # Если метки выходят за диапазон модели - маппим их
            if np.any(y >= n_classes) or np.any(y < 0):
                logger.warning(f"Метки классов {unique_labels} вне диапазона модели [0, {n_classes-1}]")
                # Создаём маппинг: старые метки -> новые (по модулю n_classes)
                y_mapped = y % n_classes
                accuracy = np.mean(predictions == y_mapped)
                
                # Потери считаем по маппированным меткам
                epsilon = 1e-15
                clipped_probs = np.clip(probabilities, epsilon, 1 - epsilon)
                loss = -np.mean(np.log(clipped_probs[np.arange(len(y)), y_mapped] + epsilon))
                
                # Per-class точность по маппированным меткам
                per_class = {}
                for cls in np.unique(y_mapped):
                    mask = y_mapped == cls
                    if np.sum(mask) > 0:
                        per_class[int(cls)] = float(np.mean(predictions[mask] == cls))
            else:
                # Стандартный расчёт
                accuracy = np.mean(predictions == y)
                
                epsilon = 1e-15
                clipped_probs = np.clip(probabilities, epsilon, 1 - epsilon)
                loss = -np.mean(np.log(clipped_probs[np.arange(len(y)), y] + epsilon))
                
                per_class = {}
                for cls in np.unique(y):
                    mask = y == cls
                    if np.sum(mask) > 0:
                        per_class[int(cls)] = float(np.mean(predictions[mask] == cls))
        else:
            y_classes = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == y_classes)
            loss = 0
            per_class = {}
        
        per_record_accuracy = (predictions == (y_mapped if 'y_mapped' in locals() else (y if y.ndim == 1 else np.argmax(y, axis=1)))).astype(float)
        
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
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        
        classes, counts = np.unique(y, return_counts=True)
        
        return {
            'classes': classes.tolist(),
            'counts': counts.tolist(),
            'total': int(np.sum(counts))
        }

    def get_top_classes(self, y: np.ndarray, top_n: int = 5) -> Dict[str, List]:
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        
        classes, counts = np.unique(y, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]
        top_idx = sorted_idx[:min(top_n, len(sorted_idx))]
        
        return {
            'classes': classes[top_idx].tolist(),
            'counts': counts[top_idx].tolist(),
            'n_requested': top_n
        }