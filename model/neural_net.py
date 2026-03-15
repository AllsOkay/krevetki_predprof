# backend/model/neural_net.py
# Реализация нейронной сети для классификации инопланетных сигналов
# ВАЖНО: Запрещено использовать готовые модели (TensorFlow/Keras pre-trained)
# Реализуем простую полносвязную сеть с возможностью расширения до CNN

import numpy as np
from typing import Tuple, List, Optional
import pickle
import os

# backend/model/neural_net.py
# См. полную версию в предыдущем ответе
# Ключевые импорты для других модулей:


__all__ = ['AlienSignalNet']

class AlienSignalNet:
    """
    Кастомная нейронная сеть для классификации радиосигналов.
    
    Архитектура (базовая):
    - Вход: вектор признаков сигнала (после предобработки wav)
    - Скрытые слои: 2-3 полносвязных слоя с ReLU
    - Выход: softmax для многоклассовой классификации
    
    В будущем можно расширить до 1D-CNN для работы с временными рядами.
    """
    
    def __init__(self, input_shape: Tuple[int], num_classes: int, 
                 hidden_layers: List[int] = [128, 64], 
                 learning_rate: float = 0.001):
        """
        Инициализация сети.
        
        :param input_shape: Форма входных данных (например, (1024,) для спектрограммы)
        :param num_classes: Количество классов (цивилизаций)
        :param hidden_layers: Список размеров скрытых слоёв
        :param learning_rate: Скорость обучения
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.is_trained = False
        
        # Инициализация весов (Xavier initialization для лучшей сходимости)
        self.weights = []
        self.biases = []
        
        # Первый слой: вход -> первый скрытый
        in_size = np.prod(input_shape)
        for i, out_size in enumerate(hidden_layers + [num_classes]):
            # Xavier init: W ~ N(0, sqrt(2/(fan_in + fan_out)))
            scale = np.sqrt(2.0 / (in_size + out_size))
            W = np.random.randn(in_size, out_size) * scale
            b = np.zeros((1, out_size))
            
            self.weights.append(W)
            self.biases.append(b)
            in_size = out_size  # Для следующего слоя
        
        # Для хранения градиентов при обратном распространении
        self.gradients = None
        self.activations = None
        
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """Функция активации ReLU"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Производная ReLU"""
        return (x > 0).astype(float)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Стабильная реализация softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _cross_entropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Кросс-энтропийная функция потерь"""
        # Добавляем epsilon для численной стабильности
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # One-hot encoding если нужно
        if y_true.ndim == 1:
            y_true_onehot = np.zeros_like(y_pred)
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            y_true = y_true_onehot
        
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Прямое распространение (forward pass).
        
        :param X: Входные данные, форма (batch_size, *input_shape)
        :return: Выход сети (вероятности классов)
        """
        # Flatten вход если нужно
        X_flat = X.reshape(X.shape[0], -1)
        
        self.activations = [X_flat]  # Сохраняем для backprop
        current = X_flat
        
        # Скрытые слои с ReLU
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            current = self._relu(z)
            self.activations.append(current)
        
        # Выходной слой с softmax
        z = current @ self.weights[-1] + self.biases[-1]
        output = self._softmax(z)
        self.activations.append(output)
        
        return output
    
    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Обратное распространение ошибки (backpropagation).
        
        :param X: Входные данные
        :param y_true: Истинные метки
        :param y_pred: Предсказания сети
        """
        batch_size = X.shape[0]
        X_flat = X.reshape(batch_size, -1)
        
        # One-hot encoding
        if y_true.ndim == 1:
            y_onehot = np.zeros_like(y_pred)
            y_onehot[np.arange(batch_size), y_true] = 1
        else:
            y_onehot = y_true
        
        # Градиент по выходному слою (softmax + cross-entropy)
        dz = y_pred - y_onehot  # Упрощённая форма градиента
        
        self.gradients = []
        
        # Обратный проход по слоям
        for i in reversed(range(len(self.weights))):
            # Градиенты по весам и смещениям
            dw = self.activations[i].T @ dz / batch_size
            db = np.mean(dz, axis=0, keepdims=True)
            
            self.gradients.insert(0, {'dw': dw, 'db': db})
            
            if i > 0:  # Не для входного слоя
                # Градиент по активациям предыдущего слоя
                da = dz @ self.weights[i].T
                # Применяем производную ReLU
                z_prev = self.activations[i] @ self.weights[i-1].T + self.biases[i-1] if i > 1 else X_flat @ self.weights[i-1].T + self.biases[i-1]
                dz = da * self._relu_derivative(z_prev)
    
    def update_weights(self) -> None:
        """Обновление весов с помощью градиентного спуска"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.gradients[i]['dw']
            self.biases[i] -= self.learning_rate * self.gradients[i]['db']
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Один шаг обучения (forward + backward + update).
        
        :return: (loss, accuracy) на текущем батче
        """
        # Forward pass
        y_pred = self.forward(X)
        
        # Вычисляем метрики
        loss = self._cross_entropy_loss(y_pred, y)
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y) if y.ndim == 1 else np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        
        # Backward pass
        self.backward(X, y, y_pred)
        
        # Обновление весов
        self.update_weights()
        
        return loss, accuracy
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание классов для входных данных"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает вероятности для каждого класса"""
        return self.forward(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Оценка качества модели на тестовых данных.
        
        :return: Словарь с метриками
        """
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        
        loss = self._cross_entropy_loss(y_pred, y)
        accuracy = np.mean(predictions == y) if y.ndim == 1 else np.mean(
            predictions == np.argmax(y, axis=1)
        )
        
        # Per-class accuracy (для аналитики)
        if y.ndim == 1:
            per_class_acc = {}
            for cls in np.unique(y):
                mask = y == cls
                if np.sum(mask) > 0:
                    per_class_acc[int(cls)] = float(np.mean(predictions[mask] == cls))
        else:
            per_class_acc = {}  # Упрощённо для one-hot
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'per_class_accuracy': per_class_acc,
            'predictions': predictions.tolist()
        }
    
    def save(self, filepath: str) -> None:
        """Сохранение модели в файл"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'biases': self.biases,
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'learning_rate': self.learning_rate,
                'is_trained': True
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'AlienSignalNet':
        """Загрузка модели из файла"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            input_shape=data['input_shape'],
            num_classes=data['num_classes'],
            learning_rate=data['learning_rate']
        )
        instance.weights = data['weights']
        instance.biases = data['biases']
        instance.is_trained = data['is_trained']
        return instance
    
    def count_params(self) -> int:
        """Подсчёт общего количества обучаемых параметров"""
        total = 0
        for W, b in zip(self.weights, self.biases):
            total += W.size + b.size
        return total