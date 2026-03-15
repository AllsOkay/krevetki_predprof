# backend/model/neural_net.py
# Реализация нейронной сети для классификации инопланетных сигналов

import numpy as np
from typing import Tuple, List, Optional
import pickle
import os

__all__ = ['AlienSignalNet']


class AlienSignalNet:
    """
    Кастомная нейронная сеть для классификации радиосигналов.
    """

    def __init__(self, input_shape: Tuple[int], num_classes: int, 
                 hidden_layers: List[int] = [128, 64], 
                 learning_rate: float = 0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.is_trained = False
        
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        
        in_size = np.prod(input_shape)
        for i, out_size in enumerate(hidden_layers + [num_classes]):
            scale = np.sqrt(2.0 / (in_size + out_size))
            W = np.random.randn(in_size, out_size) * scale
            b = np.zeros((1, out_size))
            
            self.weights.append(W)
            self.biases.append(b)
            in_size = out_size
        
        self.gradients = None
        self.activations = None
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _cross_entropy_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if y_true.ndim == 1:
            y_true_onehot = np.zeros_like(y_pred)
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            y_true = y_true_onehot
        
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

    def _resize_input(self, X_flat: np.ndarray, target_size: int) -> np.ndarray:
        """
        ✅ Подгоняет размер входа к ожидаемому модели через интерполяцию
        """
        current_size = X_flat.shape[1]
        
        if current_size == target_size:
            return X_flat
        
        batch_size = X_flat.shape[0]
        resized = np.zeros((batch_size, target_size))
        
        for i in range(batch_size):
            resized[i] = np.interp(
                np.linspace(0, 1, target_size),
                np.linspace(0, 1, current_size),
                X_flat[i]
            )
        
        return resized

    def forward(self, X: np.ndarray) -> np.ndarray:
        X_flat = X.reshape(X.shape[0], -1)
        
        expected_input_size = np.prod(self.input_shape)
        actual_input_size = X_flat.shape[1]
        
        # ✅ АВТОМАТИЧЕСКАЯ ПОДГОНКА РАЗМЕРА
        if actual_input_size != expected_input_size:
            X_flat = self._resize_input(X_flat, expected_input_size)
        
        self.activations = [X_flat]
        current = X_flat
        
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            current = self._relu(z)
            self.activations.append(current)
        
        z = current @ self.weights[-1] + self.biases[-1]
        output = self._softmax(z)
        self.activations.append(output)
        
        return output

    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        batch_size = X.shape[0]
        X_flat = X.reshape(batch_size, -1)
        
        expected_input_size = np.prod(self.input_shape)
        if X_flat.shape[1] != expected_input_size:
            X_flat = self._resize_input(X_flat, expected_input_size)
        
        if y_true.ndim == 1:
            y_onehot = np.zeros_like(y_pred)
            y_onehot[np.arange(batch_size), y_true] = 1
        else:
            y_onehot = y_true
        
        dz = y_pred - y_onehot
        self.gradients = []
        
        for i in reversed(range(len(self.weights))):
            dw = self.activations[i].T @ dz / batch_size
            db = np.mean(dz, axis=0, keepdims=True)
            
            self.gradients.insert(0, {'dw': dw, 'db': db})
            
            if i > 0:
                da = dz @ self.weights[i].T
                z_prev = self.activations[i] @ self.weights[i-1].T + self.biases[i-1] if i > 1 else X_flat @ self.weights[i-1].T + self.biases[i-1]
                dz = da * self._relu_derivative(z_prev)

    def update_weights(self) -> None:
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.gradients[i]['dw']
            self.biases[i] -= self.learning_rate * self.gradients[i]['db']

    def train_step(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        y_pred = self.forward(X)
        loss = self._cross_entropy_loss(y_pred, y)
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y) if y.ndim == 1 else np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        
        self.backward(X, y, y_pred)
        self.update_weights()
        
        return loss, accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        
        loss = self._cross_entropy_loss(y_pred, y)
        accuracy = np.mean(predictions == y) if y.ndim == 1 else np.mean(
            predictions == np.argmax(y, axis=1)
        )
        
        if y.ndim == 1:
            per_class_acc = {}
            for cls in np.unique(y):
                mask = y == cls
                if np.sum(mask) > 0:
                    per_class_acc[int(cls)] = float(np.mean(predictions[mask] == cls))
        else:
            per_class_acc = {}
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'per_class_accuracy': per_class_acc,
            'predictions': predictions.tolist()
        }

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights': [w.tolist() for w in self.weights],
                'biases': [b.tolist() for b in self.biases],
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'learning_rate': self.learning_rate,
                'is_trained': True
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'AlienSignalNet':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            input_shape=tuple(data['input_shape']),
            num_classes=data['num_classes'],
            learning_rate=data['learning_rate']
        )
        
        instance.weights = [np.array(w) for w in data['weights']]
        instance.biases = [np.array(b) for b in data['biases']]
        instance.is_trained = data.get('is_trained', True)
        
        return instance

    def count_params(self) -> int:
        total = 0
        for W, b in zip(self.weights, self.biases):
            W_arr = np.array(W) if not isinstance(W, np.ndarray) else W
            b_arr = np.array(b) if not isinstance(b, np.ndarray) else b
            total += W_arr.size + b_arr.size
        return total