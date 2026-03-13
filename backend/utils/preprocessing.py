"""
Утилиты предобработки данных для разных типов нейросетей.
Каждая функция адаптирует данные под конкретную архитектуру.
"""

import numpy as np
import torch
from typing import Tuple, List


def prepare_for_mlp(X: np.ndarray, y: np.ndarray = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Подготавливает данные для MLP (полносвязной сети).
    
    Args:
        X: Признаки (n_samples, n_features)
        y: Целевая переменная (опционально)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: тензоры для обучения
    """
    X_tensor = torch.FloatTensor(X)
    if y is not None:
        y_tensor = torch.FloatTensor(y) if len(y.shape) == 1 else torch.LongTensor(y)
        return X_tensor, y_tensor
    return X_tensor, None


def prepare_for_cnn(X: np.ndarray, img_size: Tuple[int, int] = (8, 4)) -> torch.Tensor:
    """
    Преобразует фич-вектор в "изображение" для CNN.
    
    Идея: числовые признаки (8 шт) + типы (18 шт) = 26 значений.
    Reshape в 2D массив для применения свёрток.
    
    Args:
        X: Признаки (n_samples, 26)
        img_size: Желаемый размер "изображения" (по умолчанию 8x4=32, лишнее обрезается)
        
    Returns:
        torch.Tensorshape (n_samples, 1, H, W) для PyTorch CNN
    """
    # Обрезаем или дополняем до нужного размера
    target_size = img_size[0] * img_size[1]
    if X.shape[1] < target_size:
        # Дополняем нулями
        padding = np.zeros((X.shape[0], target_size - X.shape[1]))
        X_padded = np.hstack([X, padding])
    else:
        X_padded = X[:, :target_size]
    
    # Reshape в 2D + добавляем канал (как в изображениях)
    X_2d = X_padded.reshape(-1, 1, img_size[0], img_size[1])
    
    return torch.FloatTensor(X_2d)


def prepare_for_rnn(X: np.ndarray, sequence_length: int = 5) -> torch.Tensor:
    """
    Создаёт последовательности для RNN/LSTM.
    
    Идея: используем ID покемонов как "временную ось".
    Для каждого покемона берём его статы + статы предыдущих N покемонов.
    
    Args:
        X: Признаки всех покемонов (n_total, n_features)
        sequence_length: Длина последовательности для LSTM
        
    Returns:
        torch.Tensorshape (n_samples, sequence_length, n_features)
    """
    sequences = []
    
    for i in range(sequence_length, len(X)):
        # Берём окно из предыдущих sequence_length покемонов
        seq = X[i-sequence_length:i]
        sequences.append(seq)
    
    return torch.FloatTensor(np.array(sequences))


def prepare_for_autoencoder(X: np.ndarray) -> torch.Tensor:
    """
    Подготовка данных для автоэнкодера.
    
    Автоэнкодер учится восстанавливать входные данные,
    поэтому X используется и как вход, и как целевое значение.
    
    Args:
        X: Признаки для сжатия
        
    Returns:
        torch.Tensor для обучения
    """
    return torch.FloatTensor(X)


def prepare_for_siamese(X: np.ndarray, pairs: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Создаёт пары данных для сиамской сети.
    
    Args:
        X: Все признаки (n_samples, n_features)
        pairs: Список кортежей (idx1, idx2) для сравнения
        
    Returns:
        Tuple:
            - torch.Tensor: пары признаков (n_pairs, 2, n_features)
            - torch.Tensor: метки схожести (1 если один тип, 0 иначе)
    """
    pair_data = []
    labels = []
    
    for idx1, idx2 in pairs:
        pair_data.append([X[idx1], X[idx2]])
        # Упрощённая метка: 1 если типы совпадают
        labels.append(1 if idx1 == idx2 else 0)  # В реальности нужна логика сравнения типов
    
    return torch.FloatTensor(np.array(pair_data)), torch.FloatTensor(labels)