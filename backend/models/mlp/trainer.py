"""
Модуль обучения MLP.

Особенности:
- Поддержка как регрессии, так и классификации
- Автоматический выбор функции потерь
- Валидация на отложенной выборке
- Сохранение лучшей модели по метрике
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .model import PokemonMLP


class MLPTrainer:
    """
    Тренер для MLP с поддержкой ранней остановки и сохранения чекпоинтов.
    
    Args:
        model: Экземпляр PokemonMLP
        task: 'regression' или 'classification'
        lr: Скорость обучения
        device: 'cpu' или 'cuda'
    """
    
    def __init__(
        self, 
        model: PokemonMLP, 
        task: str = 'regression',
        lr: float = 0.001,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.task = task
        self.device = device
        
        # Автоматический выбор функции потерь
        if task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss() if model.network[-1].__class__.__name__ != 'Softmax' else nn.NLLLoss()
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.best_loss = float('inf')
        
    def train_epoch(self, loader: DataLoader) -> float:
        """
        Один эпизод обучения.
        
        Returns:
            float: средний loss за эпоху
        """
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            # Обнуляем градиенты
            self.optimizer.zero_grad()
            
            # Прямой проход
            predictions = self.model(X_batch)
            
            # Для классификации с softmax нужен log для NLLLoss
            if self.task == 'classification' and self.model.network[-1].__class__.__name__ == 'Softmax':
                predictions = torch.log(predictions + 1e-8)  # Защита от log(0)
            
            # Вычисляем loss
            loss = self.criterion(predictions, y_batch)
            
            # Обратный проход
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def validate(self, loader: DataLoader) -> float:
        """
        Валидация модели.
        
        Returns:
            float: средний loss на валидации
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions = self.model(X_batch)
                
                if self.task == 'classification' and self.model.network[-1].__class__.__name__ == 'Softmax':
                    predictions = torch.log(predictions + 1e-8)
                
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        save_path: str = 'mlp_model.pth'
    ) -> dict:
        """
        Полный цикл обучения.
        
        Args:
            X_train, y_train: Обучающая выборка
            X_val, y_val: Валидационная выборка (опционально)
            epochs: Количество эпох
            batch_size: Размер батча
            patience: Ранняя остановка, если нет улучшений
            save_path: Путь для сохранения лучшей модели
            
        Returns:
            dict: история обучения { 'train_loss': [...], 'val_loss': [...] }
        """
        # Создаём DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train) if self.task == 'regression' else torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        history = {'train_loss': [], 'val_loss': []}
        no_improve = 0
        
        print(f"🚀 Начало обучения MLP ({self.task})...")
        
        for epoch in range(epochs):
            # Обучение
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Валидация
            if X_val is not None:
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val),
                    torch.FloatTensor(y_val) if self.task == 'regression' else torch.LongTensor(y_val)
                )
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                
                # Ранняя остановка
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    torch.save(self.model.state_dict(), save_path)
                    no_improve = 0
                    print(f"  Epoch {epoch+1}: val_loss={val_loss:.4f} ✅ (сохранено)")
                else:
                    no_improve += 1
                    print(f"  Epoch {epoch+1}: val_loss={val_loss:.4f}")
                    
                if no_improve >= patience:
                    print(f"⏹ Ранняя остановка на эпохе {epoch+1}")
                    break
            else:
                # Без валидации сохраняем последнюю модель
                torch.save(self.model.state_dict(), save_path)
                print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}")
        
        print(f"✅ Обучение завершено. Лучший val_loss: {self.best_loss:.4f}")
        return history