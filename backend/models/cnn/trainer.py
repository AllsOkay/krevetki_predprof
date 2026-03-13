"""
Модуль обучения CNN.

Особенности:
- Работа с 4D тензорами (batch, channel, height, width)
- Автоматический выбор функции потерь для классификации
- Валидация и сохранение лучшей модели
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Optional

from .model import PokemonCNN


class CNNTrainer:
    """
    Тренер для сверточной сети PokemonCNN.
    
    Args:
        model: Экземпляр PokemonCNN
        lr: Скорость обучения
        device: 'cpu' или 'cuda'
    """
    
    def __init__(self, model: PokemonCNN, lr: float = 0.001, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()  # Для многоклассовой классификации
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.best_loss = float('inf')
        
    def train_epoch(self, loader: DataLoader) -> float:
        """Один эпизод обучения"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch.long())
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def validate(self, loader: DataLoader) -> float:
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch.long())
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 16,
        patience: int = 10,
        save_path: str = 'models/cnn/cnn_model.pth'
    ) -> dict:
        """
        Полный цикл обучения.
        
        Args:
            X_train: Признаки (n_samples, 1, H, W)
            y_train: Метки классов (n_samples,)
            X_val, y_val: Валидационная выборка
            epochs: Количество эпох
            batch_size: Размер батча
            patience: Ранняя остановка
            save_path: Путь сохранения модели
            
        Returns:
            dict: история обучения
        """
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        history = {'train_loss': [], 'val_loss': []}
        no_improve = 0
        
        print(f"🚀 Начало обучения CNN...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            if X_val is not None:
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val),
                    torch.LongTensor(y_val)
                )
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
                    no_improve = 0
                    print(f"  Epoch {epoch+1}: val_loss={val_loss:.4f} ✅")
                else:
                    no_improve += 1
                    print(f"  Epoch {epoch+1}: val_loss={val_loss:.4f}")
                    
                if no_improve >= patience:
                    print(f"⏹ Ранняя остановка на эпохе {epoch+1}")
                    break
            else:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}")
        
        print(f"✅ Обучение CNN завершено")
        return history