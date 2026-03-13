"""
Модуль обучения RNN/LSTM.

Особенности:
- Работа с последовательностями (batch, seq_len, features)
- Поддержка регрессии и классификации
- Градиентный клиппинг для стабильности LSTM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Optional

from .model import PokemonRNN


class RNNTrainer:
    """
    Тренер для рекуррентной сети PokemonRNN.
    
    Args:
        model: Экземпляр PokemonRNN
        task: 'regression' или 'classification'
        lr: Скорость обучения
        device: 'cpu' или 'cuda'
        clip_grad: Максимальная норма градиента для клиппинга
    """
    
    def __init__(
        self,
        model: PokemonRNN,
        task: str = 'regression',
        lr: float = 0.001,
        device: str = 'cpu',
        clip_grad: float = 1.0
    ):
        self.model = model.to(device)
        self.task = task
        self.device = device
        self.clip_grad = clip_grad
        
        if task == 'regression':
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.best_loss = float('inf')
        
    def train_epoch(self, loader: DataLoader) -> float:
        """Один эпизод обучения с градиентным клиппингом"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            
            if self.task == 'regression':
                loss = self.criterion(predictions, y_batch)
            else:
                loss = self.criterion(predictions, y_batch.long())
                
            loss.backward()
            
            # Градиентный клиппинг для стабильности LSTM
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def validate(self, loader: DataLoader) -> float:
        """Валидация"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                predictions = self.model(X_batch)
                
                if self.task == 'regression':
                    loss = self.criterion(predictions, y_batch)
                else:
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
        batch_size: int = 8,  # Меньший батч для последовательностей
        patience: int = 10,
        save_path: str = 'models/rnn/rnn_model.pth'
    ) -> dict:
        """Полный цикл обучения"""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train) if self.task == 'regression' else torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        history = {'train_loss': [], 'val_loss': []}
        no_improve = 0
        
        print(f"🚀 Начало обучения RNN ({self.task})...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            if X_val is not None:
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val),
                    torch.FloatTensor(y_val) if self.task == 'regression' else torch.LongTensor(y_val)
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
        
        print(f"✅ Обучение RNN завершено")
        return history