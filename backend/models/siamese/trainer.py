"""
Модуль обучения Siamese Network.

Особенности:
- Контрастивная функция потерь для обучения схожести
- Работа с парами данных
- Косинусное сходство для инференса
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Optional

from .model import PokemonSiamese


class SiameseTrainer:
    """
    Тренер для сиамской сети PokemonSiamese.
    
    Args:
        model: Экземпляр PokemonSiamese
        lr: Скорость обучения
        device: 'cpu' или 'cuda'
    """
    
    def __init__(self, model: PokemonSiamese, lr: float = 0.001, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        # Контрастивная лосс-функция определена в модели
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.best_loss = float('inf')
        
    def train_epoch(self, loader: DataLoader) -> float:
        """Один эпизод обучения"""
        self.model.train()
        total_loss = 0
        
        for (pair_batch, label_batch) in loader:
            x1 = pair_batch[:, 0, :].to(self.device)  # Первый покемон пары
            x2 = pair_batch[:, 1, :].to(self.device)  # Второй покемон пары
            y = label_batch.to(self.device)
            
            self.optimizer.zero_grad()
            emb1, emb2, _ = self.model(x1, x2)
            loss = self.model.contrastive_loss(emb1, emb2, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def validate(self, loader: DataLoader) -> float:
        """Валидация"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for (pair_batch, label_batch) in loader:
                x1 = pair_batch[:, 0, :].to(self.device)
                x2 = pair_batch[:, 1, :].to(self.device)
                y = label_batch.to(self.device)
                
                emb1, emb2, _ = self.model(x1, x2)
                loss = self.model.contrastive_loss(emb1, emb2, y)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def fit(
        self,
        pairs: torch.Tensor,
        labels: torch.Tensor,
        pairs_val: Optional[torch.Tensor] = None,
        labels_val: Optional[torch.Tensor] = None,
        epochs: int = 50,
        batch_size: int = 16,
        patience: int = 10,
        save_path: str = 'models/siamese/siamese_model.pth'
    ) -> dict:
        """
        Обучение сиамской сети.
        
        Args:
            pairs: Пары признаков (n_pairs, 2, features)
            labels: Метки схожести (1 = похожи, 0 = непохожи)
            pairs_val, labels_val: Валидационные данные
            epochs: Количество эпох
            batch_size: Размер батча
            patience: Ранняя остановка
            save_path: Путь сохранения
            
        Returns:
            dict: история обучения
        """
        train_dataset = TensorDataset(pairs, labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        history = {'train_loss': [], 'val_loss': []}
        no_improve = 0
        
        print(f"🚀 Начало обучения Siamese Network...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            if pairs_val is not None:
                val_dataset = TensorDataset(pairs_val, labels_val)
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
        
        print(f"✅ Обучение Siamese завершено")
        return history