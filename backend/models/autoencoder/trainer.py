"""
Модуль обучения Autoencoder.

Особенности:
- Восстановление входных данных как целевая переменная
- Метрика: MSE между входом и восстановлением
- Возможность получения латентных представлений
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Optional

from .model import PokemonAutoencoder


class AutoencoderTrainer:
    """
    Тренер для автоэнкодера PokemonAutoencoder.
    
    Args:
        model: Экземпляр PokemonAutoencoder
        lr: Скорость обучения
        device: 'cpu' или 'cuda'
    """
    
    def __init__(self, model: PokemonAutoencoder, lr: float = 0.001, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()  # Восстановление: MSE между входом и выходом
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.best_loss = float('inf')
        
    def train_epoch(self, loader: DataLoader) -> float:
        """Один эпизод обучения"""
        self.model.train()
        total_loss = 0
        
        for X_batch, _ in loader:  # Для автоэнкодера вход = выход
            X_batch = X_batch.to(self.device)
            
            self.optimizer.zero_grad()
            _, reconstructed = self.model(X_batch)
            loss = self.criterion(reconstructed, X_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def validate(self, loader: DataLoader) -> float:
        """Валидация"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                _, reconstructed = self.model(X_batch)
                loss = self.criterion(reconstructed, X_batch)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def fit(
        self,
        X: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        save_path: str = 'models/autoencoder/ae_model.pth'
    ) -> dict:
        """
        Обучение автоэнкодера.
        
        Args:
            X: Признаки для обучения (вход = целевое значение)
            X_val: Валидационные данные
            epochs: Количество эпох
            batch_size: Размер батча
            patience: Ранняя остановка
            save_path: Путь сохранения
            
        Returns:
            dict: история обучения
        """
        train_dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(X))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        history = {'train_loss': [], 'val_loss': []}
        no_improve = 0
        
        print(f"🚀 Начало обучения Autoencoder (latent_dim={self.model.latent_dim})...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            if X_val is not None:
                val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val))
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
        
        print(f"✅ Обучение Autoencoder завершено")
        return history
    
    def get_latent_representations(self, X: np.ndarray) -> np.ndarray:
        """
        Получает латентные представления для данных.
        
        Args:
            X: Входные данные
            
        Returns:
            np.ndarray: латентные векторы
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            latent = self.model.encode(X_tensor).cpu().numpy()
        
        return latent