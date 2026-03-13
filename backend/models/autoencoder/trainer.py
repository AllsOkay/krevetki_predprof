"""
Модуль обучения автоэнкодера для детекции "обычности" покемонов.

Подход: Unsupervised Learning (без размеченных данных).
Автоэнкодер учится восстанавливать характеристики "типичных" покемонов.

После обучения:
- Низкая ошибка восстановления = покемон "обычный" (похож на большинство)
- Высокая ошибка восстановления = покемон "уникальный" (аномалия)

"Обычность" в процентах = 100% - нормализованная ошибка восстановления
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from config import AE_CONFIG, TRAINING_CONFIG
from database import PokemonDatabase
from models.autoencoder.model import PokemonAutoencoder


# ==================== Препроцессинг данных ====================

def prepare_pokemon_features(pokemon: Dict) -> np.ndarray:
    """
    Преобразует данные покемона в числовой вектор для автоэнкодера.
    
    Входные признаки (26):
    - 8 числовых характеристик (нормализованные [0, 1])
    - 18 типов (one-hot encoding)
    
    Args:
        pokemon: Словарь с данными покемона
    
    Returns:
        np.ndarray: Вектор признаков размера [26]
    """
    # Максимальные значения для нормализации статов
    max_stats = np.array([255, 190, 230, 194, 230, 180, 60, 100], dtype=np.float32)
    
    # Числовые признаки
    stats = np.array([
        pokemon.get('hp', 0),
        pokemon.get('attack', 0),
        pokemon.get('defense', 0),
        pokemon.get('special_attack', 0) or pokemon.get('special-attack', 0),
        pokemon.get('special_defense', 0) or pokemon.get('special-defense', 0),
        pokemon.get('speed', 0),
        pokemon.get('height', 0) or 0,
        pokemon.get('weight', 0) or 0
    ], dtype=np.float32)
    
    # Нормализация к [0, 1]
    normalized_stats = np.clip(stats / max_stats, 0, 1)
    
    # One-hot encoding типов
    types_list = [
        "normal", "fighting", "flying", "poison", "ground", "bug", "ghost", 
        "steel", "fire", "water", "grass", "electric", "psychic", "ice", 
        "dragon", "fairy", "dark", "rock"
    ]
    types_onehot = np.zeros(len(types_list), dtype=np.float32)
    
    types_str = pokemon.get('types', '')
    if types_str:
        for t in types_str.split(','):
            t = t.strip().lower()
            if t in types_list:
                types_onehot[types_list.index(t)] = 1.0
    
    # Объединяем признаки
    features = np.concatenate([normalized_stats, types_onehot])
    
    return features


# ==================== Dataset для PyTorch ====================

class PokemonAEDataset(Dataset):
    """
    Dataset для обучения автоэнкодера.
    
    Возвращает нормализованные признаки покемона.
    """
    
    def __init__(self, pokemon_list: List[Dict]):
        self.pokemon_list = pokemon_list
    
    def __len__(self):
        return len(self.pokemon_list)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        pokemon = self.pokemon_list[idx]
        features = prepare_pokemon_features(pokemon)
        return torch.FloatTensor(features)


# ==================== Основной класс для обучения ====================

class AETrainer:
    """
    Класс-обертка для обучения автоэнкодера.
    """
    
    def __init__(self, db: PokemonDatabase = None):
        self.db = db or PokemonDatabase()
        self.device = torch.device(TRAINING_CONFIG['device'])
        self.config = AE_CONFIG
        
        # Инициализация модели
        self.model = PokemonAutoencoder(
            input_dim=self.config.get('input_dim', 26),
            latent_dim=self.config.get('latent_dim', 8),
            hidden_dims=self.config.get('hidden_dims', [16, 12]),
            dropout=self.config.get('dropout', 0.2)
        ).to(self.device)
        
        # Оптимизатор и функция потерь
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('epochs', 30)
        )
    
    def train(self, pokemon_list: List[Dict] = None, verbose: bool = True) -> dict:
        """
        Запускает процесс обучения автоэнкодера.
        
        Args:
            pokemon_list: Список покемонов для обучения
            verbose: Выводить ли прогресс
        
        Returns:
            dict: Статистика обучения
        """
        if pokemon_list is None:
            pokemon_list = self.db.get_all_pokemon()
        
        # Фильтруем покемонов с достаточными данными
        pokemon_list = [p for p in pokemon_list if p.get('hp') is not None]
        
        min_samples = self.config.get('min_samples', 50)
        if len(pokemon_list) < min_samples:
            print(f"⚠️ Недостаточно данных: {len(pokemon_list)} < {min_samples}")
            return {
                'success': False,
                'reason': 'insufficient_data',
                'current_count': len(pokemon_list),
                'required_count': min_samples
            }
        
        if verbose:
            print(f"🎯 Начало обучения автоэнкодера на {len(pokemon_list)} покемонах...")
            print(f"📊 Вход: {self.config.get('input_dim')} признаков, Латент: {self.config.get('latent_dim')}")
        
        # Подготовка DataLoader
        dataset = PokemonAEDataset(pokemon_list)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        self.model.train()
        loss_history = []
        
        # ==================== Цикл обучения ====================
        epochs = self.config.get('epochs', 30)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                
                # Прямой проход
                _, reconstructed = self.model(batch)
                
                # Вычисление потерь (MSE)
                loss = self.criterion(reconstructed, batch)
                
                # Обратный проход
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            self.scheduler.step()
            
            avg_loss = epoch_loss / batch_count
            loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Эпоха {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # ==================== Сохранение модели ====================
        model_path = self.config.get('model_path', Path('models/autoencoder/ae_model.pth'))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        
        if verbose:
            print(f"✅ Модель сохранена в {model_path}")
            print(f"📉 Финальная потеря: {loss_history[-1]:.4f}")
        
        # Сохраняем "обычность" для всех покемонов в БД
        self._update_ordinariness(pokemon_list)
        
        return {
            'success': True,
            'epochs': epochs,
            'samples': len(pokemon_list),
            'final_loss': loss_history[-1],
            'loss_history': loss_history
        }
    
    def _update_ordinariness(self, pokemon_list: List[Dict]):
        """
        Вычисляет "обычность" для всех покемонов и сохраняет в БД.
        
        Обычность (%) = 100 - нормализованная ошибка восстановления
        """
        self.model.eval()
        updated_count = 0
        
        # Собираем все ошибки для нормализации
        all_errors = []
        pokemon_data = []
        
        for pokemon in pokemon_list:
            try:
                features = prepare_pokemon_features(pokemon)
                tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    error = self.model.get_reconstruction_error(tensor).item()
                
                all_errors.append(error)
                pokemon_data.append((pokemon['id'], pokemon['name'], error))
                
            except Exception as e:
                print(f"⚠️ Ошибка для {pokemon.get('name', 'unknown')}: {e}")
                continue
        
        if not all_errors:
            return
        
        # Нормализуем ошибки к диапазону [0, 100] для "обычности"
        min_err = min(all_errors)
        max_err = max(all_errors)
        err_range = max_err - min_err if max_err > min_err else 1.0
        
        for pid, name, error in pokemon_data:
            # Низкая ошибка = высокая обычность
            ordinariness = 100 - ((error - min_err) / err_range * 100)
            ordinariness = round(max(0, min(100, ordinariness)), 1)
            
            result_data = {
                'ordinariness': ordinariness,
                'reconstruction_error': round(error, 4),
                'error_min': round(min_err, 4),
                'error_max': round(max_err, 4)
            }
            
            self.db.save_model_result(
                pokemon_id=pid,
                model_type='autoencoder',
                result_data=result_data,
                confidence=ordinariness / 100
            )
            updated_count += 1
        
        print(f"📦 Обновлена 'обычность' для {updated_count} покемонов")
    
    def load_model(self, model_path: str = None):
        """Загружает предобученные веса модели."""
        path = model_path or self.config.get('model_path', Path('models/autoencoder/ae_model.pth'))
        
        if not path.exists():
            print(f"⚠️ Файл модели не найден: {path}")
            return False
        
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"✅ Автоэнкодер загружен из {path}")
        return True