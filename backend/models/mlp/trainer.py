"""
Модуль обучения MLP модели для предсказания вероятности победы.

Подход к обучению: Supervised Learning с симулированными боями.

Генерация обучающих данных:
1. Берём двух случайных покемонов из базы
2. Сравниваем их BST (Base Stat Total)
3. Добавляем элемент случайности (10-20% шанс на upset)
4. Записываем результат (1 = победа, 0 = поражение)

Это создаёт реалистичную модель боёв где статы важны, но не гарантируют победу.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from config import MLP_CONFIG, TRAINING_CONFIG
from database import PokemonDatabase
from models.mlp.model import PokemonMLPClassifier


# ==================== Константы ====================

# Максимальные значения для нормализации статов
MAX_STATS = {
    'hp': 255,
    'attack': 190,
    'defense': 230,
    'speed': 180
}

# Шанс на неожиданную победу (upset)
UPSET_CHANCE = 0.15


def calculate_bst(pokemon: Dict) -> int:
    """Вычисляет сумму базовых характеристик."""
    stats = [
        pokemon.get('hp', 0),
        pokemon.get('attack', 0),
        pokemon.get('defense', 0),
        pokemon.get('special_attack', 0) or pokemon.get('special-attack', 0),
        pokemon.get('special_defense', 0) or pokemon.get('special-defense', 0),
        pokemon.get('speed', 0)
    ]
    return sum(stats)


def normalize_stats(hp: float, atk: float, def_: float, spd: float) -> np.ndarray:
    """
    Нормализует характеристики к диапазону [0, 1].
    
    Args:
        hp, atk, def_, spd: Значения характеристик
    
    Returns:
        np.ndarray: Нормализованный вектор [4]
    """
    return np.array([
        min(hp / MAX_STATS['hp'], 1.0),
        min(atk / MAX_STATS['attack'], 1.0),
        min(def_ / MAX_STATS['defense'], 1.0),
        min(spd / MAX_STATS['speed'], 1.0)
    ], dtype=np.float32)


def simulate_battle(pokemon1: Dict, pokemon2: Dict) -> int:
    """
    Симулирует бой между двумя покемонами.
    
    Логика:
    - Сравниваем BST
    - Добавляем случайность (upset chance)
    
    Returns:
        int: 1 если pokemon1 победил, 0 иначе
    """
    bst1 = calculate_bst(pokemon1)
    bst2 = calculate_bst(pokemon2)
    
    # Базовая вероятность победы на основе разницы BST
    bst_diff = bst1 - bst2
    base_win_prob = 0.5 + (bst_diff / 1000)  # Нормализация
    base_win_prob = max(0.1, min(0.9, base_win_prob))
    
    # Добавляем случайность
    if np.random.random() < UPSET_CHANCE:
        return 1 - int(base_win_prob > 0.5)
    
    return 1 if np.random.random() < base_win_prob else 0


# ==================== Dataset для PyTorch ====================

class PokemonBattleDataset(Dataset):
    """
    Dataset для обучения MLP на симулированных боях.
    
    Возвращает:
    - Нормализованные статы покемона [4]
    - Результат боя (1 = победа, 0 = поражение)
    """
    
    def __init__(self, pokemon_list: List[Dict], num_samples: int = 10000):
        """
        Args:
            pokemon_list: Список покемонов для симуляции боёв
            num_samples: Количество боёв для генерации
        """
        self.pokemon_list = pokemon_list
        self.num_samples = num_samples
        
        # Генерируем обучающие данные
        self.data = self._generate_battles()
    
    def _generate_battles(self) -> List[Tuple[np.ndarray, int]]:
        """Генерирует симулированные бои."""
        battles = []
        
        for _ in range(self.num_samples):
            # Выбираем двух случайных покемонов
            p1 = np.random.choice(self.pokemon_list)
            p2 = np.random.choice(self.pokemon_list)
            
            # Нормализуем статы p1
            stats = normalize_stats(
                p1.get('hp', 50),
                p1.get('attack', 50),
                p1.get('defense', 50),
                p1.get('speed', 50)
            )
            
            # Симулируем бой
            result = simulate_battle(p1, p2)
            
            battles.append((stats, result))
        
        return battles
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stats, result = self.data[idx]
        
        stats_tensor = torch.FloatTensor(stats)
        result_tensor = torch.FloatTensor([result])[0]
        
        return stats_tensor, result_tensor


# ==================== Основной класс для обучения ====================

class MLPTrainer:
    """
    Класс-обертка для обучения MLP модели.
    """
    
    def __init__(self, db: PokemonDatabase = None):
        self.db = db or PokemonDatabase()
        self.device = torch.device(TRAINING_CONFIG['device'])
        self.config = MLP_CONFIG
        
        # Инициализация модели
        self.model = PokemonMLPClassifier(
            input_dim=self.config.get('input_dim', 4),
            hidden_dims=self.config.get('hidden_layers', [64, 32, 16]),
            dropout=self.config.get('dropout', 0.3)
        ).to(self.device)
        
        # Оптимизатор и функция потерь
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        self.criterion = nn.BCELoss()  # Binary Cross Entropy для вероятности
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('epochs', 30)
        )
    
    def train(self, pokemon_list: List[Dict] = None, verbose: bool = True) -> dict:
        """
        Запускает процесс обучения модели.
        
        Args:
            pokemon_list: Список покемонов для генерации боёв
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
            print(f"🎯 Начало обучения MLP на {len(pokemon_list)} покемонах...")
            print(f"📊 Генерация {self.config.get('num_battles', 10000)} симулированных боёв")
        
        # Подготовка DataLoader
        dataset = PokemonBattleDataset(
            pokemon_list, 
            num_samples=self.config.get('num_battles', 10000)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        self.model.train()
        loss_history = []
        acc_history = []
        
        # ==================== Цикл обучения ====================
        epochs = self.config.get('epochs', 30)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for stats, labels in dataloader:
                stats = stats.to(self.device)
                labels = labels.to(self.device)
                
                # Прямой проход
                predictions = self.model(stats).squeeze()
                
                # Вычисление потерь
                loss = self.criterion(predictions, labels)
                
                # Обратный проход
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Статистика
                epoch_loss += loss.item()
                predicted_classes = (predictions >= 0.5).float()
                correct += (predicted_classes == labels).sum().item()
                total += labels.size(0)
            
            self.scheduler.step()
            
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct / total
            loss_history.append(avg_loss)
            acc_history.append(accuracy)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Эпоха {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2%}")

            # В конце метода train():
            metrics_for_db = []
            for epoch, (loss, acc) in enumerate(zip(loss_history, acc_history), start=1):
                metrics_for_db.append({
                    'epoch': epoch,
                    'loss': float(loss),
                    'accuracy': float(acc),
                    'metric_name': 'win_prediction',
                    'metric_value': float(acc)
                })

            self.db.save_training_metrics('mlp', metrics_for_db)
        
        # ==================== Сохранение модели ====================
        model_path = self.config.get('model_path', Path('models/mlp/mlp_model.pth'))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        
        if verbose:
            print(f"✅ Модель сохранена в {model_path}")
            print(f"📈 Финальная точность: {acc_history[-1]:.2%}")
        
        return {
            'success': True,
            'epochs': epochs,
            'samples': len(dataset),
            'pokemon_count': len(pokemon_list),
            'final_loss': loss_history[-1],
            'final_accuracy': acc_history[-1],
            'loss_history': loss_history,
            'accuracy_history': acc_history
        }
    
    def load_model(self, model_path: str = None):
        """Загружает предобученные веса модели."""
        path = model_path or self.config.get('model_path', Path('models/mlp/mlp_model.pth'))
        
        if not path.exists():
            print(f"⚠️ Файл модели не найден: {path}")
            return False
        
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"✅ MLP модель загружена из {path}")
        return True