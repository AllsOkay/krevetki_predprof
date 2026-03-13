"""
Модуль обучения RNN модели для классификации силы покемонов.

Подход к обучению: Supervised Learning с размеченными данными.
Классы силы определяются на основе суммы базовых характеристик (BST):
- Слабый (weak): BST < 400
- Средний (medium): 400 <= BST < 550
- Сильный (strong): BST >= 550

Это соответствует распределению покемонов в игре:
- ~40% слабые (ранние эволюции, обычные)
- ~45% средние (финальные эволюции)
- ~15% сильные (легендарные, псевдо-легендарные)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from config import RNN_CONFIG, TRAINING_CONFIG
from database import PokemonDatabase
from models.rnn.model import PokemonRNNClassifier


# ==================== Константы для классификации ====================

STRENGTH_THRESHOLDS = {
    'weak': 400,      # BST < 400
    'medium': 550,    # 400 <= BST < 550
    'strong': 9999    # BST >= 550
}

CLASS_NAMES = ['weak', 'medium', 'strong']
CLASS_LABELS_RU = {
    'weak': '🔹 Слабый',
    'medium': '🔸 Средний',
    'strong': '🔴 Сильный'
}


def calculate_bst(pokemon: Dict) -> int:
    """
    Вычисляет сумму базовых характеристик (Base Stat Total).
    
    Args:
        pokemon: Словарь с данными покемона
    
    Returns:
        int: Сумма всех 6 характеристик
    """
    stats = [
        pokemon.get('hp', 0),
        pokemon.get('attack', 0),
        pokemon.get('defense', 0),
        pokemon.get('special_attack', 0) or pokemon.get('special-attack', 0),
        pokemon.get('special_defense', 0) or pokemon.get('special-defense', 0),
        pokemon.get('speed', 0)
    ]
    return sum(stats)


def get_strength_category(bst: int) -> int:
    """
    Определяет категорию силы на основе BST.
    
    Args:
        bst: Сумма базовых характеристик
    
    Returns:
        int: Индекс класса (0=weak, 1=medium, 2=strong)
    """
    if bst < STRENGTH_THRESHOLDS['weak']:
        return 0  # weak
    elif bst < STRENGTH_THRESHOLDS['medium']:
        return 1  # medium
    else:
        return 2  # strong


def normalize_stats(pokemon: Dict) -> np.ndarray:
    """
    Нормализует характеристики покемона для подачи в RNN.
    
    Нормализация важна потому что:
    - HP может быть до 255, а Speed до 180
    - LSTM чувствителен к масштабу входных данных
    
    Args:
        pokemon: Словарь с данными покемона
    
    Returns:
        np.ndarray: Нормализованная последовательность [6, 1]
    """
    # Максимальные значения для нормализации (примерные максимумы в Pokemon)
    max_stats = np.array([255, 190, 230, 194, 230, 180], dtype=np.float32)
    
    stats = np.array([
        pokemon.get('hp', 0),
        pokemon.get('attack', 0),
        pokemon.get('defense', 0),
        pokemon.get('special_attack', 0) or pokemon.get('special-attack', 0),
        pokemon.get('special_defense', 0) or pokemon.get('special-defense', 0),
        pokemon.get('speed', 0)
    ], dtype=np.float32)
    
    # Нормализация к диапазону [0, 1]
    normalized = stats / max_stats
    normalized = np.clip(normalized, 0, 1)
    
    # Преобразуем в последовательность [seq_len=6, features=1]
    sequence = normalized.reshape(-1, 1)
    
    return sequence


# ==================== Dataset для PyTorch ====================

class PokemonStrengthDataset(Dataset):
    """
    Dataset для обучения RNN классификатору силы.
    
    Возвращает:
    - Последовательность нормализованных статов [6, 1]
    - Метку класса (0, 1, или 2)
    """
    
    def __init__(self, pokemon_list: List[Dict]):
        """
        Args:
            pokemon_list: Список словарей с данными покемонов
        """
        self.pokemon_list = pokemon_list
    
    def __len__(self):
        return len(self.pokemon_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pokemon = self.pokemon_list[idx]
        
        # Получаем последовательность статов
        sequence = normalize_stats(pokemon)
        
        # Вычисляем BST и определяем класс
        bst = calculate_bst(pokemon)
        label = get_strength_category(bst)
        
        # Конвертируем в тензоры
        sequence_tensor = torch.FloatTensor(sequence)  # [6, 1]
        label_tensor = torch.LongTensor([label])[0]    # скаляр
        
        return sequence_tensor, label_tensor


# ==================== Основной класс для обучения ====================

class RNNTrainer:
    """
    Класс-обертка для обучения RNN модели.
    """
    
    def __init__(self, db: PokemonDatabase = None):
        """
        Args:
            db: Экземпляр PokemonDatabase
        """
        self.db = db or PokemonDatabase()
        self.device = torch.device(TRAINING_CONFIG['device'])
        self.config = RNN_CONFIG
        
        # Инициализация модели
        self.model = PokemonRNNClassifier(
            input_size=1,
            hidden_size=self.config.get('hidden_dim', 32),
            num_layers=2,
            num_classes=3,
            dropout=0.3
        ).to(self.device)
        
        # Оптимизатор и планировщик
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('epochs', 30)
        )
    
    def train(self, pokemon_list: List[Dict] = None, verbose: bool = True) -> dict:
        """
        Запускает процесс обучения модели.
        
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
            print(f"🎯 Начало обучения RNN на {len(pokemon_list)} покемонах...")
            print(f"📊 Классы: weak(<400), medium(400-550), strong(>=550)")
        
        # Подготовка DataLoader
        dataset = PokemonStrengthDataset(pokemon_list)
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
            
            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)  # [batch, 6, 1]
                labels = labels.to(self.device)        # [batch]
                
                # Прямой проход
                logits = self.model(sequences)
                
                # Вычисление потерь
                loss = self.criterion(logits, labels)
                
                # Обратный проход
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Статистика
                epoch_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            self.scheduler.step()
            
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct / total
            loss_history.append(avg_loss)
            acc_history.append(accuracy)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Эпоха {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2%}")
        
        # ==================== Сохранение модели ====================
        model_path = self.config.get('model_path', Path('models/rnn/rnn_model.pth'))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        
        if verbose:
            print(f"✅ Модель сохранена в {model_path}")
            print(f"📈 Финальная точность: {acc_history[-1]:.2%}")
        
        # Сохраняем эмбеддинги/предсказания в БД
        self._update_predictions(pokemon_list)
        
        return {
            'success': True,
            'epochs': epochs,
            'samples': len(pokemon_list),
            'final_loss': loss_history[-1],
            'final_accuracy': acc_history[-1],
            'loss_history': loss_history,
            'accuracy_history': acc_history
        }
    
    def _update_predictions(self, pokemon_list: List[Dict]):
        """
        Генерирует предсказания для всех покемонов и сохраняет в БД.
        """
        self.model.eval()
        updated_count = 0
        
        for pokemon in pokemon_list:
            try:
                sequence = normalize_stats(pokemon)
                tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    proba = self.model.predict_proba(tensor).cpu().numpy().flatten()
                    pred_class = np.argmax(proba)
                
                # Сохраняем в БД
                result_data = {
                    'class': CLASS_NAMES[pred_class],
                    'class_ru': CLASS_LABELS_RU[CLASS_NAMES[pred_class]],
                    'probabilities': proba.tolist(),
                    'confidence': float(proba[pred_class]),
                    'bst': calculate_bst(pokemon)
                }
                
                self.db.save_model_result(
                    pokemon_id=pokemon['id'],
                    model_type='rnn',
                    result_data=result_data,
                    confidence=result_data['confidence']
                )
                updated_count += 1
                
            except Exception as e:
                print(f"⚠️ Ошибка предсказания для {pokemon.get('name', 'unknown')}: {e}")
                continue
        
        print(f"📦 Обновлены предсказания для {updated_count} покемонов")
    
    def load_model(self, model_path: str = None):
        """Загружает предобученные веса модели."""
        path = model_path or self.config.get('model_path', Path('models/rnn/rnn_model.pth'))
        
        if not path.exists():
            print(f"⚠️ Файл модели не найден: {path}")
            return False
        
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"✅ Модель загружена из {path}")
        return True