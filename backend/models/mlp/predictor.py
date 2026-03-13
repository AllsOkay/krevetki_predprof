"""
Модуль инференса для предсказания вероятности победы.

Использует предобученную MLP для:
1. Приёма 4 характеристик от пользователя
2. Нормализации данных
3. Предсказания вероятности победы (0-100%)
"""
import torch
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from config import MLP_CONFIG, TRAINING_CONFIG
from database import PokemonDatabase
from models.mlp.model import PokemonMLPClassifier
from models.mlp.trainer import normalize_stats


class MLPPredictor:
    """
    Класс для предсказания вероятности победы покемона.
    """
    
    def __init__(self, db: PokemonDatabase = None):
        self.db = db or PokemonDatabase()
        self.device = torch.device(TRAINING_CONFIG['device'])
        self.config = MLP_CONFIG
        
        self.model = PokemonMLPClassifier(
            input_dim=self.config.get('input_dim', 4),
            hidden_dims=self.config.get('hidden_layers', [64, 32, 16]),
            dropout=self.config.get('dropout', 0.3)
        ).to(self.device)
        self.model_loaded = False
    
    def _ensure_model_loaded(self):
        """Ленивая загрузка весов модели."""
        if not self.model_loaded:
            model_path = self.config.get('model_path', Path('models/mlp/mlp_model.pth'))
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model_loaded = True
                print(f"✅ MLP модель загружена")
            else:
                print(f"⚠️ MLP модель не найдена: {model_path}")
    
    def predict_win_probability(self, hp: float, atk: float, 
                                 def_: float, spd: float) -> Optional[Dict]:
        """
        Предсказывает вероятность победы покемона с заданными статами.
        
        Args:
            hp, atk, def_, spd: Характеристики покемона
        
        Returns:
            Dict: Результат предсказания или None если ошибка
        """
        self._ensure_model_loaded()
        
        # Нормализация входных данных
        stats = normalize_stats(hp, atk, def_, spd)
        tensor = torch.FloatTensor(stats).unsqueeze(0).to(self.device)
        
        if not self.model_loaded:
            # Если модель не обучена, возвращаем оценку на основе BST
            bst = hp + atk + def_ + spd
            max_bst = 255 + 190 + 230 + 180  # Максимальные значения
            base_prob = (bst / max_bst) * 100
            return {
                'win_probability': round(base_prob, 1),
                'message': 'Модель не обучена. Показана примерная оценка по BST.',
                'bst': bst
            }
        
        # Предсказание от модели
        with torch.no_grad():
            proba = self.model.predict_proba(tensor).item()
        
        win_probability = round(proba * 100, 1)
        bst = hp + atk + def_ + spd
        
        return {
            'win_probability': win_probability,
            'bst': bst,
            'classification': self._get_classification(win_probability),
            'description': self._get_description(win_probability)
        }
    
    def _get_classification(self, probability: float) -> str:
        """Возвращает классификацию вероятности победы."""
        if probability >= 80:
            return '🟢 Высокая'
        elif probability >= 60:
            return '🔵 Выше среднего'
        elif probability >= 40:
            return '🟡 Средняя'
        elif probability >= 20:
            return '🟠 Ниже среднего'
        else:
            return '🔴 Низкая'
    
    def _get_description(self, probability: float) -> str:
        """Возвращает текстовое описание вероятности."""
        if probability >= 80:
            return "Отличные шансы на победу! Этот покемон сильнее большинства."
        elif probability >= 60:
            return "Хорошие шансы. Покемон имеет преимущество над многими противниками."
        elif probability >= 40:
            return "Равный бой. Шансы примерно 50/50 против случайного противника."
        elif probability >= 20:
            return "Сложный бой. Потребуется стратегия для победы."
        else:
            return "Очень низкие шансы. Рекомендуется усилить покемона."
    
    def batch_predict(self, stats_list: list) -> list:
        """
        Массовое предсказание для нескольких наборов статов.
        
        Args:
            stats_list: Список кортежей [(hp, atk, def, spd), ...]
        
        Returns:
            list: Список результатов предсказаний
        """
        results = []
        for hp, atk, def_, spd in stats_list:
            result = self.predict_win_probability(hp, atk, def_, spd)
            if result:
                results.append(result)
        return results