"""
Модуль инференса для классификации силы покемонов.

Использует предобученную RNN для:
1. Извлечения последовательности статов покемона
2. Классификации на 3 класса (weak, medium, strong)
3. Возврата категории с уверенностью модели
"""
import torch
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

from config import RNN_CONFIG, TRAINING_CONFIG
from database import PokemonDatabase
from models.rnn.model import PokemonRNNClassifier
from models.rnn.trainer import (
    normalize_stats, calculate_bst, get_strength_category,
    CLASS_NAMES, CLASS_LABELS_RU
)


class RNNPredictor:
    """
    Класс для классификации силы покемонов.
    """
    
    def __init__(self, db: PokemonDatabase = None):
        self.db = db or PokemonDatabase()
        self.device = torch.device(TRAINING_CONFIG['device'])
        self.config = RNN_CONFIG
        
        self.model = PokemonRNNClassifier(
            input_size=1,
            hidden_size=self.config.get('hidden_dim', 32),
            num_layers=2,
            num_classes=3,
            dropout=0.3
        ).to(self.device)
        self.model_loaded = False
    
    def _ensure_model_loaded(self):
        """Ленивая загрузка весов модели."""
        if not self.model_loaded:
            model_path = self.config.get('model_path', Path('models/rnn/rnn_model.pth'))
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model_loaded = True
                print(f"✅ RNN модель загружена")
            else:
                print(f"⚠️ RNN модель не найдена: {model_path}")
    
    def classify(self, pokemon_name: str) -> Optional[Dict]:
        """
        Классифицирует покемона по категории силы.
        
        Args:
            pokemon_name: Имя покемона
        
        Returns:
            Dict: Результат классификации или None если ошибка
        """
        # Проверяем БД
        pokemon = self.db.get_pokemon_by_name(pokemon_name)
        if not pokemon:
            return None
        
        # Проверяем кэш предсказаний
        cached = self.db.get_model_result(pokemon['id'], model_type='rnn')
        if cached:
            return cached
        
        # Вычисляем заново
        self._ensure_model_loaded()
        
        if not self.model_loaded:
            # Если модель не обучена, используем правило на основе BST
            bst = calculate_bst(pokemon)
            class_idx = get_strength_category(bst)
            return {
                'class': CLASS_NAMES[class_idx],
                'class_ru': CLASS_LABELS_RU[CLASS_NAMES[class_idx]],
                'confidence': 1.0,
                'bst': bst,
                'probabilities': [0.33, 0.33, 0.34]
            }
        
        # Получаем предсказание от модели
        sequence = normalize_stats(pokemon)
        tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            proba = self.model.predict_proba(tensor).cpu().numpy().flatten()
            pred_class = np.argmax(proba)
        
        bst = calculate_bst(pokemon)
        
        result = {
            'class': CLASS_NAMES[pred_class],
            'class_ru': CLASS_LABELS_RU[CLASS_NAMES[pred_class]],
            'confidence': float(proba[pred_class]),
            'bst': bst,
            'probabilities': proba.tolist()
        }
        
        # Кэшируем в БД
        self.db.save_model_result(
            pokemon_id=pokemon['id'],
            model_type='rnn',
            result_data=result,
            confidence=result['confidence']
        )
        
        return result
    
    def batch_classify(self, pokemon_names: List[str]) -> Dict[str, Dict]:
        """
        Массовая классификация нескольких покемонов.
        """
        results = {}
        
        for name in pokemon_names:
            result = self.classify(name)
            if result:
                results[name] = result
        
        return results
    
    def get_distribution(self) -> Dict:
        """
        Возвращает распределение покемонов по классам силы.
        """
        all_pokemon = self.db.get_all_pokemon()
        
        distribution = {'weak': 0, 'medium': 0, 'strong': 0}
        
        for pokemon in all_pokemon:
            result = self.db.get_model_result(pokemon['id'], model_type='rnn')
            if result and 'class' in result:
                distribution[result['class']] += 1
            else:
                bst = calculate_bst(pokemon)
                class_idx = get_strength_category(bst)
                distribution[CLASS_NAMES[class_idx]] += 1
        
        total = sum(distribution.values())
        
        return {
            'distribution': distribution,
            'total': total,
            'percentages': {
                k: round(v / total * 100, 1) if total > 0 else 0
                for k, v in distribution.items()
            }
        }