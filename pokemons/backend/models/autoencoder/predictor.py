"""
Модуль инференса для расчёта "обычности" покемонов.

Использует предобученный автоэнкодер для:
1. Извлечения признаков покемона
2. Вычисления ошибки восстановления
3. Конвертации ошибки в процент "обычности"
"""
import torch
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

from config import AE_CONFIG, TRAINING_CONFIG
from database import PokemonDatabase
from models.autoencoder.model import PokemonAutoencoder
from models.autoencoder.trainer import prepare_pokemon_features


class AEPredictor:
    """
    Класс для расчёта "обычности" покемонов.
    """
    
    def __init__(self, db: PokemonDatabase = None):
        self.db = db or PokemonDatabase()
        self.device = torch.device(TRAINING_CONFIG['device'])
        self.config = AE_CONFIG
        
        self.model = PokemonAutoencoder(
            input_dim=self.config.get('input_dim', 26),
            latent_dim=self.config.get('latent_dim', 8),
            hidden_dims=self.config.get('hidden_dims', [16, 12]),
            dropout=self.config.get('dropout', 0.2)
        ).to(self.device)
        self.model_loaded = False
    
    def _ensure_model_loaded(self):
        """Ленивая загрузка весов модели."""
        if not self.model_loaded:
            model_path = self.config.get('model_path', Path('models/autoencoder/ae_model.pth'))
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.model_loaded = True
                print(f"✅ Автоэнкодер загружен")
            else:
                print(f"⚠️ Автоэнкодер не найден: {model_path}")
    
    def get_ordinariness(self, pokemon_name: str) -> Optional[Dict]:
        """
        Вычисляет "обычность" покемона в процентах.
        
        Высокая обычность (%) = характеристики похожи на большинство покемонов
        Низкая обычность (%) = уникальные/аномальные характеристики
        
        Args:
            pokemon_name: Имя покемона
        
        Returns:
            Dict: Результат с обычностью или None если ошибка
        """
        # Проверяем БД
        pokemon = self.db.get_pokemon_by_name(pokemon_name)
        if not pokemon:
            return None
        
        # Проверяем кэш предсказаний
        cached = self.db.get_model_result(pokemon['id'], model_type='autoencoder')
        if cached:
            return cached
        
        # Вычисляем заново
        self._ensure_model_loaded()
        
        if not self.model_loaded:
            # Если модель не обучена, возвращаем заглушку
            return {
                'ordinariness': 50.0,
                'message': 'Модель не обучена. Обычность не определена.',
                'reconstruction_error': None
            }
        
        # Получаем признаки и вычисляем ошибку
        features = prepare_pokemon_features(pokemon)
        tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            error = self.model.get_reconstruction_error(tensor).item()
        
        # Для конвертации в % нужны мин/макс ошибки из обучающей выборки
        # Если нет в кэше, используем эвристику
        min_err = 0.001  # Примерные значения
        max_err = 0.5
        err_range = max_err - min_err
        
        ordinariness = 100 - ((error - min_err) / err_range * 100)
        ordinariness = round(max(0, min(100, ordinariness)), 1)
        
        result = {
            'ordinariness': ordinariness,
            'reconstruction_error': round(error, 4),
            'description': self._get_description(ordinariness)
        }
        
        # Кэшируем в БД
        self.db.save_model_result(
            pokemon_id=pokemon['id'],
            model_type='autoencoder',
            result_data=result,
            confidence=ordinariness / 100
        )
        
        return result
    
    def _get_description(self, ordinariness: float) -> str:
        """Возвращает текстовое описание уровня обычности."""
        if ordinariness >= 80:
            return "🟢 Очень обычный (характеристики как у большинства)"
        elif ordinariness >= 60:
            return "🔵 Обычный (типичные характеристики)"
        elif ordinariness >= 40:
            return "🟡 Необычный (есть уникальные черты)"
        elif ordinariness >= 20:
            return "🟠 Редкий (аномальные характеристики)"
        else:
            return "🔴 Уникальный (экстремально редкие характеристики)"
    
    def batch_get_ordinariness(self, pokemon_names: List[str]) -> Dict[str, Dict]:
        """Массовый расчёт обычности для нескольких покемонов."""
        results = {}
        for name in pokemon_names:
            result = self.get_ordinariness(name)
            if result:
                results[name] = result
        return results
    
    def get_statistics(self) -> Dict:
        """Возвращает статистику по обычности всех покемонов."""
        all_pokemon = self.db.get_all_pokemon()
        
        ordinariness_values = []
        
        for pokemon in all_pokemon:
            result = self.db.get_model_result(pokemon['id'], model_type='autoencoder')
            if result and 'ordinariness' in result:
                ordinariness_values.append(result['ordinariness'])
        
        if not ordinariness_values:
            return {'message': 'Нет данных для статистики'}
        
        return {
            'count': len(ordinariness_values),
            'mean': round(np.mean(ordinariness_values), 1),
            'median': round(np.median(ordinariness_values), 1),
            'std': round(np.std(ordinariness_values), 1),
            'min': round(min(ordinariness_values), 1),
            'max': round(max(ordinariness_values), 1),
            'very_ordinary': sum(1 for v in ordinariness_values if v >= 80),
            'ordinary': sum(1 for v in ordinariness_values if 60 <= v < 80),
            'unusual': sum(1 for v in ordinariness_values if 40 <= v < 60),
            'rare': sum(1 for v in ordinariness_values if 20 <= v < 40),
            'unique': sum(1 for v in ordinariness_values if v < 20)
        }