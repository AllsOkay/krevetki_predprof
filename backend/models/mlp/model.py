"""
MLP (Multi-Layer Perceptron) для задач классификации и регрессии.

Применение в проекте:
- Классификация: предсказание основного типа покемона по характеристикам
- Регрессия: предсказание общего боевого рейтинга (сумма всех статов)

Архитектура:
Вход (26 признаков) → Dense(64) → ReLU → Dropout → Dense(32) → ReLU → Выход
"""

import torch
import torch.nn as nn


class PokemonMLP(nn.Module):
    """
    Полносвязная нейросеть для работы с табличными данными покемонов.
    
    Args:
        input_dim: Количество входных признаков (по умолчанию 26: 8 статов + 18 типов)
        hidden_dims: Список размеров скрытых слоёв [64, 32]
        output_dim: Размер выхода (1 для регрессии, N для классификации)
        task: 'regression' или 'classification'
        dropout_rate: Вероятность отключения нейронов для регуляризации
    """
    
    def __init__(
        self, 
        input_dim: int = 26, 
        hidden_dims: list = [64, 32], 
        output_dim: int = 1,
        task: str = 'regression',
        dropout_rate: float = 0.2
    ):
        super(PokemonMLP, self).__init__()
        
        self.task = task
        
        # Собираем слои динамически
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # Функция активации
            layers.append(nn.Dropout(dropout_rate))  # Регуляризация
            prev_dim = hidden_dim
        
        # Выходной слой
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if task == 'classification' and output_dim > 1:
            # Для многоклассовой классификации добавляем softmax
            layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.
        
        Args:
            x: Входной тензор (batch_size, input_dim)
            
        Returns:
            torch.Tensor: предсказания сети
        """
        return self.network(x)
    
    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вспомогательный метод для получения класса (для классификации).
        
        Args:
            x: Входные данные
            
        Returns:
            torch.Tensor: индексы классов
        """
        if self.task == 'classification':
            with torch.no_grad():
                output = self.network(x)
                return torch.argmax(output, dim=1)
        raise ValueError("Метод доступен только для задачи классификации")