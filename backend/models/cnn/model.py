"""
CNN (Convolutional Neural Network) для классификации покемонов.

Применение:
- Классификация типа покемона на основе его характеристик, 
  представленных как "изображение" 8×4

Архитектура:
Вход (1, 8, 4) → Conv(16, 3×3) → ReLU → MaxPool → Conv(32, 3×3) → ReLU → 
Flatten → Dense(64) → ReLU → Output(18 классов типов)

Почему это работает:
Хотя данные табличные, свёртки могут выявлять локальные паттерны 
во взаимосвязях характеристик (напр. "высокий attack + низкий defense").
"""

import torch
import torch.nn as nn


class PokemonCNN(nn.Module):
    """
    Сверточная сеть для классификации типов покемонов.
    
    Входные данные должны быть предварительно reshaped в (batch, 1, H, W).
    """
    
    def __init__(
        self, 
        input_channels: int = 1,
        input_size: tuple = (8, 4),  # Высота, Ширина "изображения"
        num_classes: int = 18,  # Количество типов покемонов
        dropout_rate: float = 0.3
    ):
        super(PokemonCNN, self).__init__()
        
        # Первый сверточный блок
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),  # Сохраняем размер
            nn.ReLU(),
            nn.BatchNorm2d(16),  # Стабилизация обучения
            nn.MaxPool2d(2)  # Уменьшаем размер в 2 раза: (8,4) -> (4,2)
        )
        
        # Второй сверточный блок
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)  # (4,2) -> (2,1)
        )
        
        # Полносвязные слои после свёрток
        # После двух пулингов: 32 канала × 2 × 1 = 64 признака
        self.fc = nn.Sequential(
            nn.Linear(32 * 2 * 1, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)  # Выход: вероятности по классам
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход.
        
        Args:
            x: Тензор формы (batch, 1, 8, 4)
            
        Returns:
            torch.Tensor: вероятности классов (batch, 18)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Flatten перед полносвязными слоями
        x = x.view(x.size(0), -1)
        
        return self.fc(x)
    
    def predict_type(self, x: torch.Tensor) -> tuple:
        """
        Предсказывает тип покемона.
        
        Returns:
            Tuple: (индекс класса, имя типа)
        """
        from utils.preprocessing import POKEMON_TYPES
        
        with torch.no_grad():
            probs = self.forward(x)
            idx = torch.argmax(probs, dim=1).item()
            return idx, POKEMON_TYPES[idx]