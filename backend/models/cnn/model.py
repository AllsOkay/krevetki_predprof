"""
Архитектура сверточной нейросети (CNN) для извлечения визуальных признаков покемонов.

Эта сеть работает как "энкодер": она принимает изображение спрайта покемона
и преобразует его в компактный вектор (эмбеддинг), который сохраняет
визуальные особенности: форму, цвета, пропорции.

Для поиска похожих покемонов мы сравниваем эти векторы через косинусное сходство.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PokemonCNNEncoder(nn.Module):
    """
    Сверточная нейросеть для кодирования изображений покемонов.
    
    Архитектура:
    1. Несколько сверточных слоев с пулингом для извлечения признаков
    2. Полносвязные слои для сжатия в эмбеддинг
    3. BatchNorm и Dropout для регуляризации
    
    Вход: Изображение 64x64x3 (RGB)
    Выход: Вектор размерности embedding_dim (по умолчанию 32)
    """
    
    def __init__(self, input_channels: int = 3, embedding_dim: int = 32):
        """
        Инициализация архитектуры сети.
        
        Args:
            input_channels: Количество каналов входного изображения (3 для RGB)
            embedding_dim: Размерность выходного векторного представления
        """
        super(PokemonCNNEncoder, self).__init__()
        
        # ==================== Блок 1: Извлечение низкоуровневых признаков ====================
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),  # 64x64 -> 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                        # 64x64 -> 32x32
            nn.Dropout(0.1)
        )
        
        # ==================== Блок 2: Извлечение среднеуровневых признаков ====================
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32x32 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # 32x32 -> 16x16
            nn.Dropout(0.2)
        )
        
        # ==================== Блок 3: Извлечение высокоуровневых признаков ====================
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16x16 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # 16x16 -> 8x8
            nn.Dropout(0.3)
        )
        
        # ==================== Полносвязные слои для сжатия в эмбеддинг ====================
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),  # Сжимаем 4096 -> 128
            nn.ReLU(inplace=True),
            nn.Dropout(0.4)
        )
        
        # Финальный слой: проекция в пространство эмбеддингов
        self.embedding = nn.Linear(128, embedding_dim)  # 128 -> embedding_dim
        
        # Инициализация весов для стабильности обучения
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов методом Kaiming (рекомендуется для ReLU)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.
        
        Args:
            x: Тензор изображения размера [batch_size, channels, height, width]
        
        Returns:
            torch.Tensor: Вектор эмбеддинга размера [batch_size, embedding_dim]
        """
        x = self.conv1(x)  # [B, 3, 64, 64] -> [B, 16, 32, 32]
        x = self.conv2(x)  # [B, 16, 32, 32] -> [B, 32, 16, 16]
        x = self.conv3(x)  # [B, 32, 16, 16] -> [B, 64, 8, 8]
        
        x = x.view(x.size(0), -1)  # [B, 64, 8, 8] -> [B, 4096]
        
        x = self.fc1(x)            # [B, 4096] -> [B, 128]
        
        embedding = self.embedding(x)  # [B, 128] -> [B, embedding_dim]
        
        return embedding
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Удобный метод для получения эмбеддинга без градиентов (для инференса).
        """
        with torch.no_grad():
            return self.forward(x)