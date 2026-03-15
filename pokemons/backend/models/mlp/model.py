"""
Архитектура полносвязной нейросети (MLP) для предсказания вероятности победы.

Принцип работы:
1. Входные данные: 4 характеристики (HP, ATK, DEF, SPD) нормализованные [0, 1]
2. Несколько скрытых слоёв с ReLU активацией
3. Выходной слой с Sigmoid для вероятности победы (0-100%)

Модель обучается на симулированных боях между покемонами из базы данных.
Победа определяется сравнением BST (Base Stat Total) с элементом случайности.
"""
import torch
import torch.nn as nn


class PokemonMLPClassifier(nn.Module):
    """
    Полносвязная нейросеть для классификации вероятности победы.
    
    Архитектура:
    - Вход: 4 нормализованные характеристики
    - Скрытые слои: [64, 32, 16] с Dropout и BatchNorm
    - Выход: 1 значение (вероятность победы 0.0-1.0)
    """
    
    def __init__(self, input_dim: int = 4, hidden_dims: list = None, 
                 dropout: float = 0.3):
        """
        Инициализация архитектуры MLP.
        
        Args:
            input_dim: Размерность входа (4 стата)
            hidden_dims: Список размерностей скрытых слоёв
            dropout: Коэффициент Dropout для регуляризации
        """
        super(PokemonMLPClassifier, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        
        self.input_dim = input_dim
        
        # ==================== Построение слоёв ====================
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Выходной слой: 1 нейрон с Sigmoid для вероятности
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов методом Xavier."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.
        
        Args:
            x: Тензор размера [batch_size, 4]
        
        Returns:
            torch.Tensor: Вероятность победы размера [batch_size, 1]
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Предсказание вероятности победы (без градиентов).
        
        Args:
            x: Тензор размера [batch_size, 4]
        
        Returns:
            torch.Tensor: Вероятность победы
        """
        with torch.no_grad():
            return self.forward(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Бинарное предсказание (победа/поражение).
        
        Args:
            x: Тензор размера [batch_size, 4]
            threshold: Порог для классификации
        
        Returns:
            torch.Tensor: 1 (победа) или 0 (поражение)
        """
        with torch.no_grad():
            proba = self.forward(x)
            return (proba >= threshold).long()