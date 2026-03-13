"""
Siamese Network для сравнения похожести покемонов.

Применение:
- Поиск "похожих" покемонов не по евклидову расстоянию, а по семантической близости
- Предсказание исхода боя: насколько два покемона "сопоставимы"

Архитектура:
Два идентичных подсети (shared weights) → 
Каждая обрабатывает одного покемона → 
Сравнение эмбеддингов → Выход: схожесть [0, 1]

Принцип работы:
- Сеть учится отображать похожих покемонов в близкие точки пространства
- Контрастивная функция потерь "отталкивает" непохожие пары
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseSubnetwork(nn.Module):
    """
    Подсеть для обработки одного покемона.
    Используется дважды в сиамской архитектуре с общими весами.
    """
    
    def __init__(self, input_dim: int = 26, embedding_dim: int = 16):
        super(SiameseSubnetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
            # Без активации на выходе — косинусное сходство работает лучше с сырыми векторами
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PokemonSiamese(nn.Module):
    """
    Полная сиамская сеть для сравнения пар покемонов.
    
    Args:
        input_dim: Размерность признаков одного покемона
        embedding_dim: Размерность эмбеддинга для сравнения
        margin: Маржа для контрастивной функции потерь
    """
    
    def __init__(self, input_dim: int = 26, embedding_dim: int = 16, margin: float = 1.0):
        super(PokemonSiamese, self).__init__()
        
        # Одна подсеть с общими весами для обоих входов
        self.subnetwork = SiameseSubnetwork(input_dim, embedding_dim)
        self.margin = margin
        self.embedding_dim = embedding_dim
        
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Получить эмбеддинг одного покемона."""
        return self.subnetwork(x)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        """
        Прямой проход для пары покемонов.
        
        Args:
            x1, x2: Признаки двух покемонов (batch, input_dim)
            
        Returns:
            Tuple: (embedding1, embedding2, similarity_score)
        """
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        
        # Косинусное сходство: чем ближе к 1, тем более похожи
        similarity = F.cosine_similarity(emb1, emb2, dim=1)
        
        return emb1, emb2, similarity
    
    def contrastive_loss(self, emb1: torch.Tensor, emb2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Контрастивная функция потерь для обучения.
        
        Args:
            emb1, emb2: Эмбеддинги пары
            y: Метка (1 = похожи, 0 = непохожи)
            
        Returns:
            torch.Tensor: значение loss
        """
        # Евклидово расстояние между эмбеддингами
        distances = F.pairwise_distance(emb1, emb2)
        
        # Loss = y * d^2 + (1-y) * max(0, margin - d)^2
        loss = y * distances ** 2 + (1 - y) * torch.clamp(self.margin - distances, min=0) ** 2
        
        return loss.mean()
    
    def predict_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """
        Предсказывает схожесть двух покемонов.
        
        Returns:
            float: значение схожести [0, 1]
        """
        self.eval()
        with torch.no_grad():
            _, _, sim = self.forward(x1, x2)
            # Преобразуем косинусное сходство [-1, 1] в [0, 1]
            return (sim.item() + 1) / 2