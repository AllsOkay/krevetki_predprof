"""
Autoencoder для сжатия признаков и поиска аномальных покемонов.

Применение:
1. Сжатие 26 признаков в 8-мерный вектор (latent space) для визуализации
2. Поиск аномалий: покемоны с высокой ошибкой восстановления — "нестандартные"
3. Улучшение поиска похожих: сравниваем в latent space вместо исходных признаков

Архитектура:
Вход(26) → Encoder(64→32→8) → Latent(8) → Decoder(32→64→26) → Выход(26)

Принцип работы:
- Encoder учится сжимать данные, сохраняя важную информацию
- Decoder учится восстанавливать исходные данные из сжатого представления
- Ошибка восстановления = мера "аномальности" покемона
"""

import torch
import torch.nn as nn


class PokemonAutoencoder(nn.Module):
    """
    Автоэнкодер для работы с данными покемонов.
    
    Args:
        input_dim: Размерность входных данных (26)
        latent_dim: Размерность сжатого представления (по умолчанию 8)
        hidden_dims: Размеры скрытых слоёв энкодера [64, 32]
    """
    
    def __init__(
        self,
        input_dim: int = 26,
        latent_dim: int = 8,
        hidden_dims: list = [64, 32]
    ):
        super(PokemonAutoencoder, self).__init__()
        
        # === ENCODER: сжатие данных ===
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Последний слой энкодера: выход в latent space
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # === DECODER: восстановление данных ===
        # Зеркальная архитектура энкодера
        decoder_layers = []
        prev_dim = latent_dim
        
        # Разворачиваем hidden_dims в обратном порядке
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Выходной слой: восстановление исходной размерности
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Выход в [0, 1] как входные нормализованные данные
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.latent_dim = latent_dim
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Кодирует входные данные в latent representation.
        
        Args:
            x: Тензор (batch, input_dim)
            
        Returns:
            torch.Tensor: сжатые признаки (batch, latent_dim)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Восстанавливает данные из latent representation.
        
        Args:
            z: Латентный вектор (batch, latent_dim)
            
        Returns:
            torch.Tensor: восстановленные данные (batch, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Полный проход: encode → decode.
        
        Returns:
            Tuple: (latent_vector, reconstructed_output)
        """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return z, x_reconstructed
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет ошибку восстановления для каждого образца.
        Используется для детекции аномалий.
        
        Args:
            x: Входные данные
            
        Returns:
            torch.Tensor: MSE error для каждого образца в батче
        """
        self.eval()
        with torch.no_grad():
            _, x_rec = self.forward(x)
            # MSE по признакам для каждого образца
            errors = torch.mean((x - x_rec) ** 2, dim=1)
        return errors