"""
Архитектура автоэнкодера для детекции аномалий в характеристиках покемонов.
"""
import torch
import torch.nn as nn


class PokemonAutoencoder(nn.Module):
    """
    Автоэнкодер для анализа характеристик покемонов.
    """
    
    def __init__(self, input_dim: int = 26, latent_dim: int = 8, 
                 hidden_dims: list = None, dropout: float = 0.2):
        super(PokemonAutoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [16, 12]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # ==================== Энкодер (сжатие) ====================
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # ✅ ЗАМЕНЕНО: BatchNorm -> LayerNorm
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # ==================== Декодер (восстановление) ====================
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # ✅ ЗАМЕНЕНО: BatchNorm -> LayerNorm
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов методом Xavier."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Сжатие входных данных в латентное представление."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Восстановление данных из латентного представления."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Полный проход через автоэнкодер."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return z, x_reconstructed
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Вычисляет ошибку восстановления для каждого образца."""
        with torch.no_grad():
            _, x_reconstructed = self.forward(x)
            error = torch.mean((x - x_reconstructed) ** 2, dim=1)
            return error