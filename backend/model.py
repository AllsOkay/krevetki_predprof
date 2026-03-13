import torch
import torch.nn as nn

class PokemonAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super(PokemonAutoencoder, self).__init__()
        
        # Кодировщик (Encoder): Сжимает данные
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim) # Сжатое представление
        )
        
        # Декодировщик (Decoder): Восстанавливает данные
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, input_dim),
            nn.Sigmoid() # Выход нормализован [0, 1]
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def get_embedding(self, x):
        """Получить только вектор представления (для поиска похожих)"""
        with torch.no_grad():
            return self.encoder(x)