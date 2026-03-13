"""
CNN Module - Visual similarity search for Pokemon sprites.
"""
from .model import PokemonCNNEncoder  # ✅ Исправлено: было PokemonCNN
from .trainer import CNNTrainer
from .predictor import CNNPredictor

__all__ = ['PokemonCNNEncoder', 'CNNTrainer', 'CNNPredictor']