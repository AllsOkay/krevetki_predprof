"""CNN модель для классификации через "изображение" признаков"""
from .model import PokemonCNN
from .trainer import CNNTrainer

__all__ = ['PokemonCNN', 'CNNTrainer']