"""MLP модель для классификации и регрессии"""
from .model import PokemonMLP
from .trainer import MLPTrainer

__all__ = ['PokemonMLP', 'MLPTrainer']