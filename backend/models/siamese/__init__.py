"""Siamese Network для сравнения похожести покемонов"""
from .model import PokemonSiamese
from .trainer import SiameseTrainer

__all__ = ['PokemonSiamese', 'SiameseTrainer']