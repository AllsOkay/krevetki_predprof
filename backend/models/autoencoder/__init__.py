"""Autoencoder для сжатия признаков и поиска аномалий"""
from .model import PokemonAutoencoder
from .trainer import AutoencoderTrainer

__all__ = ['PokemonAutoencoder', 'AutoencoderTrainer']