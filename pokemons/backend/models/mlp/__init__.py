"""
MLP Module - Win probability prediction for Pokemon battles.

Architecture: Multi-Layer Perceptron (Fully Connected Network)
Purpose: Predict win probability based on input stats (HP, ATK, DEF, SPD).

Logic:
- Train on existing Pokemon data with simulated battle outcomes
- Input: 4 stats (HP, Attack, Defense, Speed)
- Output: Win probability (0.0 - 1.0)
"""
from .model import PokemonMLPClassifier
from .trainer import MLPTrainer
from .predictor import MLPPredictor

__all__ = ['PokemonMLPClassifier', 'MLPTrainer', 'MLPPredictor']