"""RNN/LSTM модель для работы с последовательностями"""
from .model import PokemonRNN
from .trainer import RNNTrainer

__all__ = ['PokemonRNN', 'RNNTrainer']