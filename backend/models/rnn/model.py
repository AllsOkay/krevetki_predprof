"""
RNN/LSTM для прогнозирования характеристик покемонов.

Применение:
- Предсказание характеристик эволюционировавшего покемона 
  на основе последовательности предыдущих форм
- Прогноз "боевого потенциала" на основе истории поколений

Архитектура:
Последовательность векторов → LSTM(32) → LSTM(16) → Dense → Выход

Почему это работает:
Если упорядочить покемонов по эволюционным цепочкам или по поколениям,
рекуррентная сеть может выявить паттерны роста характеристик.
"""

import torch
import torch.nn as nn


class PokemonRNN(nn.Module):
    """
    LSTM-сеть для работы с последовательностями данных о покемонах.
    
    Args:
        input_dim: Размерность одного шага последовательности (26 признаков)
        hidden_dim: Размер скрытого состояния LSTM
        num_layers: Количество слоёв LSTM
        output_dim: Размер выхода (для регрессии или классификации)
        task: 'regression' или 'classification'
    """
    
    def __init__(
        self,
        input_dim: int = 26,
        hidden_dim: int = 32,
        num_layers: int = 2,
        output_dim: int = 1,
        task: str = 'regression',
        dropout: float = 0.2
    ):
        super(PokemonRNN, self).__init__()
        
        self.task = task
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM слой
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Вход: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Полносвязные слои после LSTM
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        if task == 'classification' and output_dim > 1:
            self.fc.add_module('softmax', nn.Softmax(dim=1))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через LSTM.
        
        Args:
            x: Тензор формы (batch, seq_len, input_dim)
            
        Returns:
            torch.Tensor: предсказания
        """
        # LSTM возвращает: (all_hidden, (h_n, c_n))
        # Нам нужно только последнее скрытое состояние: h_n[-1]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Берём выход последнего временного шага
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        
        return self.fc(last_hidden)
    
    def predict_sequence(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Итеративное предсказание последовательности.
        
        Используется для прогнозирования характеристик будущих эволюций.
        
        Args:
            x: Начальная последовательность (batch, seq_len, input_dim)
            steps: Сколько шагов вперёд предсказать
            
        Returns:
            torch.Tensor: предсказанные следующие шаги
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            current_seq = x.clone()
            
            for _ in range(steps):
                # Предсказываем следующий шаг
                next_pred = self.forward(current_seq)  # (batch, output_dim)
                predictions.append(next_pred)
                
                # Добавляем предсказание в последовательность для следующего шага
                # (упрощение: предполагаем output_dim == input_dim)
                if next_pred.shape[1] == current_seq.shape[2]:
                    next_pred = next_pred.unsqueeze(1)  # (batch, 1, features)
                    current_seq = torch.cat([current_seq[:, 1:], next_pred], dim=1)
        
        return torch.stack(predictions, dim=1)  # (batch, steps, output_dim)