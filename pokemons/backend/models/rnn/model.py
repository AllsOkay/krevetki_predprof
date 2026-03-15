"""
Архитектура рекуррентной нейросети (LSTM) для классификации силы покемонов.

Эта сеть анализирует последовательность из 6 основных характеристик:
HP → Attack → Defense → Sp.Atk → Sp.Def → Speed

LSTM хорошо подходит для таких задач, потому что:
1. Учитывает порядок характеристик (важно для определения роли)
2. Запоминает долгосрочные зависимости между статами
3. Может выявить паттерны (например, высокий Sp.Atk при низком Defense)

Вход: Последовательность из 6 нормализованных характеристик
Выход: Вероятности для 3 классов (слабый, средний, сильный)
"""
import torch
import torch.nn as nn


class PokemonRNNClassifier(nn.Module):
    """
    LSTM-сеть для классификации силы покемонов.
    
    Архитектура:
    1. LSTM слой для обработки последовательности статов
    2. Полносвязные слои для классификации
    3. Dropout для регуляризации
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 32, 
                 num_layers: int = 2, num_classes: int = 3, dropout: float = 0.3):
        """
        Инициализация архитектуры сети.
        
        Args:
            input_size: Размерность входа на каждом шаге (1 для одного стата)
            hidden_size: Размерность скрытого состояния LSTM
            num_layers: Количество LSTM слоёв
            num_classes: Количество классов (3: weak, medium, strong)
            dropout: Коэффициент Dropout для регуляризации
        """
        super(PokemonRNNClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # ==================== LSTM Слой ====================
        # Обрабатывает последовательность из 6 характеристик
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Вход: [batch, seq_len, features]
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Двунаправленный LSTM для лучшего контекста
        )
        
        # ==================== Полносвязные слои ====================
        # После LSTM: hidden_size * 2 (bidirectional)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # *2 для bidirectional
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Выходной слой: 3 класса (weak, medium, strong)
        self.output = nn.Linear(32, num_classes)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов для стабильности обучения."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name and len(param.shape) > 1:
                nn.init.kaiming_normal_(param.data, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.
        
        Args:
            x: Тензор последовательности размера [batch_size, seq_len=6, input_size=1]
               Пример: [[[HP], [Attack], [Defense], [Sp.Atk], [Sp.Def], [Speed]]]
        
        Returns:
            torch.Tensor: Логиты для 3 классов размера [batch_size, num_classes]
        """
        # ==================== LSTM Обработка ====================
        # x: [batch, 6, 1]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch, 6, hidden*2]
        # h_n: [num_layers*2, batch, hidden]
        
        # ==================== Объединяем скрытые состояния ====================
        # Берём последнее скрытое состояние из обоих направлений
        # h_n: [num_layers*2, batch, hidden] -> берём последние 2 слоя
        h_forward = h_n[-2, :, :]  # Последнее состояние прямого направления
        h_backward = h_n[-1, :, :]  # Последнее состояние обратного направления
        
        # Конкатенируем направления
        hidden_concat = torch.cat((h_forward, h_backward), dim=1)
        # hidden_concat: [batch, hidden*2]
        
        # ==================== Полносвязные слои ====================
        x = self.fc1(hidden_concat)  # [batch, 64]
        x = self.fc2(x)              # [batch, 32]
        
        # ==================== Выходной слой ====================
        logits = self.output(x)      # [batch, 3]
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Метод для инференса (без градиентов).
        
        Args:
            x: Тензор последовательности
        
        Returns:
            torch.Tensor: Индексы предсказанных классов
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
            return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Возвращает вероятности для каждого класса.
        
        Args:
            x: Тензор последовательности
        
        Returns:
            torch.Tensor: Вероятности размера [batch_size, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities