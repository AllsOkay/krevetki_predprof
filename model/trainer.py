import numpy as np
from typing import Dict, List, Optional, Tuple
from config import Config
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model, config: Optional[Dict] = None):
        self.model = model
        self.config = config or Config.TRAIN_CONFIG.copy()
        
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        self.train_data = None
        self.val_data = None
        
        self.best_val_accuracy = 0
        self.patience_counter = 0
        
    def set_data(self, train_x: np.ndarray, train_y: np.ndarray,
                 val_x: Optional[np.ndarray] = None, 
                 val_y: Optional[np.ndarray] = None) -> None:
        self.train_data = {'x': train_x, 'y': train_y}
        
        if val_x is not None and val_y is not None:
            self.val_data = {'x': val_x, 'y': val_y}
            logger.info(f"✓ Валидационные данные: {len(val_x)} образцов")
    
    def _create_batches(self, x: np.ndarray, y: np.ndarray, 
                        batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(x)
        batches = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batches.append((x[start_idx:end_idx], y[start_idx:end_idx]))
        
        return batches
    
    def _train_epoch(self, batches: List[Tuple[np.ndarray, np.ndarray]], 
                     shuffle: bool = True) -> Dict[str, float]:
        if shuffle:
            np.random.shuffle(batches)
        
        epoch_losses = []
        epoch_accuracies = []
        
        for batch_x, batch_y in batches:
            loss, acc = self.model.train_step(batch_x, batch_y)
            epoch_losses.append(loss)
            epoch_accuracies.append(acc)
        
        return {
            'loss': np.mean(epoch_losses),
            'accuracy': np.mean(epoch_accuracies)
        }
    
    def _evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        return self.model.evaluate(x, y)
    
    def train(self, epochs: Optional[int] = None, 
              batch_size: Optional[int] = None,
              verbose: bool = True) -> Dict[str, List[float]]:
        epochs = epochs or self.config['epochs']
        batch_size = batch_size or self.config['batch_size']
        patience = self.config.get('early_stopping_patience', 5)
        save_best = self.config.get('save_best_only', True)
        
        if self.train_data is None:
            raise ValueError("Training data not set. Call set_data() first.")
        
        train_batches = self._create_batches(
            self.train_data['x'], 
            self.train_data['y'], 
            batch_size
        )
        
        logger.info(f"Начало обучения: {epochs} эпох, {len(train_batches)} батчей/эпоху")
        
        for epoch in range(epochs):
            train_metrics = self._train_epoch(train_batches)
            
            val_metrics = {}
            if self.val_data is not None:
                val_metrics = self._evaluate(
                    self.val_data['x'], 
                    self.val_data['y']
                )
            
            self.history['loss'].append(train_metrics['loss'])
            self.history['accuracy'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics.get('loss', 0))
            self.history['val_accuracy'].append(val_metrics.get('accuracy', 0))
            
            if verbose:
                val_info = ""
                if val_metrics:
                    val_info = f" | Val: {val_metrics['accuracy']:.4f} loss: {val_metrics['loss']:.4f}"
                print(f"Epoch {epoch+1}/{epochs} "
                      f"| Train acc: {train_metrics['accuracy']:.4f} loss: {train_metrics['loss']:.4f}"
                      f"{val_info}")
            
            if self.val_data is not None and save_best:
                current_val_acc = val_metrics.get('accuracy', 0)
                
                if current_val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = current_val_acc
                    self.patience_counter = 0
                    if verbose:
                        print(f"Новая лучшая модель! Сохраняем...")
                    self.model.save(Config.MODEL_SAVE_PATH)
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping: нет улучшений {patience} эпох")
                        break
        
        if not save_best or self.val_data is None:
            self.model.save(Config.MODEL_SAVE_PATH)
            logger.info(f"Модель сохранена: {Config.MODEL_SAVE_PATH}")
        
        self.model.is_trained = True
        logger.info(f"Обучение завершено! Лучшая вал. точность: {self.best_val_accuracy:.4f}")
        
        return self.history
    
    def get_training_plot_data(self) -> Dict[str, List]:
        """
        Возвращает данные для построения графика обучения.
        
        :return: Словарь с данными для графиков точности и потерь
        """
        return {
            'epochs': list(range(1, len(self.history['accuracy']) + 1)),
            'train_accuracy': self.history['accuracy'],
            'val_accuracy': self.history['val_accuracy'],
            'train_loss': self.history['loss'],
            'val_loss': self.history['val_loss']
        }