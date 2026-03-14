"""
Модуль обучения RNN модели для классификации силы покемонов.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from config import RNN_CONFIG, TRAINING_CONFIG
from database import PokemonDatabase
from models.rnn.model import PokemonRNNClassifier


STRENGTH_THRESHOLDS = {
    'weak': 400,
    'medium': 550,
    'strong': 9999
}

CLASS_NAMES = ['weak', 'medium', 'strong']
CLASS_LABELS_RU = {
    'weak': '🔹 Слабый',
    'medium': '🔸 Средний',
    'strong': '🔴 Сильный'
}


def calculate_bst(pokemon: Dict) -> int:
    stats = [
        pokemon.get('hp', 0),
        pokemon.get('attack', 0),
        pokemon.get('defense', 0),
        pokemon.get('special_attack', 0) or pokemon.get('special-attack', 0),
        pokemon.get('special_defense', 0) or pokemon.get('special-defense', 0),
        pokemon.get('speed', 0)
    ]
    return sum(stats)


def get_strength_category(bst: int) -> int:
    if bst < STRENGTH_THRESHOLDS['weak']:
        return 0
    elif bst < STRENGTH_THRESHOLDS['medium']:
        return 1
    else:
        return 2


def normalize_stats(pokemon: Dict) -> np.ndarray:
    max_stats = np.array([255, 190, 230, 194, 230, 180], dtype=np.float32)
    
    stats = np.array([
        pokemon.get('hp', 0),
        pokemon.get('attack', 0),
        pokemon.get('defense', 0),
        pokemon.get('special_attack', 0) or pokemon.get('special-attack', 0),
        pokemon.get('special_defense', 0) or pokemon.get('special-defense', 0),
        pokemon.get('speed', 0)
    ], dtype=np.float32)
    
    normalized = np.clip(stats / max_stats, 0, 1)
    return normalized.reshape(-1, 1)


class PokemonStrengthDataset(Dataset):
    def __init__(self, pokemon_list: List[Dict]):
        self.pokemon_list = pokemon_list
    
    def __len__(self):
        return len(self.pokemon_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pokemon = self.pokemon_list[idx]
        sequence = normalize_stats(pokemon)
        bst = calculate_bst(pokemon)
        label = get_strength_category(bst)
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])[0]


class RNNTrainer:
    def __init__(self, db: PokemonDatabase = None):
        self.db = db or PokemonDatabase()
        self.device = torch.device(TRAINING_CONFIG['device'])
        self.config = RNN_CONFIG
        
        self.model = PokemonRNNClassifier(
            input_size=1,
            hidden_size=self.config.get('hidden_dim', 32),
            num_layers=2,
            num_classes=3,
            dropout=0.3
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('epochs', 30)
        )
    
    def train(self, pokemon_list: List[Dict] = None, verbose: bool = True) -> dict:
        if pokemon_list is None:
            pokemon_list = self.db.get_all_pokemon()
        
        pokemon_list = [p for p in pokemon_list if p.get('hp') is not None]
        
        min_samples = self.config.get('min_samples', 50)
        if len(pokemon_list) < min_samples:
            return {
                'success': False,
                'reason': 'insufficient_data',
                'current_count': len(pokemon_list),
                'required_count': min_samples
            }
        
        if verbose:
            print(f"🎯 Начало обучения RNN на {len(pokemon_list)} покемонах...")
        
        dataset = PokemonStrengthDataset(pokemon_list)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        self.model.train()
        loss_history = []
        acc_history = []
        epochs = self.config.get('epochs', 30)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(sequences)
                loss = self.criterion(logits, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            self.scheduler.step()
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct / total
            loss_history.append(avg_loss)
            acc_history.append(accuracy)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Эпоха {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2%}")
        
        # ✅ ВАЖНО: Сохранение метрик в БД (этого не было!)
        metrics_for_db = []
        for epoch, (loss, acc) in enumerate(zip(loss_history, acc_history), start=1):
            metrics_for_db.append({
                'epoch': epoch,
                'loss': float(loss),
                'accuracy': float(acc),
                'metric_name': 'classification',
                'metric_value': float(acc)
            })
        
        self.db.save_training_metrics('rnn', metrics_for_db)
        print(f"📊 Метрики обучения сохранены в БД ({len(metrics_for_db)} записей)")
        
        # Сохранение модели
        model_path = self.config.get('model_path', Path('models/rnn/rnn_model.pth'))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        
        if verbose:
            print(f"✅ Модель сохранена в {model_path}")
            print(f"📈 Финальная точность: {acc_history[-1]:.2%}")
        
        self._update_predictions(pokemon_list)
        
        return {
            'success': True,
            'epochs': epochs,
            'samples': len(dataset),
            'pokemon_count': len(pokemon_list),
            'final_loss': loss_history[-1],
            'final_accuracy': acc_history[-1],
            'loss_history': loss_history,
            'accuracy_history': acc_history
        }
    
    def _update_predictions(self, pokemon_list: List[Dict]):
        self.model.eval()
        updated_count = 0
        
        for pokemon in pokemon_list:
            try:
                sequence = normalize_stats(pokemon)
                tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    proba = self.model.predict_proba(tensor).cpu().numpy().flatten()
                    pred_class = np.argmax(proba)
                
                result_data = {
                    'class': CLASS_NAMES[pred_class],
                    'class_ru': CLASS_LABELS_RU[CLASS_NAMES[pred_class]],
                    'probabilities': proba.tolist(),
                    'confidence': float(proba[pred_class]),
                    'bst': calculate_bst(pokemon)
                }
                
                self.db.save_model_result(
                    pokemon_id=pokemon['id'],
                    model_type='rnn',
                    result_data=result_data,
                    confidence=result_data['confidence']
                )
                updated_count += 1
                
            except Exception as e:
                print(f"⚠️ Ошибка: {e}")
                continue
        
        print(f"📦 Обновлены предсказания для {updated_count} покемонов")
    
    def load_model(self, model_path: str = None):
        path = model_path or self.config.get('model_path', Path('models/rnn/rnn_model.pth'))
        if not path.exists():
            return False
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        return True