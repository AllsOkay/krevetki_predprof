"""
Модуль обучения сверточной нейросети для покемонов.

Подход к обучению: Self-Supervised Learning через аугментации.
Поскольку у нас нет размеченных данных ("покемон А похож на Б"),
мы используем контрастивное обучение:

1. Берем изображение покемона
2. Создаем две аугментированные версии (поворот, цвет, шум)
3. Обучаем сеть выдавать похожие эмбеддинги для аугментаций одного покемона
4. И разные эмбеддинги для разных покемонов

Это позволяет сети научиться извлекать инвариантные визуальные признаки.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # ✅ ВАЖНО: импорт функциональных функций
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional

from config import CNN_CONFIG, TRAINING_CONFIG
from database import PokemonDatabase
from models.cnn.model import PokemonCNNEncoder


# ==================== Аугментации изображений ====================

def random_augment(image: np.ndarray) -> np.ndarray:
    """
    Применяет случайные аугментации к изображению для контрастивного обучения.
    """
    img = Image.fromarray(image.astype(np.uint8))
    
    if np.random.rand() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    angle = np.random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR)
    
    brightness = np.random.uniform(0.8, 1.2)
    contrast = np.random.uniform(0.8, 1.2)
    
    augmented = np.array(img).astype(np.float32) / 255.0
    
    noise = np.random.normal(0, 0.01, augmented.shape)
    augmented = np.clip(augmented + noise, 0, 1)
    
    return augmented


def load_pokemon_sprite(url: str, target_size: Tuple[int, int] = (64, 64)) -> Optional[np.ndarray]:
    """
    Загружает и препроцессит спрайт покемона из URL.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize(target_size, Image.BILINEAR)
        array = np.array(img).astype(np.float32) / 255.0
        
        return array
        
    except Exception as e:
        print(f"⚠️ Не удалось загрузить спрайт {url}: {e}")
        return None


# ==================== Dataset для PyTorch ====================

class PokemonSpriteDataset(Dataset):
    """
    Dataset для загрузки и аугментации спрайтов покемонов.
    """
    
    def __init__(self, pokemon_list: List[dict], input_size: Tuple[int, int] = (64, 64)):
        self.pokemon_list = pokemon_list
        self.input_size = input_size
        self.cache = {}
    
    def __len__(self):
        return len(self.pokemon_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        pokemon = self.pokemon_list[idx]
        sprite_url = pokemon['sprite_url']
        pokemon_id = pokemon['id']
        
        if sprite_url not in self.cache:
            image = load_pokemon_sprite(sprite_url, self.input_size)
            if image is None:
                image = np.random.randn(*self.input_size, 3) * 0.1 + 0.5
                image = np.clip(image, 0, 1)
            self.cache[sprite_url] = image
        else:
            image = self.cache[sprite_url]
        
        view1 = random_augment(image)
        view2 = random_augment(image)
        
        view1 = torch.FloatTensor(view1).permute(2, 0, 1)
        view2 = torch.FloatTensor(view2).permute(2, 0, 1)
        
        return view1, view2, pokemon_id


# ==================== Функция потерь для контрастивного обучения ====================

def contrastive_loss(embedding1: torch.Tensor, embedding2: torch.Tensor, 
                     temperature: float = 0.1) -> torch.Tensor:
    """
    Контрастивная функция потерь (упрощенная версия InfoNCE).
    """
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)
    
    batch_size = embedding1.size(0)
    labels = torch.arange(batch_size, device=embedding1.device)
    logits = (embedding1 @ embedding2.T) / temperature
    loss = F.cross_entropy(logits, labels)
    
    return loss


# ==================== Основной класс для обучения ====================

class CNNTrainer:
    """
    Класс-обертка для обучения CNN модели.
    """
    
    def __init__(self, db: PokemonDatabase = None):
        self.db = db or PokemonDatabase()
        self.device = torch.device(TRAINING_CONFIG['device'])
        self.config = CNN_CONFIG
        
        self.model = PokemonCNNEncoder(
            input_channels=3,
            embedding_dim=self.config['embedding_dim']
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['epochs']
        )
    
    def train(self, pokemon_list: List[dict] = None, verbose: bool = True) -> dict:
        """
        Запускает процесс обучения модели.
        """
        if pokemon_list is None:
            pokemon_list = self.db.get_pokemon_with_sprites()
        
        # Фильтруем покемонов с валидными URL спрайтов
        pokemon_list = [p for p in pokemon_list if p.get('sprite_url')]
        
        if len(pokemon_list) < self.config['min_samples']:
            print(f"⚠️ Недостаточно данных для обучения: {len(pokemon_list)} < {self.config['min_samples']}")
            print(f"💡 Совет: Добавьте больше покемонов через кнопку 'Добавить всех'")
            return {
                'success': False, 
                'reason': 'insufficient_data',
                'current_count': len(pokemon_list),
                'required_count': self.config['min_samples']
            }
        
        # Ограничиваем максимальное количество для ускорения обучения
        if len(pokemon_list) > self.config['max_samples']:
            print(f"ℹ️ Ограничиваем выборку до {self.config['max_samples']} покемонов для скорости")
            step = len(pokemon_list) // self.config['max_samples']
            pokemon_list = pokemon_list[::step][:self.config['max_samples']]
        
        if verbose:
            print(f"🎯 Начало обучения CNN на {len(pokemon_list)} покемонах...")
            print(f"📊 Параметры: epochs={self.config['epochs']}, batch_size={self.config['batch_size']}")
        
        dataset = PokemonSpriteDataset(pokemon_list, self.config['input_size'][:2])
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        self.model.train()
        loss_history = []
        
        for epoch in range(self.config['epochs']):
            epoch_loss = 0.0
            batch_count = 0
            
            for view1, view2, ids in dataloader:
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)
                
                emb1 = self.model(view1)
                emb2 = self.model(view2)
                
                loss = contrastive_loss(emb1, emb2)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            self.scheduler.step()
            avg_loss = epoch_loss / batch_count
            loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Эпоха {epoch+1}/{self.config['epochs']}, Loss: {avg_loss:.4f}")
        
        model_path = self.config['model_path']
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        
        if verbose:
            print(f"✅ Модель сохранена в {model_path}")
        
        self._update_embeddings(pokemon_list)
        
        return {
            'success': True,
            'epochs': self.config['epochs'],
            'samples': len(pokemon_list),
            'final_loss': loss_history[-1],
            'loss_history': loss_history,
            'min_required': self.config['min_samples']
        }
    
    def _update_embeddings(self, pokemon_list: List[dict]):
        """
        Генерирует эмбеддинги для всех покемонов и сохраняет в БД.
        """
        self.model.eval()
        updated_count = 0
        
        for pokemon in pokemon_list:
            try:
                image = load_pokemon_sprite(
                    pokemon['sprite_url'], 
                    self.config['input_size'][:2]
                )
                if image is None:
                    continue
                
                tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
                tensor = tensor.to(self.device)
                
                with torch.no_grad():
                    embedding = self.model.get_embedding(tensor).cpu().numpy().flatten()
                
                self.db.save_embedding(
                    pokemon_id=pokemon['id'],
                    model_type='cnn',
                    embedding=embedding
                )
                updated_count += 1
                
            except Exception as e:
                print(f"⚠️ Ошибка генерации эмбеддинга для {pokemon['name']}: {e}")
                continue
        
        print(f"📦 Обновлены эмбеддинги для {updated_count} покемонов")
    
    def load_model(self, model_path: str = None):
        """Загружает предобученные веса модели."""
        path = model_path or self.config['model_path']
        
        if not path.exists():
            print(f"⚠️ Файл модели не найден: {path}")
            return False
        
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"✅ Модель загружена из {path}")
        return True