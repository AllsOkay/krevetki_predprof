"""
Модуль инференса для поиска визуально похожих покемонов.
"""
import torch
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from config import CNN_CONFIG, TRAINING_CONFIG
from database import PokemonDatabase
from models.cnn.model import PokemonCNNEncoder
from models.cnn.trainer import load_pokemon_sprite


class CNNPredictor:
    """
    Класс для поиска визуально похожих покемонов.
    """
    
    def __init__(self, db: PokemonDatabase = None):
        self.db = db or PokemonDatabase()
        self.device = torch.device(TRAINING_CONFIG['device'])
        self.config = CNN_CONFIG
        
        self.model = PokemonCNNEncoder(
            input_channels=3,
            embedding_dim=self.config['embedding_dim']
        ).to(self.device)
        self.model_loaded = False
    
    def _ensure_model_loaded(self):
        """Ленивая загрузка весов модели при первом использовании."""
        if not self.model_loaded:
            self.model.load_state_dict(
                torch.load(self.config['model_path'], map_location=self.device)
            )
            self.model.eval()
            self.model_loaded = True
    
    def get_embedding(self, pokemon_name: str) -> Optional[np.ndarray]:
        """
        Получает или вычисляет эмбеддинг для покемона по имени.
        """
        pokemon = self.db.get_pokemon_by_name(pokemon_name)
        if not pokemon or not pokemon.get('sprite_url'):
            return None
        
        cached = self.db.get_embedding(pokemon['id'], model_type='cnn')
        if cached is not None:
            return cached
        
        self._ensure_model_loaded()
        
        image = load_pokemon_sprite(
            pokemon['sprite_url'],
            self.config['input_size'][:2]
        )
        if image is None:
            return None
        
        tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            embedding = self.model.get_embedding(tensor).cpu().numpy().flatten()
        
        self.db.save_embedding(pokemon['id'], 'cnn', embedding)
        
        return embedding
    
    def find_similar(self, pokemon_name: str, top_k: int = 5) -> List[Dict]:
        """
        Находит k визуально похожих покемонов.
        """
        target_embedding = self.get_embedding(pokemon_name)
        if target_embedding is None:
            return []
        
        all_embeddings = self.db.get_all_embeddings(model_type='cnn')
        
        target_pokemon = self.db.get_pokemon_by_name(pokemon_name)
        if not target_pokemon:
            return []
        
        target_id = target_pokemon['id']
        similarities = []
        
        for pid, embedding in all_embeddings.items():
            if pid == target_id:
                continue
            
            sim = np.dot(target_embedding, embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(embedding) + 1e-8
            )
            
            conn = self.db._get_connection()
            name_row = conn.execute('SELECT name FROM pokemon WHERE id = ?', (pid,)).fetchone()
            conn.close()
            
            if name_row:
                similarities.append({
                    'id': pid,
                    'name': name_row['name'],
                    'similarity': float(sim)
                })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def batch_find_similar(self, pokemon_names: List[str], top_k: int = 5) -> Dict[str, List[Dict]]:
        """
        Массовый поиск похожих для нескольких покемонов.
        """
        results = {}
        all_embeddings = self.db.get_all_embeddings(model_type='cnn')
        
        for name in pokemon_names:
            target_embedding = self.get_embedding(name)
            if target_embedding is None:
                results[name] = []
                continue
            
            target_pokemon = self.db.get_pokemon_by_name(name)
            if not target_pokemon:
                results[name] = []
                continue
            
            target_id = target_pokemon['id']
            similarities = []
            
            for pid, embedding in all_embeddings.items():
                if pid == target_id:
                    continue
                
                sim = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding) + 1e-8
                )
                
                conn = self.db._get_connection()
                name_row = conn.execute('SELECT name FROM pokemon WHERE id = ?', (pid,)).fetchone()
                conn.close()
                
                if name_row:
                    similarities.append({
                        'id': pid,
                        'name': name_row['name'],
                        'similarity': float(sim)
                    })
            
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            results[name] = similarities[:top_k]
        
        return results