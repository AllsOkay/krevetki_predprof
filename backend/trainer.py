import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import os
from model import PokemonAutoencoder

DB_PATH = 'pokemon.db'
MODEL_PATH = 'model.pth'
SCALER_PATH = 'scaler.pkl'
TYPES_LIST = [
    "normal", "fighting", "flying", "poison", "ground", "bug", "ghost", "steel",
    "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "fairy",
    "dark", "rock"
]

def get_data_from_db():
    """Загружает данные из SQLite и превращает в DataFrame"""
    if not os.path.exists(DB_PATH):
        return None
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM pokemon", conn)
    conn.close()
    
    if df.empty:
        return None
        
    return df

def preprocess_data(df):
    """
    Преобразует данные в числовой вектор:
    1. Числовые признаки (stats, height, weight) -> нормализация 0-1
    2. Типы -> One-Hot Encoding
    """
    num_cols = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed', 'height', 'weight']
    
    # 1. Числовые данные
    num_data = df[num_cols].values.astype(float)
    
    # Нормализация MinMax (сохраняем мин/макс для будущего использования)
    min_vals = num_data.min(axis=0)
    max_vals = num_data.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Избегаем деления на 0
    
    num_normalized = (num_data - min_vals) / range_vals
    
    # 2. Типы (One-Hot)
    types_matrix = np.zeros((len(df), len(TYPES_LIST)))
    for i, types_str in enumerate(df['types']):
        if pd.isna(types_str):
            continue
        p_types = [t.strip() for t in types_str.split(',')]
        for t in p_types:
            if t in TYPES_LIST:
                types_matrix[i, TYPES_LIST.index(t)] = 1
                
    # Объединяем
    X = np.hstack([num_normalized, types_matrix])
    
    scaler_data = {
        'min': min_vals,
        'max': max_vals,
        'range': range_vals,
        'columns': num_cols
    }
    
    return X, scaler_data, df['name'].tolist(), df['id'].tolist()

def train_model():
    """Основная функция обучения"""
    print("🔄 Загрузка данных из БД...")
    df = get_data_from_db()
    
    if df is None or len(df) < 10:
        print("⚠️ Недостаточно данных для обучения (нужно минимум 10 покемонов)")
        return False
    
    print("⚙️ Предобработка данных...")
    X, scaler_data, names, ids = preprocess_data(df)
    
    # Сохраняем скалер
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler_data, f)
    
    # Подготовка тензоров
    X_tensor = torch.FloatTensor(X)
    dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Инициализация модели
    input_dim = X.shape[1]
    model = PokemonAutoencoder(input_dim=input_dim, latent_dim=16)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🧠 Начало обучения (Вход: {input_dim}, Сжатие: 16)...")
    model.train()
    epochs = 100
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            _, decoded = model(batch_x)
            loss = criterion(decoded, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 20 == 0:
            print(f"Эпоха {epoch+1}/{epochs}, Потеря: {total_loss/len(loader):.4f}")
            
    # Сохранение модели
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Модель обучена и сохранена в {MODEL_PATH}")
    return True

def load_model_and_scaler():
    """Загружает готовую модель и скалер"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
        
    model = PokemonAutoencoder(input_dim=26, latent_dim=16) # 8 stats + 18 types
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler