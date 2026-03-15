import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-shrimp-key-change-in-prod!')
    SERVER_HOST = os.getenv('SERVER_HOST', '127.0.0.1')
    SERVER_PORT = int(os.getenv('SERVER_PORT', '5000'))
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/users.db')
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    TRAIN_DATA_PATH = os.getenv('TRAIN_DATA_PATH', 'data/train.npz')
    VALID_DATA_PATH = os.getenv('VALID_DATA_PATH', 'data/valid.npz')
    TEST_DATA_PATH = os.getenv('TEST_DATA_PATH', 'data/test.npz')
    TEST_DATA_PASSWORD = os.getenv('TEST_DATA_PASSWORD', '')
    MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', 'alien_model_final.pkl')
    TEMP_DIR = os.getenv('TEMP_DIR', 'temp')
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    TRAIN_CONFIG = {
        'epochs': int(os.getenv('TRAIN_EPOCHS', '50')),
        'batch_size': int(os.getenv('TRAIN_BATCH_SIZE', '32')),
        'learning_rate': float(os.getenv('TRAIN_LR', '0.001')),
        'validation_split': 0.25,
        'early_stopping_patience': 5,
        'save_best_only': True
    }
    
    USE_DATASPHERE_INFERENCE = os.getenv('USE_DATASPHERE', 'false').lower() == 'true'
    DATASPHERE_NODE_ID = os.getenv('DATASPHERE_NODE_ID', '')
    DATASPHERE_FOLDER_ID = os.getenv('DATASPHERE_FOLDER_ID', '')
    DATASPHERE_IAM_TOKEN = os.getenv('DATASPHERE_IAM_TOKEN', '')
    
    DATASPHERE_TIMEOUT = int(os.getenv('DATASPHERE_TIMEOUT', '60'))
    
    TOKEN_EXPIRY_SECONDS = int(os.getenv('TOKEN_EXPIRY', '86400'))
    
    MIN_PASSWORD_LENGTH = int(os.getenv('MIN_PASSWORD_LENGTH', '8'))
    
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)