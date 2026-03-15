import sys
import os
import logging

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    from backend.app import app
    
    logger.info(f"Запуск Alien Signal Classifier v1.0")
    logger.info(f"Режим: {'DEBUG' if Config.DEBUG else 'PRODUCTION'}")
    logger.info(f"Сервер: http://{Config.SERVER_HOST}:{Config.SERVER_PORT}")
    
    app.run(
        host=Config.SERVER_HOST,
        port=Config.SERVER_PORT,
        debug=Config.DEBUG,
        use_reloader=Config.DEBUG
    )

if __name__ == '__main__':
    main()