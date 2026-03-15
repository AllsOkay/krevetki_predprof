# backend/model/data_utils.py
# Утилиты для загрузки и предобработки данных
# Работа с .npz архивами и восстановление повреждённых меток классов

import numpy as np
import os
from typing import Dict, Optional, Union, List
from scipy.io import wavfile
import io


def load_npz_file(filepath: str, password: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Загружает данные из .npz архива.
    
    :param filepath: Путь к файлу .npz
    :param password: Пароль для зашифрованных архивов (опционально)
    :return: Словарь с массивами из архива
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    # Для зашифрованных архивов нужна дополнительная обработка
    # numpy.load не поддерживает пароли напрямую, поэтому:
    if password:
        # В реальном проекте использовать proper decryption
        # Здесь упрощённая заглушка
        pass
    
    # Загружаем архив с allow_pickle для совместимости
    data = np.load(filepath, allow_pickle=True)
    
    # Конвертируем в обычный dict для удобства
    result = {key: data[key] for key in data.files}
    data.close()
    
    return result


def load_dataset(filepath: str) -> Dict[str, np.ndarray]:
    """
    Загружает набор данных для обучения/валидации/теста.
    
    Ожидает структуру:
    - 'x' или 'train_x'/'valid_x'/'test_x': массив сигналов
    - 'y' или 'train_y'/'valid_y'/'test_y': массив меток
    
    :param filepath: Путь к .npz файлу
    :return: Словарь с ключами 'x' и 'y'
    """
    raw_data = load_npz_file(filepath)
    
    # Нормализуем имена ключей к единому формату
    result = {}
    
    # Ищем массив с признаками (сигналами)
    for key in ['x', 'train_x', 'valid_x', 'test_x', 'data', 'signals']:
        if key in raw_data:  # ✅ ИСПРАВЛЕНО: было "raw_"
            result['x'] = raw_data[key]
            break
    
    # Ищем массив с метками
    for key in ['y', 'train_y', 'valid_y', 'test_y', 'labels', 'targets']:
        if key in raw_data:  # ✅ ИСПРАВЛЕНО: было "raw_"
            result['y'] = raw_data[key]
            break
    
    if 'x' not in result:
        raise ValueError(f"No signal data found in {filepath}. Available keys: {list(raw_data.keys())}")
    
    return result


def restore_class_labels(labels: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Восстанавливает порядковые целочисленные метки классов из повреждённых строк.
    
    Проблема: практиканты сохранили классы как строки вместо чисел.
    Решение: сопоставляем уникальные строковые значения с числами 0, 1, 2...
    
    :param labels: Массив меток (строки или числа)
    :return: Массив с целочисленными метками от 0 до n_classes-1
    """
    if labels is None:
        return None
    
    # Если метки уже числовые - возвращаем как есть
    if np.issubdtype(labels.dtype, np.integer):
        return labels.astype(int)
    
    # Если метки строковые - восстанавливаем соответствие
    # ✅ ИСПРАВЛЕНО: совместимость с разными версиями NumPy
    if labels.dtype == object or labels.dtype.kind in ('U', 'S', 'a'):
        # Находим уникальные значения и сортируем для детерминизма
        unique_labels = np.unique(labels)
        
        # Создаём маппинг: строка -> число
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Преобразуем массив
        restored = np.array([label_to_int[label] for label in labels], dtype=int)
        
        return restored
    
    # Для float меток (если были нормализованы) - округляем
    if np.issubdtype(labels.dtype, np.floating):
        return np.round(labels).astype(int)
    
    # Если тип неизвестен - пробуем конвертировать
    return np.array(labels, dtype=int)


def preprocess_wav_signal(wav_data: Union[bytes, np.ndarray], 
                          target_length: int = 16000) -> np.ndarray:
    """
    Предобрабатывает WAV-сигнал для подачи в нейросеть.
    
    Шаги предобработки:
    1. Чтение WAV-файла (если переданы байты)
    2. Нормализация амплитуды к [-1, 1]
    3. Приведение к фиксированной длине (обрезка/паддинг)
    4. Вычисление спектрограммы (опционально, для улучшения качества)
    
    :param wav_data: WAV-данные (байты) или уже загруженный массив
    :param target_length: Целевая длина сигнала в сэмплах
    :return: Нормализованный numpy-массив готовый для модели
    """
    import scipy.io.wavfile as wavfile
    from scipy.signal import spectrogram
    
    # Шаг 1: Чтение WAV если нужно
    # ✅ ИСПРАВЛЕНО: параметр назывался wav_ в сигнатуре, но wav_data в теле
    if isinstance(wav_data, bytes):
        with io.BytesIO(wav_data) as buffer:
            sample_rate, audio = wavfile.read(buffer)
    else:
        # Предполагаем что это уже массив аудио-сэмплов
        audio = wav_data
        sample_rate = 16000  # дефолтное значение
    
    # Шаг 2: Конвертация в float и нормализация
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    # Если уже float - предполагаем что уже нормализован
    
    # Шаг 3: Приведение к фиксированной длине
    if len(audio) > target_length:
        # Обрезаем с начала (можно изменить стратегию)
        audio = audio[:target_length]
    elif len(audio) < target_length:
        # Дополняем нулями в конце
        padding = np.zeros(target_length - len(audio), dtype=audio.dtype)
        audio = np.concatenate([audio, padding])
    
    # Шаг 4: Вычисление спектрограммы (преобразование время->частота)
    # Это помогает модели лучше различать паттерны в сигналах
    f, t, Sxx = spectrogram(
        audio, 
        fs=sample_rate,
        nperseg=256,      # Размер окна Фурье
        noverlap=128,     # Перекрытие окон для плавности
        window='hann'     # Окно Ханна для уменьшения артефактов
    )
    
    # Логарифмическое сжатие для улучшения контраста спектрограммы
    Sxx_db = 10 * np.log10(Sxx + 1e-10)  # +epsilon для избежания log(0)
    
    # Нормализация к диапазону [0, 1] для стабильности обучения
    Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-8)
    
    return Sxx_norm.astype(np.float32)


def batch_preprocess(signals: List[Union[bytes, np.ndarray]], 
                     target_length: int = 16000) -> np.ndarray:
    """
    Пакетная предобработка нескольких сигналов.
    
    :param signals: Список сигналов для обработки
    :param target_length: Целевая длина каждого сигнала
    :return: Массив формы (batch_size, freq_bins, time_frames)
    """
    processed = [preprocess_wav_signal(sig, target_length) for sig in signals]
    # Stack создаёт батч из отдельных спектрограмм
    return np.stack(processed, axis=0)


def split_train_val(x: np.ndarray, y: np.ndarray, 
                    val_ratio: float = 0.25, 
                    random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Разделяет данные на обучающую и валидационную выборки.
    
    :param x: Массив признаков (сигналов)
    :param y: Массив меток классов
    :param val_ratio: Доля данных для валидации (по умолчанию 25%)
    :param random_state: Seed для воспроизводимости
    :return: Словарь с 'train_x', 'train_y', 'val_x', 'val_y'
    """
    np.random.seed(random_state)
    
    n_samples = len(x)
    indices = np.random.permutation(n_samples)
    
    val_size = int(n_samples * val_ratio)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    return {
        'train_x': x[train_indices],
        'train_y': y[train_indices],
        'val_x': x[val_indices],
        'val_y': y[val_indices]
    }