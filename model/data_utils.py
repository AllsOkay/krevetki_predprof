import numpy as np
import os
from typing import Dict, Optional, Union, List
from scipy.io import wavfile
import io


def load_npz_file(filepath: str, password: Optional[str] = None) -> Dict[str, np.ndarray]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    if password:
        pass
    data = np.load(filepath, allow_pickle=True)
    
    result = {key: data[key] for key in data.files}
    data.close()
    
    return result


def load_dataset(filepath: str) -> Dict[str, np.ndarray]:
    raw_data = load_npz_file(filepath)
    
    result = {}
    
    for key in ['x', 'train_x', 'valid_x', 'test_x', 'data', 'signals']:
        if key in raw_data:
            result['x'] = raw_data[key]
            break
    
    for key in ['y', 'train_y', 'valid_y', 'test_y', 'labels', 'targets']:
        if key in raw_data:
            result['y'] = raw_data[key]
            break
    
    if 'x' not in result:
        raise ValueError(f"No signal data found in {filepath}. Available keys: {list(raw_data.keys())}")
    
    return result


def restore_class_labels(labels: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if labels is None:
        return None
    
    if np.issubdtype(labels.dtype, np.integer):
        return labels.astype(int)
    
    if labels.dtype == object or labels.dtype.kind in ('U', 'S', 'a'):
        unique_labels = np.unique(labels)
        
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        
        restored = np.array([label_to_int[label] for label in labels], dtype=int)
        
        return restored
    
    if np.issubdtype(labels.dtype, np.floating):
        return np.round(labels).astype(int)
    
    return np.array(labels, dtype=int)


def preprocess_wav_signal(wav_data: Union[bytes, np.ndarray], 
                          target_length: int = 16000) -> np.ndarray:
    import scipy.io.wavfile as wavfile
    from scipy.signal import spectrogram
    if isinstance(wav_data, bytes):
        with io.BytesIO(wav_data) as buffer:
            sample_rate, audio = wavfile.read(buffer)
    else:
        audio = wav_data
        sample_rate = 16000
    
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        padding = np.zeros(target_length - len(audio), dtype=audio.dtype)
        audio = np.concatenate([audio, padding])
    
    f, t, Sxx = spectrogram(
        audio, 
        fs=sample_rate,
        nperseg=256,
        noverlap=128,
        window='hann'
    )
    
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min() + 1e-8)
    
    return Sxx_norm.astype(np.float32)


def batch_preprocess(signals: List[Union[bytes, np.ndarray]], 
                     target_length: int = 16000) -> np.ndarray:
    processed = [preprocess_wav_signal(sig, target_length) for sig in signals]
    return np.stack(processed, axis=0)


def split_train_val(x: np.ndarray, y: np.ndarray, 
                    val_ratio: float = 0.25, 
                    random_state: int = 42) -> Dict[str, np.ndarray]:
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