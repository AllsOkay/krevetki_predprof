import librosa
import numpy as np
from pathlib import Path

class AudioPreprocessor:
    def __init__(self, sr=16000, n_mfcc=40, hop_length=512, n_fft=2048):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
    
    def load_audio(self, file_path, duration=None):
        audio, sr = librosa.load(file_path, sr=self.sr, duration=duration)
        return audio
    
    def extract_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sr, 
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        return mfcc.T  # Транспонируем: (time_steps, features)
    
    def extract_spectrogram(self, audio):
        spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr, 
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
        return spec_db.T  # (time_steps, features)
    
    def normalize(self, features):
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        return (features - mean) / std
    
    def pad_sequence(self, features, max_length):
        if len(features) >= max_length:
            return features[:max_length]
        else:
            pad_width = ((0, max_length - len(features)), (0, 0))
            return np.pad(features, pad_width, mode='constant')
    
    def process(self, file_path, max_length=1000, feature_type='mfcc'):
        audio = self.load_audio(file_path)
        
        if feature_type == 'mfcc':
            features = self.extract_mfcc(audio)
        elif feature_type == 'spectrogram':
            features = self.extract_spectrogram(audio)
        else:
            raise ValueError("feature_type должен быть 'mfcc' или 'spectrogram'")
        
        features = self.normalize(features)
        features = self.pad_sequence(features, max_length)
        
        return features