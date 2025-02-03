import librosa
import numpy as np
import torch
from typing import Tuple, Dict

class AudioPreprocessor:
    def __init__(self, sample_rate: int = 22050, duration: int = 3):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = sample_rate * duration
        
    def extract_features(self, audio_path: str) -> Dict[str, torch.Tensor]:
        # 오디오 로드
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        # 길이 표준화
        if len(audio) > self.target_length:
            audio = audio[:self.target_length]
        else:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
            
        # 특성 추출
        features = {
            'mfcc': self._extract_mfcc(audio),
            'mel_spectrogram': self._extract_mel_spectrogram(audio),
            'chroma': self._extract_chroma(audio)
        }
        
        return features
    
    def _extract_mfcc(self, audio: np.ndarray) -> torch.Tensor:
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=40)
        return torch.FloatTensor(mfcc)
    
    def _extract_mel_spectrogram(self, audio: np.ndarray) -> torch.Tensor:
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return torch.FloatTensor(mel_spec_db)
    
    def _extract_chroma(self, audio: np.ndarray) -> torch.Tensor:
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        return torch.FloatTensor(chroma) 