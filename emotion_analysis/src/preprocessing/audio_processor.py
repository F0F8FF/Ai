import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
import soundfile as sf
import os

class AudioProcessor:
    def __init__(self, sample_rate=16000, duration=3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
    def process_audio(self, audio_path):
        # 오디오 로드
        audio, sr = sf.read(audio_path)
        
        # 스테레오를 모노로 변환 (필요한 경우)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # 샘플레이트 변환 (필요한 경우)
        if sr != self.sample_rate:
            audio = self._resample(audio, sr)
        
        # 길이 표준화
        audio = self._pad_or_trim(audio)
        
        # 정규화
        audio = self._normalize(audio)
        
        return audio
    
    def _resample(self, audio, orig_sr):
        return torchaudio.transforms.Resample(orig_sr, self.sample_rate)(torch.FloatTensor(audio))
    
    def _pad_or_trim(self, audio):
        if len(audio) > self.target_length:
            audio = audio[:self.target_length]
        else:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        return audio
    
    def _normalize(self, audio):
        return (audio - audio.mean()) / (audio.std() + 1e-8)

class EmotionDataset(Dataset):
    def __init__(self, data_dir, processor, feature_extractor):
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.samples = []
        self.emotions = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }
        self._load_data(data_dir)
        
        # 감정 레이블을 숫자로 인코딩
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(sorted(self.emotions.values()))}
    
    def _load_data(self, data_dir):
        for actor_dir in os.listdir(data_dir):
            if actor_dir.startswith('Actor_'):
                actor_path = os.path.join(data_dir, actor_dir)
                for file_name in os.listdir(actor_path):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(actor_path, file_name)
                        emotion_code = file_name.split('-')[2]
                        emotion = self.emotions[emotion_code]
                        self.samples.append((file_path, emotion))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, emotion = self.samples[idx]
        
        # 오디오 처리
        audio = self.processor.process_audio(audio_path)
        
        # 특성 추출
        features = self.feature_extractor.extract_features(audio)
        
        return {
            'audio': torch.FloatTensor(audio),
            'mel_spectrogram': features['mel_spectrogram'],
            'mfcc': features['mfcc'],
            'emotion': self.emotion_to_idx[emotion],
            'emotion_label': emotion
        }