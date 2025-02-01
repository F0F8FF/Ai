import os
import soundfile as sf  # librosa 대신 soundfile 사용
import numpy as np
import torch
from torch.utils.data import Dataset

class AudioProcessor:
    def __init__(self, data_path, sample_rate=22050, duration=3.0):
        """오디오 처리를 위한 클래스
        
        Args:
            data_path (str): 오디오 파일이 있는 디렉토리 경로
            sample_rate (int): 샘플링 레이트 (기본값: 22050)
            duration (float): 오디오 길이 (초) (기본값: 3.0)
        """
        self.data_path = data_path
        self.sample_rate = int(sample_rate)
        self.duration = float(duration)
        self.target_length = int(self.sample_rate * self.duration)
        
        # 데이터 파일 목록 가져오기
        self.file_paths = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        """데이터 파일 목록과 레이블 로드"""
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    # RAVDESS 파일명 형식: 03-01-01-01-01-01-01.wav
                    # 세 번째 숫자가 감정 레이블 (01 = neutral, 02 = calm, 03 = happy, etc.)
                    emotion_label = int(file.split('-')[2]) - 1  # 0-based indexing
                    self.file_paths.append(file_path)
                    self.labels.append(emotion_label)
    
    def load_audio(self, file_path):
        """오디오 파일 로드 및 전처리
        
        Args:
            file_path (str): 오디오 파일 경로
            
        Returns:
            np.ndarray: 전처리된 오디오 데이터
        """
        # 오디오 로드
        audio, sr = sf.read(file_path)
        
        # 스테레오를 모노로 변환 (필요한 경우)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 샘플링 레이트 변환 (필요한 경우)
        if sr != self.sample_rate:
            # resample using numpy (간단한 방법)
            duration = len(audio) / sr
            new_length = int(duration * self.sample_rate)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)
        
        # 길이 조정
        if len(audio) > self.target_length:
            audio = audio[:self.target_length]
        elif len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        
        return audio

class EmotionDataset(Dataset):
    def __init__(self, audio_processor, feature_extractor):
        self.audio_processor = audio_processor
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.audio_processor.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.audio_processor.file_paths[idx]
        label = self.audio_processor.labels[idx]
        
        # 오디오 로드 및 특성 추출
        audio = self.audio_processor.load_audio(file_path)
        features = self.feature_extractor.extract_features(audio)
        
        return torch.FloatTensor(features), torch.tensor(label, dtype=torch.long)