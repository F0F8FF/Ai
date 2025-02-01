import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from typing import Dict, Any
from src.preprocessing.preprocessor import AudioPreprocessor

class EmotionDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, preprocessor: AudioPreprocessor):
        self.data = data_df
        self.preprocessor = preprocessor
        
        # 감정 레이블 인코딩
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(sorted(data_df['emotion'].unique()))}
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.data.iloc[idx]
        
        # 오디오 특성 추출
        features = self.preprocessor.extract_features(row['path'])
        
        # 레이블 인코딩
        label = torch.tensor(self.emotion_to_idx[row['emotion']])
        
        return {
            'features': features,
            'label': label,
            'emotion': row['emotion']
        } 