import librosa
import numpy as np
import torch

class FeatureExtractor:
    def __init__(self):
        self.sr = 22050
        
    def extract_features(self, audio):
        """오디오에서 특징 추출"""
        try:
            # MFCC 특성 (64개의 특성으로 조정)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=64)
            
            # 시간 차원을 64로 조정
            target_length = 64
            if mfcc.shape[1] > target_length:
                mfcc = mfcc[:, :target_length]
            elif mfcc.shape[1] < target_length:
                pad_width = ((0, 0), (0, target_length - mfcc.shape[1]))
                mfcc = np.pad(mfcc, pad_width, mode='constant')
            
            # 차원 순서 변경: (시간, 특성) -> (특성, 시간)
            features = mfcc.T
            
            return features
            
        except Exception as e:
            print(f"특징 추출 에러: {str(e)}")
            return np.zeros((64, 64))  # 에러 시 기본값 반환 