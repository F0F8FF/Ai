import torch
import torchaudio
import numpy as np

class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=80, n_mfcc=40):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
        # Mel Spectrogram 변환기 수정
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,  # 수정됨
            hop_length=256,  # 수정됨
            f_min=20,
            f_max=8000
        )
        
        # MFCC 변환기
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': 1024,
                'hop_length': 256,
                'n_mels': n_mels,
                'f_min': 20,
                'f_max': 8000
            }
        )
    
    def extract_features(self, audio):
        # 입력이 numpy array인 경우 torch tensor로 변환
        if isinstance(audio, np.ndarray):
            audio = torch.FloatTensor(audio)
        
        # 차원 추가 (배치 차원)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # 특성 추출
        mel_spec = self.mel_spec(audio)
        mfcc = self.mfcc_transform(audio)
        
        # log scale 적용 (epsilon 값 증가)
        mel_spec = torch.log(mel_spec + 1e-5)
        
        return {
            'mel_spectrogram': mel_spec,
            'mfcc': mfcc
        } 