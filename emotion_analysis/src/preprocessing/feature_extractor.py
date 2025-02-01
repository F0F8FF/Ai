import torch
import torchaudio
import torchaudio.transforms as T

class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=80):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # 특성 추출을 위한 변환기들
        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels
        )
        
        self.amplitude_to_db = T.AmplitudeToDB()
    
    def extract_features(self, audio):
        """오디오에서 특성 추출
        
        Args:
            audio (np.ndarray): 오디오 신호
            
        Returns:
            torch.Tensor: 추출된 특성 [2816]
        """
        try:
            # numpy array를 torch tensor로 변환
            if not isinstance(audio, torch.Tensor):
                audio = torch.FloatTensor(audio)
            
            # 차원 추가
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Mel spectrogram 계산
            mel_spec = self.melspec(audio)
            mel_spec_db = self.amplitude_to_db(mel_spec)
            
            # 정규화
            mel_spec_norm = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
            
            # 크기 조정 (2816 차원으로)
            features = torch.nn.functional.adaptive_avg_pool2d(
                mel_spec_norm, 
                (44, 64)
            ).flatten()
            
            # 정확히 2816 차원으로 맞추기
            if len(features) > 2816:
                features = features[:2816]
            elif len(features) < 2816:
                features = torch.nn.functional.pad(features, (0, 2816 - len(features)))
            
            print(f"Extracted features shape: {features.shape}")
            return features
            
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            raise 