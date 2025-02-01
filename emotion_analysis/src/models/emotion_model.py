import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_emotions=8):
        super(EmotionRecognitionModel, self).__init__()
        
        # CNN 특성 추출기
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # BiLSTM 레이어
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        
        # 어텐션 레이어
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        # 최종 분류기
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions)
        )
        
        # 감정 매핑
        self.emotion_map = {
            0: "neutral",
            1: "happy",
            2: "sad",
            3: "angry",
            4: "fearful",
            5: "disgust",
            6: "surprised",
            7: "calm"
        }
        
        # device 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def extract_features(self, audio):
        """고급 오디오 특성 추출"""
        try:
            # MFCC 특성
            mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # 스펙트럴 특성
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=22050)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=22050)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=22050)
            
            # 크로마 특성
            chroma = librosa.feature.chroma_stft(y=audio, sr=22050)
            
            # 멜 스펙트로그램
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=22050)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 모든 특성 결합
            features = np.concatenate([
                mfcc.flatten(),
                mfcc_delta.flatten(),
                mfcc_delta2.flatten(),
                spectral_centroids.flatten(),
                spectral_rolloff.flatten(),
                spectral_contrast.flatten(),
                chroma.flatten(),
                mel_spec_db.flatten()
            ])
            
            return torch.FloatTensor(features)
            
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return torch.zeros(2816)
    
    def forward(self, audio_features):
        try:
            # 배치 차원 추가
            if audio_features.dim() == 1:
                audio_features = audio_features.unsqueeze(0)
            
            # 채널 차원 추가
            x = audio_features.unsqueeze(1)
            
            # CNN 특성 추출
            x = self.conv_layers(x)
            
            # 차원 변환 (Transformer 입력용)
            x = x.transpose(1, 2)
            
            # Transformer 처리
            x = self.transformer(x)
            
            # BiLSTM 처리
            x, _ = self.lstm(x)
            
            # 어텐션 처리
            x, _ = self.attention(x, x, x)
            
            # 전역 평균 풀링
            x = torch.mean(x, dim=1)
            
            # 분류
            logits = self.classifier(x)
            
            return logits
            
        except Exception as e:
            print(f"Forward pass error: {str(e)}")
            return torch.zeros((1, len(self.emotion_map))).to(self.device)
    
    def predict_emotion(self, audio_features, text=None):
        self.eval()
        with torch.no_grad():
            # 특성 추출
            if not isinstance(audio_features, torch.Tensor):
                audio_features = self.extract_features(audio_features)
            
            audio_features = audio_features.to(self.device)
            
            # 예측
            logits = self(audio_features)
            probabilities = F.softmax(logits, dim=1)
            
            # 상위 2개 감정 확인
            top_probs, top_indices = torch.topk(probabilities, 2, dim=1)
            
            # 첫 번째 감정의 확률이 충분히 높은지 확인
            if top_probs[0][0] > 0.4:  # 임계값
                predicted_emotion = self.emotion_map[top_indices[0][0].item()]
                confidence = top_probs[0][0].item()
            else:
                # 두 번째로 높은 감정 고려
                predicted_emotion = self.emotion_map[top_indices[0][1].item()]
                confidence = top_probs[0][1].item()
            
            return {
                'emotion': predicted_emotion,
                'probabilities': {
                    self.emotion_map[i]: p.item()
                    for i, p in enumerate(probabilities[0])
                },
                'confidence': confidence
            }