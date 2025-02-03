import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x):
        # Self-attention
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        
        # Feed forward
        fed_forward = self.feed_forward(x)
        x = self.norm2(x + fed_forward)
        return x

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        
        # CNN 특징 추출기 (입력 채널 64로 수정)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 텍스트 특징 추출기
        self.text_encoder = nn.Sequential(
            nn.Linear(768, 512),  # deepseek 모델의 출력 차원을 512로 변환
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Attention 레이어
        self.attention = AttentionBlock(embed_dim=512)
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
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
    
    def get_text_features(self, text_encoded):
        """텍스트 특징 추출"""
        # text_encoded의 hidden_states나 last_hidden_state를 사용
        if hasattr(text_encoded, 'last_hidden_state'):
            text_features = text_encoded.last_hidden_state.mean(dim=1)
        else:
            text_features = text_encoded['last_hidden_state'].mean(dim=1)
        
        # 텍스트 특징 변환
        text_features = self.text_encoder(text_features)
        return text_features
    
    def forward(self, audio, text_features=None):
        # 오디오 특징 추출 (B, 64, T)
        x = self.conv_layers(audio)
        
        # Attention 적용
        x = x.transpose(1, 2)  # (B, T, 512)
        x = self.attention(x)
        x = torch.mean(x, dim=1)  # Global average pooling
        
        # 분류
        x = self.classifier(x)
        return x
    
    def predict_emotion(self, audio_features, text=None):
        self.eval()
        with torch.no_grad():
            # 특성 추출
            if not isinstance(audio_features, torch.Tensor):
                audio_features = self.extract_features(audio_features)
            
            audio_features = audio_features.to(self.device)
            
            # 예측
            logits = self(audio_features, text_features)
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