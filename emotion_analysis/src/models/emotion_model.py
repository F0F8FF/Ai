import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class DeepSeekEmotionModel(nn.Module):
    def __init__(self, num_emotions=8):
        super(DeepSeekEmotionModel, self).__init__()
        
        # DeepSeek 모델
        model_name = "deepseek-ai/deepseek-moe-16b-base"
        self.deepseek = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 파라미터 고정
        self.deepseek.eval()
        for param in self.deepseek.parameters():
            param.requires_grad = False
            
        # 간단한 CNN
        self.audio_conv = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 두 번째 블록
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 세 번째 블록
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        ).float()
        
        # 텍스트 프로젝션
        self.text_proj = nn.Sequential(
            nn.Linear(2048, 256),  # DeepSeek hidden_size = 2048
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        ).float()
        
        # 오디오 프로젝션
        self.audio_proj = nn.Sequential(
            nn.Linear(128 * 2 * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        ).float()
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions)
        ).float()
        
    def get_text_features(self, text_input):
        with torch.no_grad():
            outputs = self.deepseek(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask']
            )
            text_features = outputs.last_hidden_state[:, 0, :].float()
            return self.text_proj(text_features)
    
    def forward(self, audio_features, cached_text_features=None, text_description=None):
        # 오디오 처리
        x = audio_features.float()
        x = x.unsqueeze(1) if len(x.shape) == 3 else x
        
        # CNN
        x = self.audio_conv(x)
        audio_features = x.view(x.size(0), -1)
        audio_features = self.audio_proj(audio_features)
        
        # 텍스트 특성
        if cached_text_features is not None:
            text_features = cached_text_features.float()
        elif text_description is not None:
            text_features = self.get_text_features(text_description)
        else:
            raise ValueError("Either cached_text_features or text_description must be provided")
        
        # 배치 크기 맞추기
        if text_features.size(0) != audio_features.size(0):
            text_features = text_features.repeat(audio_features.size(0), 1)
        
        # 특성 결합 및 분류
        combined = torch.cat([audio_features, text_features], dim=1)
        return self.classifier(combined)