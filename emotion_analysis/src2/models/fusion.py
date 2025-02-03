import torch
import torch.nn as nn

class MultiModalFusion(nn.Module):
    def __init__(self, text_dim=768, audio_dim=512, video_dim=512, num_emotions=7):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + audio_dim + video_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_emotions)
        )
        
    def forward(self, text_features, audio_features, video_features):
        combined = torch.cat([text_features, audio_features, video_features], dim=1)
        return self.fusion(combined) 