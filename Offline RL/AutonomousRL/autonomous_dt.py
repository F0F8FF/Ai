import torch
import torch.nn as nn
import torch.nn.functional as F

class AutonomousDecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim=5,
        action_dim=2,
        max_length=50,
        hidden_size=128,
        n_layer=3,
        n_head=4,
        dropout=0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 기본 인코더
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.action_encoder = nn.Linear(action_dim, hidden_size)
        self.return_encoder = nn.Linear(1, hidden_size)
        
        # 위치 인코딩
        self.position_encoding = nn.Embedding(max_length, hidden_size)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=4*hidden_size,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer
        )
        
        # 출력 헤드
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # 인코딩
        state_embeddings = self.state_encoder(states)    # [batch_size, seq_length, hidden_size]
        action_embeddings = self.action_encoder(actions) # [batch_size, seq_length, hidden_size]
        returns_embeddings = self.return_encoder(returns_to_go.unsqueeze(-1)) # [batch_size, seq_length, hidden_size]
        
        # 위치 인코딩 추가
        position_embeddings = self.position_encoding(timesteps) # [batch_size, seq_length, hidden_size]
        
        # 시퀀스 결합 (concatenate along seq_length dimension)
        sequence = torch.cat(
            [returns_embeddings, state_embeddings, action_embeddings],
            dim=1
        )  # [batch_size, 3*seq_length, hidden_size]
        
        # Transformer 통과
        sequence = sequence.transpose(0, 1)  # [3*seq_length, batch_size, hidden_size]
        sequence = self.transformer(sequence)
        sequence = sequence.transpose(0, 1)  # [batch_size, 3*seq_length, hidden_size]
        
        # 행동 예측 (state embeddings 다음의 위치에서)
        action_preds = self.action_head(sequence[:, seq_length:2*seq_length, :])
        
        # 크기 맞추기
        if action_preds.shape[1] != actions.shape[1]:
            action_preds = action_preds[:, :actions.shape[1], :]
        
        return action_preds