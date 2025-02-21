import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 시드 설정
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# MIMIC-III 데이터 로딩 및 전처리
class SepsisDataset:
    def __init__(self, data_path):
        # 데이터 로드
        self.data = pd.read_csv(data_path)
        
        # 상태 변수 정의
        self.state_cols = [
            'heart_rate', 'sbp', 'dbp', 'temp', 'resp_rate',
            'wbc', 'creatinine', 'platelet', 'spo2', 'lactate'
        ]
        
        # 행동 변수 정의
        self.action_cols = [
            'antibiotics', 'fluid', 'vasopressor'
        ]
        
        # 스케일러 초기화
        self.state_scaler = StandardScaler()
        self.action_scaler = StandardScaler()
        
        # 데이터 정규화
        self.states = self.state_scaler.fit_transform(self.data[self.state_cols])
        self.actions = self.action_scaler.fit_transform(self.data[self.action_cols])
        
        # 보상 계산
        self.rewards = self._compute_rewards()
        
    def _compute_rewards(self):
        # SOFA 점수 계산 및 보상 정의
        sofa_scores = self._compute_sofa_scores()
        rewards = -sofa_scores  # SOFA 점수가 낮을수록 좋음
        return rewards
    
    def _compute_sofa_scores(self):
        # 간단한 SOFA 점수 계산
        sofa_scores = np.zeros(len(self.data))
        
        # 각 항목별 점수 계산
        sofa_scores += (self.data['platelet'] < 150) * 1
        sofa_scores += (self.data['creatinine'] > 1.2) * 1
        sofa_scores += (self.data['sbp'] < 100) * 1
        
        return sofa_scores

# 데이터셋 클래스
class OfflineRLDataset(Dataset):
    def __init__(self, states, actions, rewards, next_states, dones):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        self.rewards = torch.FloatTensor(rewards).unsqueeze(1)
        self.next_states = torch.FloatTensor(next_states)
        self.dones = torch.FloatTensor(dones).unsqueeze(1)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.net(state)

# CQL Algorithm
class CQL:
    def __init__(self, state_dim, action_dim, device='cuda'):
        self.device = device
        
        # 네트워크 초기화
        self.q_net1 = QNetwork(state_dim, action_dim).to(device)
        self.q_net2 = QNetwork(state_dim, action_dim).to(device)
        self.target_q_net1 = QNetwork(state_dim, action_dim).to(device)
        self.target_q_net2 = QNetwork(state_dim, action_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        
        # 타겟 네트워크 초기화
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        
        # 옵티마이저
        self.q_optimizer1 = torch.optim.Adam(self.q_net1.parameters(), lr=3e-4)
        self.q_optimizer2 = torch.optim.Adam(self.q_net2.parameters(), lr=3e-4)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)
        
        self.gamma = 0.99
        self.tau = 0.005
        self.cql_alpha = 1.0
        
    def train_step(self, batch):
        states, actions, rewards, next_states, dones = [b.to(self.device) for b in batch]
        
        # Q-value 계산
        current_q1 = self.q_net1(states, actions)
        current_q2 = self.q_net2(states, actions)
        
        # 타겟 계산
        with torch.no_grad():
            next_actions = self.policy_net(next_states)
            target_q1 = self.target_q_net1(next_states, next_actions)
            target_q2 = self.target_q_net2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards + (1 - dones) * self.gamma * target_q
        
        # CQL 손실
        random_actions = torch.FloatTensor(actions.shape).uniform_(-1, 1).to(self.device)
        random_q1 = self.q_net1(states, random_actions)
        random_q2 = self.q_net2(states, random_actions)
        
        cql_loss1 = torch.logsumexp(random_q1, dim=0) - current_q1.mean()
        cql_loss2 = torch.logsumexp(random_q2, dim=0) - current_q2.mean()
        
        # Q 손실
        q_loss1 = F.mse_loss(current_q1, target_value) + self.cql_alpha * cql_loss1
        q_loss2 = F.mse_loss(current_q2, target_value) + self.cql_alpha * cql_loss2
        
        # Q 네트워크 업데이트
        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        self.q_optimizer1.step()
        
        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        self.q_optimizer2.step()
        
        # 정책 업데이트
        policy_actions = self.policy_net(states)
        q_value = torch.min(
            self.q_net1(states, policy_actions),
            self.q_net2(states, policy_actions)
        )
        policy_loss = -q_value.mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # 타겟 네트워크 소프트 업데이트
        self._soft_update(self.q_net1, self.target_q_net1)
        self._soft_update(self.q_net2, self.target_q_net2)
        
        return {
            'q_loss1': q_loss1.item(),
            'q_loss2': q_loss2.item(),
            'policy_loss': policy_loss.item()
        }
    
    def _soft_update(self, local_net, target_net):
        for local_param, target_param in zip(local_net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'q1_state_dict': self.q_net1.state_dict(),
            'q2_state_dict': self.q_net2.state_dict(),
            'target_q1_state_dict': self.target_q_net1.state_dict(),
            'target_q2_state_dict': self.target_q_net2.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.q_net1.load_state_dict(checkpoint['q1_state_dict'])
        self.q_net2.load_state_dict(checkpoint['q2_state_dict'])
        self.target_q_net1.load_state_dict(checkpoint['target_q1_state_dict'])
        self.target_q_net2.load_state_dict(checkpoint['target_q2_state_dict'])

def train(data_path, num_epochs=100, batch_size=256, save_path=None):
    # 시드 설정
    set_seed()
    
    # 데이터 준비
    print("Loading and preprocessing data...")
    dataset = SepsisDataset(data_path)
    train_dataset = OfflineRLDataset(
        dataset.states[:-1],
        dataset.actions[:-1],
        dataset.rewards[:-1],
        dataset.states[1:],
        np.zeros(len(dataset.states)-1)  # done 신호
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # 모델 초기화
    state_dim = len(dataset.state_cols)
    action_dim = len(dataset.action_cols)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CQL(state_dim, action_dim, device)
    
    # 학습 루프
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch in train_loader:
            losses = model.train_step(batch)
            epoch_losses.append(losses)
        
        # 에포크 결과 출력
        if (epoch + 1) % 10 == 0:
            mean_losses = {k: np.mean([loss[k] for loss in epoch_losses])
                         for k in epoch_losses[0].keys()}
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Losses: {mean_losses}")
    
    # 모델 저장
    if save_path:
        model.save(save_path)
        print(f"Model saved to {save_path}")
    
    return model

def evaluate_model(model, test_states):
    """
    학습된 모델을 평가하는 함수
    """
    model.policy_net.eval()
    with torch.no_grad():
        states = torch.FloatTensor(test_states).to(model.device)
        actions = model.policy_net(states)
    return actions.cpu().numpy()

if __name__ == "__main__":
    # 설정
    DATA_PATH = "path_to_your_sepsis_data.csv"
    SAVE_PATH = "sepsis_cql_model.pth"
    NUM_EPOCHS = 100
    BATCH_SIZE = 256
    
    # 학습 실행
    trained_model = train(
        data_path=DATA_PATH,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        save_path=SAVE_PATH
    )
    
    print("Training completed!")
