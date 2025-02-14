import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import List, Tuple

# 재현성을 위한 시드 설정
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# DQN 신경망
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# 리플레이 버퍼 클래스
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

# DQN 에이전트
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=5e-4)
        self.memory = ReplayBuffer(capacity=50000)
        
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997
        self.target_update = 5
        self.action_dim = action_dim
    
    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def train(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0
        
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).view(-1, 1).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).view(-1, 1).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).view(-1, 1).to(self.device)
        
        current_q_values = self.q_network(state_batch).gather(1, action_batch)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0].view(-1, 1)
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # 그래디언트 클리핑 추가
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# 학습 함수
def train_dqn(env_name: str = "CartPole-v1", episodes: int = 500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 보상 정규화 추가
            normalized_reward = reward / 100.0
            
            agent.memory.push(state, action, normalized_reward, next_state, done)
            loss = agent.train()
            
            state = next_state
            total_reward += reward
            
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        agent.update_epsilon()
        rewards_history.append(total_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    return rewards_history

# 결과 시각화 함수
def plot_rewards(rewards: List[float]):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

# 메인 실행
if __name__ == "__main__":
    set_seed(42)
    rewards = train_dqn()
    plot_rewards(rewards)
