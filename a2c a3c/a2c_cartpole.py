import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(ActorCritic, self).__init__()
        
        # 간단한 네트워크 구조
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

class A2CAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.001)
        
        self.gamma = 0.99
        self.entropy_coef = 0.01  # 엔트로피 계수
        self.max_grad_norm = 0.5  # gradient clipping
        self.states = []
        self.actions = []
        self.rewards = []
        
        # 최고 성능 모델 저장을 위한 변수들
        self.best_reward = float('-inf')
        self.best_model_state = None
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state)
        action = torch.multinomial(action_probs, 1).item()
        return action
    
    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def train(self):
        if len(self.states) == 0:
            return
        
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        
        # 할인된 리워드 계산
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # 액터-크리틱 출력 계산
        action_probs, state_values = self.actor_critic(states)
        
        # 어드밴티지 계산
        advantages = returns - state_values.squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)  # 정규화 추가
        
        # 액터 손실 계산
        log_probs = torch.log(action_probs + 1e-10)  # 수치 안정성
        selected_log_probs = log_probs[range(len(actions)), actions]
        entropy = -(action_probs * log_probs).sum(1).mean()
        actor_loss = -(selected_log_probs * advantages.detach()).mean() - self.entropy_coef * entropy
        
        # 크리틱 손실 계산
        critic_loss = F.smooth_l1_loss(state_values.squeeze(), returns)  # MSE 대신 Huber Loss
        
        # 전체 손실 계산
        loss = actor_loss + 0.5 * critic_loss
        
        # 역전파 및 최적화
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # 메모리 초기화
        self.states = []
        self.actions = []
        self.rewards = []

def train(env_name='CartPole-v1', episodes=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = A2CAgent(state_dim, action_dim)
    rewards_history = []
    
    best_reward = float('-inf')
    best_model_state = None
    
    for episode in range(episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward)
            episode_reward += reward
            state = next_state
            
            if done:
                agent.train()
        
        rewards_history.append(episode_reward)
        
        if episode >= 100:  
            avg_reward = np.mean(rewards_history[-100:]) 
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model_state = agent.actor_critic.state_dict()
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}, Average Reward: {avg_reward:.2f}")
    
    if best_model_state is not None:
        agent.actor_critic.load_state_dict(best_model_state)
    
    return rewards_history, agent

if __name__ == "__main__":
    rewards, agent = train()
    
    # 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
