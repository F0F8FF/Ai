import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from typing import List, Tuple

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.probs = []
        self.values = []
        self.dones = []
    
    def push(self, state, action, reward, next_state, prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.probs.append(prob)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.probs = []
        self.values = []
        self.dones = []

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(ActorCritic, self).__init__()
        
        # 공유 네트워크 층
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Actor (정책) 네트워크
        self.actor = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (가치) 네트워크
        self.critic = nn.Sequential(
            nn.Linear(256, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_probs, value

class PPO:
    def __init__(self, state_dim: int, action_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        
        self.memory = PPOMemory()
        
        # PPO 하이퍼파라미터
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.ppo_epochs = 10
        self.batch_size = 64
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.actor_critic(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            prob = dist.log_prob(action)
            
        return action.item(), prob.item(), value.item()
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   dones: List[bool], next_value: float) -> List[float]:
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            
        return advantages
    
    def update(self):
        # 데이터 준비
        states = torch.FloatTensor(self.memory.states).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_probs = torch.FloatTensor(self.memory.probs).to(self.device)
        values = torch.FloatTensor(self.memory.values).to(self.device)
        
        # GAE 계산
        with torch.no_grad():
            _, next_value = self.actor_critic(
                torch.FloatTensor(self.memory.next_states[-1]).unsqueeze(0).to(self.device)
            )
        
        advantages = self.compute_gae(
            self.memory.rewards, self.memory.values, 
            self.memory.dones, next_value.item()
        )
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + values
        
        # PPO 업데이트
        for _ in range(self.ppo_epochs):
            for idx in range(0, len(states), self.batch_size):
                batch_states = states[idx:idx + self.batch_size]
                batch_actions = actions[idx:idx + self.batch_size]
                batch_old_probs = old_probs[idx:idx + self.batch_size]
                batch_advantages = advantages[idx:idx + self.batch_size]
                batch_returns = returns[idx:idx + self.batch_size]
                
                # 현재 정책으로 행동 확률과 가치 계산
                action_probs, values = self.actor_critic(batch_states)
                dist = Categorical(action_probs)
                new_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 비율 계산
                ratio = torch.exp(new_probs - batch_old_probs)
                
                # PPO 목적 함수
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic 손실
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # 전체 손실
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 최적화
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        self.memory.clear()

def train_ppo(env_name: str = "CartPole-v1", episodes: int = 500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPO(state_dim, action_dim)
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False
        
        while not done:
            action, prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, prob, value, done)
            
            if len(agent.memory.states) >= agent.batch_size or done:
                agent.update()
            
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    return rewards_history

def plot_rewards(rewards: List[float]):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards (PPO)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

if __name__ == "__main__":
    set_seed(42)
    rewards = train_ppo()
    plot_rewards(rewards)
