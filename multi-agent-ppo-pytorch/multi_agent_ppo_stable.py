import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from pettingzoo.mpe import simple_spread_v3
from typing import List, Dict, Tuple
import random

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.action_probs = []
        self.values = []
        self.dones = []
    
    def push(self, state, action, reward, next_state, action_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.action_probs.append(action_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.action_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(ActorCritic, self).__init__()
       
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), 
            lr=1e-4,
            eps=1e-5,
            betas=(0.9, 0.999)
        )
        
        self.memory = Memory()
        
        # 하이퍼파라미터 최적화
        self.gamma = 0.98
        self.gae_lambda = 0.92
        self.clip_epsilon = 0.15
        self.value_coef = 0.8
        self.entropy_coef = 0.02
        self.max_grad_norm = 0.3
        self.batch_size = 256
        self.ppo_epochs = 5
        
        # 보상 정규화를 위한 running statistics
        self.reward_mean = 0
        self.reward_std = 1
        self.alpha = 0.05
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, value = self.actor_critic(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_prob = dist.log_prob(action)
        
        return action.item(), action_prob.item(), value.item()
    
    def normalize_reward(self, reward):
        normalized_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        self.reward_mean = (1 - self.alpha) * self.reward_mean + self.alpha * reward
        self.reward_std = (1 - self.alpha) * self.reward_std + self.alpha * abs(reward - self.reward_mean)
        return normalized_reward
    
    def compute_gae(self, rewards, values, dones, next_value):
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
        
        return torch.FloatTensor(advantages).to(self.device)
    
    def update(self):
        if len(self.memory) == 0:
            return
        
        # 데이터 준비
        states = torch.FloatTensor(np.array(self.memory.states)).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_probs = torch.FloatTensor(self.memory.action_probs).to(self.device)
        rewards = torch.FloatTensor(self.memory.rewards).to(self.device)
        values = torch.FloatTensor(self.memory.values).to(self.device)
        dones = torch.FloatTensor(self.memory.dones).to(self.device)
        
        # GAE 계산
        with torch.no_grad():
            _, next_value = self.actor_critic(
                torch.FloatTensor(self.memory.next_states[-1]).unsqueeze(0).to(self.device)
            )
            advantages = self.compute_gae(self.memory.rewards, self.memory.values, 
                                        self.memory.dones, next_value.item())
            returns = advantages + values
        
        # PPO 업데이트
        for _ in range(self.ppo_epochs):
            action_probs, current_values = self.actor_critic(states)
            dist = Categorical(action_probs)
            current_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Actor loss
            ratio = torch.exp(current_probs - old_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            value_loss = 0.5 * (returns - current_values.squeeze()).pow(2).mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        self.memory.clear()

def train(episodes=1000):
    env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=20)
    
    state_dim = 18
    action_dim = 5
    
    agents = {agent_id: PPOAgent(state_dim, action_dim) 
             for agent_id in env.possible_agents}
    
    rewards_history = {agent_id: [] for agent_id in env.possible_agents}
    best_reward = float('-inf')
    
    # 이동 평균 계산을 위한 윈도우
    window_size = 50
    moving_averages = {agent_id: [] for agent_id in env.possible_agents}
    
    for episode in range(episodes):
        env.reset()
        episode_rewards = {agent_id: 0 for agent_id in env.possible_agents}
        done = False
        
        while not done:
            for agent_id in env.agents:
                observation, reward, termination, truncation, _ = env.last()
                done = termination or truncation
                
                if done:
                    break
                
                action, prob, value = agents[agent_id].select_action(observation)
                
                # 보상 정규화 및 클리핑
                norm_reward = agents[agent_id].normalize_reward(reward)
                norm_reward = np.clip(norm_reward, -0.5, 0.5)
                
                agents[agent_id].memory.push(
                    observation, action, norm_reward, observation,
                    prob, value, done
                )
                
                episode_rewards[agent_id] += reward
                
                if len(agents[agent_id].memory) >= agents[agent_id].batch_size:
                    agents[agent_id].update()
                
                env.step(action)
        
        # 에피소드 종료 후 업데이트
        for agent_id in env.possible_agents:
            if len(agents[agent_id].memory) > 0:
                agents[agent_id].update()
        
        # 결과 저장 및 출력
        avg_reward = sum(episode_rewards.values()) / len(episode_rewards)
        if avg_reward > best_reward:
            best_reward = avg_reward
        
        for agent_id in env.possible_agents:
            rewards_history[agent_id].append(episode_rewards[agent_id])
            # 이동 평균 계산
            if len(rewards_history[agent_id]) >= window_size:
                moving_avg = np.mean(rewards_history[agent_id][-window_size:])
                moving_averages[agent_id].append(moving_avg)
        
        if episode % 10 == 0:
            current_avgs = {
                agent_id: np.mean(rewards_history[agent_id][-10:])
                for agent_id in env.possible_agents
            }
            print(f"Episode {episode}, Current Rewards: {current_avgs}, Best: {best_reward:.2f}")
            if episode >= window_size:
                smooth_avgs = {
                    agent_id: moving_averages[agent_id][-1]
                    for agent_id in env.possible_agents
                }
                print(f"Smooth Average (last {window_size}): {smooth_avgs}")
    
    return rewards_history, moving_averages

def plot_rewards(rewards_history):
    plt.figure(figsize=(10, 5))
    for agent_id, rewards in rewards_history.items():
        plt.plot(rewards, label=agent_id)
    plt.title('Training Rewards (Multi-Agent PPO)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    set_seed(42)
    rewards_history, _ = train()
    plot_rewards(rewards_history)
