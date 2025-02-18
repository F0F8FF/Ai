import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import matplotlib.pyplot as plt
from typing import Tuple, List
import random

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class ReplayBuffer:
    def __init__(self, capacity: int = 1000000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self) -> int:
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 256):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(state)
        mean = torch.tanh(self.mean(features)) * self.max_action
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        
        action = torch.clamp(x_t, -self.max_action, self.max_action)
        
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

class SAC:
    def __init__(
        self, 
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: torch.device,
        env_name: str = "BipedalWalker-v3",
        hidden_dim: int = 256,
        discount: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        batch_size: int = 256,
        reward_scale: float = 1.0,
        auto_entropy_tuning: bool = True,
        target_entropy: float = None
    ):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.env_name = env_name
        
        # 자동 엔트로피 조절
        self.auto_entropy_tuning = auto_entropy_tuning
        if self.auto_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -float(action_dim)
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 0.2
        
        # BipedalWalker-v3 특화 설정
        if env_name == "BipedalWalker-v3":
            self.reward_scale = 1.0
            self.target_entropy = -1.0
            self.tau = 0.005
            self.discount = 0.99
            hidden_dim = 400  # 더 큰 네트워크
        
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer(capacity=1000000)
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if evaluate:
                mean, _ = self.actor.forward(state)
                return mean.cpu().numpy()[0]
            else:
                action, _ = self.actor.sample(state)
                return action.cpu().numpy()[0]
    
    def train(self) -> Tuple[float, float, float]:
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0, 0
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        reward = reward * self.reward_scale
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.discount * (target_q - self.alpha * next_log_pi)
        
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        action_pi, log_pi = self.actor.sample(state)
        q1_pi = self.critic.q1_forward(state, action_pi)
        
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            
            actor_loss = (self.alpha.detach() * log_pi - q1_pi).mean()
        else:
            actor_loss = (self.alpha * log_pi - q1_pi).mean()
            alpha_loss = torch.tensor(0.0).to(self.device)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        return actor_loss.item(), critic_loss.item(), alpha_loss.item()

def train_sac(env_name: str, episodes: int = 1000, evaluate: bool = True):
    env = gym.make(env_name)
    eval_env = gym.make(env_name) if evaluate else None
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SAC(state_dim, action_dim, max_action, device, env_name=env_name)
    
    rewards_history = []
    eval_rewards_history = []
    actor_losses = []
    critic_losses = []
    alpha_losses = []
    
    for episode in range(episodes):
        state = env.reset()[0]
        episode_reward = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        episode_alpha_loss = 0
        steps = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > agent.batch_size:
                actor_loss, critic_loss, alpha_loss = agent.train()
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                episode_alpha_loss += alpha_loss
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        rewards_history.append(episode_reward)
        
        if steps > 0:
            actor_losses.append(episode_actor_loss / steps)
            critic_losses.append(episode_critic_loss / steps)
            alpha_losses.append(episode_alpha_loss / steps)
        
        # 평가
        if evaluate and episode % 10 == 0:
            eval_reward = 0
            state = eval_env.reset()[0]
            while True:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                eval_reward += reward
                state = next_state
                if done:
                    break
            eval_rewards_history.append(eval_reward)
            print(f"Episode {episode}, Train Reward: {episode_reward:.2f}, Eval Reward: {eval_reward:.2f}")
        else:
            print(f"Episode {episode}, Train Reward: {episode_reward:.2f}")
    
    return rewards_history, eval_rewards_history, actor_losses, critic_losses, alpha_losses

def plot_training_results(rewards: List[float], eval_rewards: List[float], 
                         actor_losses: List[float], critic_losses: List[float], 
                         alpha_losses: List[float], env_name: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot rewards
    ax1.plot(rewards, label='Train Rewards', alpha=0.6)
    if len(eval_rewards) > 0:
        eval_x = np.linspace(0, len(rewards)-1, len(eval_rewards))
        ax1.plot(eval_x, eval_rewards, label='Eval Rewards', linewidth=2)
    ax1.set_title(f'Training Rewards (SAC) - {env_name}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    
    # Plot losses
    ax2.plot(actor_losses, label='Actor Loss', alpha=0.6)
    ax2.plot(critic_losses, label='Critic Loss', alpha=0.6)
    ax2.plot(alpha_losses, label='Alpha Loss', alpha=0.6)
    ax2.set_title('Training Losses')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    set_seed(42)
    
    env_name = "BipedalWalker-v3"
    episodes = 1000
    
    print(f"\nTraining on {env_name}")
    rewards, eval_rewards, actor_losses, critic_losses, alpha_losses = train_sac(
        env_name, 
        episodes=episodes,
        evaluate=True
    )
    plot_training_results(rewards, eval_rewards, actor_losses, critic_losses, alpha_losses, env_name)
