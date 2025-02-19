import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List
import random
import matplotlib.pyplot as plt

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
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 400):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.max_action = max_action
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 400):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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

class TD3:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: torch.device,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        lr: float = 3e-4,
        hidden_dim: int = 400
    ):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_action = max_action
        
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer()
        self.total_it = 0
    
    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state).cpu().numpy()[0]
            if noise > 0:
                noise = np.random.normal(0, noise * self.max_action, size=action.shape)
                action = np.clip(action + noise, -self.max_action, self.max_action)
        return action
    
    def train(self, batch_size: int = 256) -> Tuple[float, float]:
        self.total_it += 1
        
        # Sample replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.discount * target_q
        
        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = 0.0
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            actor_loss = actor_loss.item()
        
        return actor_loss, critic_loss.item()

def train_td3(env_name: str, episodes: int = 1000, evaluate: bool = True):
    env = gym.make(env_name)
    eval_env = gym.make(env_name) if evaluate else None
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = TD3(state_dim, action_dim, max_action, device)
    
    rewards_history = []
    eval_rewards_history = []
    actor_losses = []
    critic_losses = []
    
    if env_name == "HalfCheetah-v4":
        reward_scale = 1.0
        eval_noise = 0.1
        initial_noise = 0.3
        final_noise = 0.1
    else:
        reward_scale = 1.0
        eval_noise = 0.0
        initial_noise = 0.2
        final_noise = 0.1
    
    for episode in range(episodes):
        # Noise annealing
        current_noise = initial_noise - (initial_noise - final_noise) * (episode / episodes)
        
        state = env.reset()[0]
        episode_reward = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        steps = 0
        
        while True:
            action = agent.select_action(state, noise=current_noise)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            reward = reward * reward_scale
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > 256:
                actor_loss, critic_loss = agent.train()
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        rewards_history.append(episode_reward)
        
        if steps > 0:
            actor_losses.append(episode_actor_loss / steps)
            critic_losses.append(episode_critic_loss / steps)
        
        # Evaluation
        if evaluate and episode % 10 == 0:
            eval_reward = 0
            state = eval_env.reset()[0]
            while True:
                action = agent.select_action(state, noise=eval_noise)
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
    
    return rewards_history, eval_rewards_history, actor_losses, critic_losses

def plot_training_results(rewards: List[float], eval_rewards: List[float], 
                         actor_losses: List[float], critic_losses: List[float], 
                         env_name: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot rewards
    ax1.plot(rewards, label='Train Rewards', alpha=0.6)
    if len(eval_rewards) > 0:
        eval_x = np.linspace(0, len(rewards)-1, len(eval_rewards))
        ax1.plot(eval_x, eval_rewards, label='Eval Rewards', linewidth=2)
    ax1.set_title(f'Training Rewards (TD3) - {env_name}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    
    # Plot losses
    ax2.plot(actor_losses, label='Actor Loss', alpha=0.6)
    ax2.plot(critic_losses, label='Critic Loss', alpha=0.6)
    ax2.set_title('Training Losses')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    set_seed(42)
    
    env_name = "HalfCheetah-v4"
    episodes = 1000
    
    print(f"\nTraining on {env_name}")
    rewards, eval_rewards, actor_losses, critic_losses = train_td3(
        env_name, 
        episodes=episodes,
        evaluate=True
    )
    plot_training_results(rewards, eval_rewards, actor_losses, critic_losses, env_name)
