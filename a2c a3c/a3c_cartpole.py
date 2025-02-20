import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(ActorCritic, self).__init__()
        
        # 매우 단순한 네트워크
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU()
        )
        
        # 직접적인 연결
        self.actor = nn.Linear(32, n_actions)
        self.critic = nn.Linear(32, 1)
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.shared(x)
        policy = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return policy, value

class Worker(mp.Process):
    def __init__(self, worker_id, global_model, optimizer, global_ep_idx):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.global_model = global_model
        self.optimizer = optimizer
        self.global_ep_idx = global_ep_idx
        
        self.env = gym.make('CartPole-v1')
        self.local_model = ActorCritic(4, 2)
        self.local_model.load_state_dict(self.global_model.state_dict())
        
        self.gamma = 0.99
        self.t_max = 20  # 더 긴 업데이트 간격
        self.entropy_beta = 0.01
        self.epsilon = 0.2  # PPO 클리핑 범위
    
    def run(self):
        print(f"Worker {self.worker_id} started")
        while self.global_ep_idx.value < 1000:
            state = self.env.reset()[0]
            done = False
            episode_reward = 0
            
            states, actions, rewards = [], [], []
            values, log_probs = [], []
            steps = 0
            
            while not done:
                state_tensor = torch.FloatTensor(state)
                policy, value = self.local_model(state_tensor)
                
                # epsilon-greedy 탐험
                if np.random.random() < max(0.1, 1.0 - self.global_ep_idx.value / 500):
                    action = np.random.randint(2)
                    action_tensor = torch.tensor([action])
                    dist = Categorical(policy)
                    log_prob = dist.log_prob(action_tensor)
                else:
                    dist = Categorical(policy)
                    action_tensor = dist.sample()
                    action = action_tensor.item()
                    log_prob = dist.log_prob(action_tensor)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                reward = min(1.0, reward * 0.01)
                
                states.append(state)
                actions.append(action_tensor)
                rewards.append(reward)
                values.append(value.squeeze())
                log_probs.append(log_prob)
                
                state = next_state
                episode_reward += reward * 100
                steps += 1
                
                # t_max 스텝마다 또는 에피소드가 끝났을 때 업데이트
                if (steps % self.t_max == 0 or done) and len(states) > 0:
                    with torch.no_grad():
                        _, next_value = self.local_model(torch.FloatTensor(next_state))
                        next_value = next_value.squeeze() if not done else 0
                    
                    # GAE 계산
                    gae = 0
                    returns = []
                    for r, v in zip(reversed(rewards), reversed(values)):
                        delta = r + self.gamma * next_value - v
                        gae = delta + self.gamma * 0.95 * gae
                        returns.insert(0, gae + v)
                        next_value = v
                    
                    # 배치 처리
                    states = torch.FloatTensor(np.array(states))
                    actions = torch.cat(actions)
                    returns = torch.FloatTensor(returns).unsqueeze(1)
                    log_probs = torch.stack(log_probs)
                    
                    # 현재 정책으로 재평가
                    policy, values = self.local_model(states)
                    dist = Categorical(policy)
                    entropy = dist.entropy().mean()
                    
                    # Advantage 계산
                    advantages = returns - values.detach()
                    
                    # Actor loss (단순화된 버전)
                    actor_loss = -(log_probs * advantages.squeeze()).mean()
                    
                    # Critic loss
                    critic_loss = F.mse_loss(values, returns)
                    
                    # Total loss
                    total_loss = actor_loss + 0.5 * critic_loss - self.entropy_beta * entropy
                    
                    # 업데이트
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 0.5)
                    
                    for global_param, local_param in zip(self.global_model.parameters(),
                                                       self.local_model.parameters()):
                        if global_param.grad is None:
                            global_param.grad = local_param.grad
                    
                    self.optimizer.step()
                    self.local_model.load_state_dict(self.global_model.state_dict())
                    
                    # 버퍼 초기화
                    states, actions, rewards = [], [], []
                    values, log_probs = [], []
                
                with self.global_ep_idx.get_lock():
                    if done:
                        self.global_ep_idx.value += 1
                        if self.global_ep_idx.value % 10 == 0:
                            print(f'Episode: {self.global_ep_idx.value}, Worker: {self.worker_id}, '
                                  f'Reward: {episode_reward}')

def train():
    global_model = ActorCritic(4, 2)
    global_model.share_memory()
    
    optimizer = torch.optim.Adam(global_model.parameters(), lr=0.0003)  
    global_ep_idx = mp.Value('i', 0)
    
    workers = [Worker(i, global_model, optimizer, global_ep_idx) 
              for i in range(4)]
    
    [w.start() for w in workers]
    [w.join() for w in workers]
    
    return global_model

if __name__ == "__main__":
    set_seed()
    
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    print("Training started...")
    trained_model = train()
    print("Training finished!")
