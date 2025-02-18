# SAC (Soft Actor-Critic) Implementation

PyTorch implementation of Soft Actor-Critic (SAC) algorithm for continuous control tasks. This implementation includes automatic entropy tuning and supports multiple environments.

## Supported Environments
- Pendulum-v1
- MountainCarContinuous-v0
- LunarLanderContinuous-v2
- BipedalWalker-v3

## Features
- Automatic entropy adjustment
- Environment-specific hyperparameter settings
- Evaluation mode during training
- Training visualization (rewards and losses)
- Gradient clipping (max norm: 1.0)
- Layer normalization
- Xavier uniform initialization


## Implementation Details
- Actor Network: 
  - 2 hidden layers (256 units each)
  - Layer normalization and Tanh activation
  - Gaussian policy with state-dependent mean and std
- Critic Network: 
  - Dual Q-networks
  - 2 hidden layers (256 units each)
  - Layer normalization and Tanh activation
- Replay Buffer: 1M size with uniform sampling
- Optimization: Adam optimizer

## Hyperparameters
- Learning rate: 3e-4
- Discount factor (gamma): 0.99
- Soft update coefficient (tau): 0.005
- Hidden layer size: 256 (400 for BipedalWalker)
- Batch size: 256
- Replay buffer size: 1,000,000
- Reward scale: Environment specific

## Environment-Specific Settings
### Pendulum-v1
- Reward scale: 0.1
- Episodes: 200
- Default target entropy

### MountainCarContinuous-v0
- Reward scale: 1.0
- Target entropy: -1.0
- Episodes: 200

### LunarLanderContinuous-v2
- Reward scale: 1.0
- Target entropy: -2.0
- Episodes: 500

### BipedalWalker-v3
- Reward scale: 1.0
- Target entropy: -1.0
- Discount factor: 0.99
- Tau: 0.005
- Episodes: 1000
- Hidden dim: 400 (larger network)

## Training Process
- Training evaluation every 10 episodes
- Automatic entropy tuning with environment-specific target entropy
- Gradient clipping for both actor and critic (max norm: 1.0)
- Plots include:
  - Training rewards
  - Evaluation rewards (every 10 episodes)
  - Actor, critic, and alpha losses


## Performance Benchmarks
- Pendulum-v1: Converges within 200 episodes
- MountainCarContinuous-v0: Solves environment consistently
- LunarLanderContinuous-v2: Achieves stable landing
- BipedalWalker-v3: 
  - Success threshold: 300+ reward
  - Convergence: ~300-400 episodes
  - Stable walking behavior achieved

## References
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)

## License
MIT License
