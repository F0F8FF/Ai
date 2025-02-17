# Multi-Agent PPO Implementation for MPE

This repository contains two implementations of Multi-Agent Proximal Policy Optimization (MAPPO) for the simple_spread environment in Multi-Particle Environment (MPE).


## Implementations

### 1. Stable Version (`multi_agent_ppo_stable.py`)
- Optimized for stability and consistent performance
- Reward normalization with running statistics
- Moving average performance tracking
- Smaller network with LayerNorm (128 units)
- Best average reward: -15 to -20

### 2. Experimental Version (`multi_agent_ppo_experimental.py`)
- Focused on exploration and learning capacity
- Larger network architecture (256 units)
- Extended training cycles
- Higher variance in learning
- Average reward: -20 to -25

## Architecture Comparison

| Feature | Stable | Experimental |
|---------|--------|--------------|
| Shared Network Size | 128 units | 256 units |
| Activation | Tanh | ReLU |
| Normalization | LayerNorm | None |
| Batch Size | 256 | 64 |
| Learning Rate | 1e-4 | 3e-4 |
| Gamma | 0.98 | 0.99 |
| GAE Lambda | 0.92 | 0.95 |
| Clip Epsilon | 0.15 | 0.2 |
| Value Coefficient | 0.8 | 0.5 |
| Entropy Coefficient | 0.02 | 0.01 |
| PPO Epochs | 5 | 10 |
| Max Grad Norm | 0.3 | 0.5 |

## Performance

### 1. Stable Version
- Optimized for stability and consistent performance
- Reward normalization with running statistics
- Moving average performance tracking
- Smaller network with LayerNorm (128 units)
- Average reward: -15 to -20
- Best reward: -7.21

### 2. Experimental Version
- Focused on exploration and learning capacity
- Larger network architecture (256 units)
- Extended training cycles
- Higher variance in learning
- Average reward: -20 to -25

## Key Features

- Multi-Agent PPO implementation
- PettingZoo MPE simple_spread environment
- Actor-Critic architecture
- Generalized Advantage Estimation (GAE)
- Reward normalization and clipping
- Performance monitoring


## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [Multi-Agent Deep Reinforcement Learning](https://arxiv.org/abs/1910.00091)

## License

MIT License

## Contributing

Feel free to open issues or submit pull requests for improvements.
