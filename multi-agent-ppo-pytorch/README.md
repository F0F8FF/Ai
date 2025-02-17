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
| Network Size | 128 units | 256 units |
| Activation | Tanh | ReLU |
| Normalization | LayerNorm | None |
| Batch Size | 256 | 2048 |
| Learning Rate | 1e-4 | 3e-4 |

## Hyperparameters

### Stable Version
- Gamma: 0.98
- GAE Lambda: 0.92
- PPO Clip: 0.15
- Value Coefficient: 0.8
- Entropy Coefficient: 0.02
- PPO Epochs: 5

### Experimental Version
- Gamma: 0.99
- GAE Lambda: 0.95
- PPO Clip: 0.2
- Value Coefficient: 0.5
- Entropy Coefficient: 0.01
- PPO Epochs: 10


## Performance

### Stable Version
- Consistent performance
- Lower variance
- Better agent cooperation
- Best reward: -7.21

### Experimental Version
- Higher learning potential
- More exploration
- Higher variance
- Suitable for experimentation

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
