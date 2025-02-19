# TD3 PyTorch Implementation

PyTorch implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm tested on three different environments.

## Files
- `td3_cheetah.py`: Implementation for HalfCheetah-v4 (using gym)
- `td3_hopper.py`: Implementation for Hopper-v4 (using gymnasium)
- `td3_pendulum.py`: Implementation for Pendulum-v1 (using gym)


## Key Features
- Twin Delayed DDPG (TD3) with experience replay
- Target networks with soft updates
- Delayed policy updates
- Real-time training visualization

## Environment Specific Settings

### HalfCheetah-v4
- Episodes: 1000
- Hidden dim: 400
- Policy noise: 0.2
- Noise clip: 0.5

### Hopper-v4
- Episodes: 1000
- Hidden dim: 256
- Policy noise: 0.2
- Noise clip: 0.5
- Minimum steps before training: 5000

### Pendulum-v1
- Episodes: 200
- Hidden dim: 400
- Policy noise: 0.1
- Noise clip: 0.25
- Reward scale: 0.05

## References
- [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)

## License
MIT License
