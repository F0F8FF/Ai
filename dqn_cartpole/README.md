# DQN CartPole

PyTorch implementation of Deep Q-Network (DQN) for solving the CartPole environment.

## Features
- Deep Q-Network with Experience Replay
- Target Network for Stable Learning
- Gradient Clipping
- Reward Normalization
- Dropout Layers for Regularization


## Results
![Figure_1](https://github.com/user-attachments/assets/e144d28f-a48c-49d7-bc31-7e3780bb17c4)


## Architecture
- Input Layer: 4 (State Dimension)
- Hidden Layers: 256 units with ReLU and Dropout
- Output Layer: 2 (Action Dimension)

## Hyperparameters
- Learning Rate: 5e-4
- Batch Size: 128
- Replay Buffer Size: 50000
- Target Update Frequency: 5 episodes
- Discount Factor (γ): 0.99
- Initial Epsilon: 1.0
- Minimum Epsilon: 0.05
- Epsilon Decay: 0.997
