# AutonomousDecisionRL

An offline reinforcement learning project for autonomous vehicle decision-making using the nuScenes dataset and Decision Transformer architecture.

## Overview

This project implements a decision-making system for autonomous vehicles using offline reinforcement learning. It utilizes the nuScenes dataset to train a Transformer-based model that can predict vehicle trajectories based on historical state, action, and reward data.

## Features

- Transformer-based decision-making model for autonomous driving
- Integration with nuScenes dataset for real-world driving scenarios
- Data augmentation techniques for improved model generalization
- Trajectory prediction and visualization capabilities
- Comprehensive training pipeline with early stopping and learning rate scheduling


## Requirements

- Python 3.7+
- PyTorch
- nuScenes devkit
- numpy
- matplotlib
- tqdm


## Model Architecture

The Decision Transformer model (`autonomous_dt.py`) includes:
- State encoding (5-dimensional state space)
- Action prediction (2-dimensional action space)
- Multi-head attention mechanism
- Position encoding
- Returns-to-go conditioning

## Dataset

The project uses the nuScenes dataset with the following features:
- Vehicle state information (position, orientation, velocity)
- Action sequences
- Reward calculation based on driving metrics
- Data augmentation (random rotations and noise addition)

## Results

The model generates trajectories that:
- Follow realistic driving patterns
- Maintain smooth acceleration and steering
- Adapt to different initial conditions
- Consider target returns for trajectory optimization

## Future Work

- Implementation of additional driving scenarios
- Integration with real-time control systems
- Extension to multi-agent scenarios
- Improvement of reward function design

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- nuScenes dataset team for providing the autonomous driving dataset
- Decision Transformer paper authors for the architectural inspiration
