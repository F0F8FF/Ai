# A2C & A3C Implementation for CartPole

This project implements and compares A2C (Advantage Actor-Critic) and A3C (Asynchronous Advantage Actor-Critic) algorithms in the OpenAI Gym CartPole-v1 environment.


## Algorithm Comparison

### A2C (Synchronous)
- Single process operation
- Synchronous learning method
- More stable learning process
- Relatively simple implementation

### A3C (Asynchronous)
- Multi-process parallel learning
- Asynchronous learning method  
- Faster learning speed
- More complex implementation


### Key Techniques
- Actor-Critic architecture
- Advantage calculation
- Entropy bonus
- Epsilon-greedy exploration
- GAE (Generalized Advantage Estimation)

## Hyperparameters

### A2C
- Learning rate: 0.001
- Gamma: 0.99
- Entropy bonus: 0.01
- Batch size: 32

### A3C
- Learning rate: 0.005
- Gamma: 0.99
- t_max (update interval): 5
- Entropy bonus: 0.01
- Workers: 4

## Performance Comparison

### A2C
- Training time: ~1000 episodes
- Final performance: Average 300-400 points
- Stability: Moderate variance

### A3C
- Training time: ~1000 episodes
- Final performance: 500 points (perfect score)
- Stability: Very stable after 690 episodes

## Requirements

- Python 3.6+
- PyTorch
- Gymnasium
- NumPy


## Notes

- A3C requires 'spawn' multiprocessing mode on Windows
- Automatically uses GPU if CUDA is available
- A2C runs on single process with no platform restrictions

## References

- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [OpenAI Gym CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [Actor-Critic Methods](https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html#actor-critic)

## License

MIT License

## Conclusion

A3C demonstrates higher final performance and stability compared to A2C, but comes with implementation complexity and platform restrictions. A2C shows sufficient performance for simple environments.

## Conclusion

A3C demonstrates higher final performance and stability compared to A2C, but comes with implementation complexity and platform restrictions. A2C shows sufficient performance for simple environments.

