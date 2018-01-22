# hindsight-experience-replay
Hindsight Experience Replay - Bit flipping experiment in Tensorflow.

A very simple implementation of https://arxiv.org/pdf/1707.01495.pdf bit flipping experiment in Tensorflow.

This implementation includes:
  * Double DQN with 1 hidden layer of size 256.
  * Hindsight experience replay memory with "K-future" strategy.
  * A very simple bit-flipping evironment as mentioned in the original paper.

## TODO's
- [x] Plot training curve with respect to rewards.
- [ ] Train on bit length higher than 30
