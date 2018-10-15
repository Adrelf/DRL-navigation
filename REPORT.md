# Algorithm
DQN with some improvements:
  - Deep Reinforcement Learning with Double Q-learning
  - Dueling Network Architecture
  - Prioritized Experience Replay
  
# Model architecture

# Hyperparameters tuning
The most important parameter to tune is the learnin rate.
A basic grid search method can be applied to find the optimal value for the learning rate.

Hyperparamters (not optimized)
  - learning rate: 5e-4
  - optimizer: Adam
  - batch size: 64
  - discount factor: 0.99
  - primary network update: 4
  - soft update for target paramters: 1e-3
  - exploration research parameters:
      - starting value of epsilon: 1.0
      - minimum value of epsilon: 0.01
      - multiplicative factor (per episode) for decreasing epsilon: 0.99
  - PER parameters:
      - alpha parameter for prioritized replay buffer: 0.6
      - initial value of beta for prioritized replay buffer: 0.4
      - number of iterations over which beta will be annealed from initial value to 1.0: total_timesteps
      - epsilon to add to the TD errors when updating priorities: 1e-6
      
# Performance assessment

Solved Requirements: Considered solved when the average reward is greater than or equal to +13 over 100 consecutive trials.

Important: no optimization on stettings parameters.

Performance for DQN Agent w/o PER.
Environment solved in 330 episodes. Average score: 13.13

![alt text](https://github.com/Adrelf/DRL-navigation/blob/master/images/Banana_Nav.png)

# Future improvements
