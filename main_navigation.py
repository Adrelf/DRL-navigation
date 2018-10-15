from unityagents import UnityEnvironment
from collections import deque
from brain.dqn_agent import Agent
from brain.dqn_agent_PER import Agent_PER
import matplotlib.pyplot as plt
import torch
import numpy as np

# Flag to indicate if Agent is based on PER or not: flag_PER = 1 ==> Agent with PER
flag_PER = 0

# Create the environment
env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

# Get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents in the environment
print('Number of agents:', len(env_info.agents))

# Number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# Examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

if flag_PER:
  # DQN Agent with prioritized experience replay
  agent = Agent_PER(state_size=state_size, action_size=action_size, seed=0)
else:
  # DQN Agent
  agent = Agent(state_size=state_size, action_size=action_size, seed=0)


def dqn(env, n_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.99):
  """Deep Q-Learning

  Params
  ======
      n_episodes (int): maximum number of training episodes
      max_t (int): maximum number of timesteps per episode
      eps_start (float): starting value of epsilon, for epsilon-greedy action selection
      eps_end (float): minimum value of epsilon
      eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
  """
  scores = []  # list containing scores from each episode
  scores_window = deque(maxlen=100)  # last 100 scores
  eps = eps_start  # initialize epsilon
  for i_episode in range(1, n_episodes + 1):
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]  # get the initial state
    score = 0
    for t in range(max_t):
      action = agent.act(state, eps)
      env_info = env.step(action)[brain_name]
      next_state = env_info.vector_observations[0]  # get the next state
      reward = env_info.rewards[0]  # get the reward
      done = env_info.local_done[0]  # see if episode has finished
      agent.step(state, action, reward, next_state, done)
      state = next_state
      score += reward
      if done:
        break
    scores_window.append(score)  # save most recent score
    scores.append(score)  # save most recent score
    eps = max(eps_end, eps_decay * eps)  # decrease epsilon
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if i_episode % 100 == 0:
      print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    if np.mean(scores_window) >= 13.0:
      print(
        '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_window)))
      if flag_PER:
        torch.save(agent.qnetwork_local.state_dict(), './saved_models/model_PER.pth')
      else:
        torch.save(agent.qnetwork_local.state_dict(), './saved_models/model.pth')
      break
  return scores

scores = dqn(env)
env.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

