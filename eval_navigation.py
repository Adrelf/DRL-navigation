from unityagents import UnityEnvironment
from brain.dqn_agent import Agent
from brain.dqn_agent_PER import Agent_PER
import torch
import time

# Flag to indicate if Agent is based on PER or not: flag_PER = 1 ==> Agent with PER
flag_PER = 0

env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
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

# load the weights from file
if flag_PER:
  agent.qnetwork_local.load_state_dict(torch.load('./saved_models/model_PER.pth'))
else:
  agent.qnetwork_local.load_state_dict(torch.load('./saved_models/model.pth'))


for i in range(2):
  env_info = env.reset(train_mode=True)[brain_name]
  state = env_info.vector_observations[0]  # get the initial state
  score = 0
  while True:
    time.sleep(.02)
    action = agent.act(state)
    env_info = env.step(action)[brain_name]
    state = env_info.vector_observations[0]  # get the next state
    reward = env_info.rewards[0]  # get the reward
    done = env_info.local_done[0]  # see if episode has finished
    score += reward
    if done:
      break
  print('Episode = {}'.format(i))
  print('Score = {}'.format(score))

env.close()