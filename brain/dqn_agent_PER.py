import numpy as np
import random
from brain.model import QNetwork
from utils.memory import PrioritizedReplayBuffer, ReplayBuffer
from utils.schedules import LinearSchedule
import torch
import torch.optim as optim

N_EPISODES = 1000
BUFFER_SIZE = int(1e5)  # replay buffer size
PRIORITIZED_REPLAY_ALPHA = 0.6 # alpha parameter for prioritized replay buffer
PRIORITIZED_REPLAY_BETA0 = 0.4 # initial value of beta for prioritized replay buffer
PRIORITIZED_REPLAY_BETA_ITERS = None # number of iterations over which beta will be annealed from initial value
                                      # to 1.0. If set to None equals to total_timesteps.
PRIORITIZED_REPLAY_EPS = 1e-6 # epsilon to add to the TD errors when updating priorities
BATCH_SIZE = 64  # minibatch size
LEARNING_STARTS = 100
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
TRAIN_FREQ = 1 # update the model every `train_freq` steps
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent_PER():
  """Interacts with and learns from the environment."""

  def __init__(self, state_size, action_size, seed):
    """Initialize an Agent object.

    Params
    ======
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        seed (int): random seed
    """
    self.state_size = state_size
    self.action_size = action_size
    self.seed = random.seed(seed)

    # Q-Network
    self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
    self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
    self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

    # Replay memory
    # Create the replay buffer
    self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=PRIORITIZED_REPLAY_ALPHA)
    if PRIORITIZED_REPLAY_BETA_ITERS is None:
        prioritized_replay_beta_iters = N_EPISODES
    self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                     initial_p=PRIORITIZED_REPLAY_BETA0,
                                     final_p=1.0)
    # Initialize time step (for updating every UPDATE_EVERY steps)
    self.t_step = 0
    self.t = 0

  def step(self, state, action, reward, next_state, done):
    # Save experience in replay memory
    self.memory.add(state, action, reward, next_state, float(done))
    self.t += 1

    # Learn every UPDATE_EVERY time steps
    self.t_step = (self.t_step + 1) % UPDATE_EVERY

    # Update target network periodically
    if self.t_step == 0:
      udpate_target_network_flag = True
    else:
      udpate_target_network_flag = False

    if self.t > LEARNING_STARTS and (self.t % TRAIN_FREQ) == 0:
      # Minimize the error in Bellman's equation on a batch sampled from replay buffer
      experiences = self.memory.sample(BATCH_SIZE, beta=self.beta_schedule.value(self.t))
      td_errors = self.learn(experiences, GAMMA, udpate_target_network_flag)
      (states, actions, rewards, next_states, dones, weights, batch_idxes) = experiences
      new_priorities = np.abs(td_errors) + PRIORITIZED_REPLAY_EPS
      self.memory.update_priorities(batch_idxes, new_priorities)


  def act(self, state, eps=0.):
    """Returns actions for given state as per current policy.

    Params
    ======
        state (array_like): current state
        eps (float): epsilon, for epsilon-greedy action selection
    """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    self.qnetwork_local.eval()
    with torch.no_grad():
      action_values = self.qnetwork_local(state)
    self.qnetwork_local.train()

    # Epsilon-greedy action selection
    if random.random() > eps:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))

  def learn(self, experiences, gamma, udpate_target_network_flag):
    """Update value parameters using given batch of experience tuples.

    Params
    ======
        experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor
    """
    states, actions, rewards, next_states, dones, weights, batch_idxes = experiences
    states = torch.from_numpy(np.vstack([state for state in states])).float().to(device)
    actions = torch.from_numpy(np.vstack([action for action in actions])).long().to(device)
    rewards = torch.from_numpy(np.vstack([reward for reward in rewards])).float().to(device)
    next_states = torch.from_numpy(np.vstack([next_state for next_state in next_states])).float().to(device)
    dones = torch.from_numpy(np.vstack([done for done in dones])).float().to(device)
    weights = torch.from_numpy(np.vstack([weight for weight in weights])).float().to(device)

    # Get max predicted Q values (for next states) from target model
    Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    # Compute Q targets for current states
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

    # Get expected Q values from local model
    Q_expected = self.qnetwork_local(states).gather(1, actions)

    # Compute td error
    td_error = Q_expected - Q_targets
    td_error_ = td_error.detach().numpy()
    # Compute loss
    loss = td_error**2
    #loss = F.mse_loss(Q_expected, Q_targets)
    # Minimize the loss
    self.optimizer.zero_grad()
    (weights * loss).mean().backward()
    self.optimizer.step()

    if udpate_target_network_flag:
      # ------------------- update target network ------------------- #
      self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    return td_error_


  def soft_update(self, local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
      target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)