import matplotlib.pyplot as plt
import numpy as np

eps_start = 1.0
eps_end = 0.01
eps_decay = 0.9985
n_episodes = 10000
eps = eps_start

eps_record = np.zeros((n_episodes))
eps_record[0] = eps
for idx in range(n_episodes-1):
  eps = max(eps_end, eps_decay * eps)  # decrease epsilon
  eps_record[idx+1] = eps

# data display
fig = plt.figure()
plt.plot(np.arange(n_episodes), eps_record)
plt.ylabel('eps')
plt.xlabel('Episode #')
plt.show()