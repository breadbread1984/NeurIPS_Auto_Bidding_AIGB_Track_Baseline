#!/usr/bin/python3

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from bisect import bisect

class EpisodeReplayBuffer(Dataset):
  def __init__(self, csv_path, chunksize = 1000):
    super(EpisodeReplayBuffer, self).__init__()
    self.states, self.next_states, self.rewards, self.actions, self.returns_to_go, self.dones = list(), list(), list(), list(), list(), list()
    traj_len = 0
    for chunk in pd.read_csv(csv_path, chunksize = chunksize):
      for index, row in chunk.iterrows():
        self.states.append(np.array(row['state']))
        self.next_states.append(np.array(row['next_state']))
        self.rewards.append(row['reward'])
        self.actions.append(row['action'])
        self.dones.append(row['done'])
        traj_len += 1
        if row['done'] != 0:
          returns_to_go = self.discount_cumsum(self.rewards[-traj_len:])
          self.returns_to_go.extend(returns_to_go.tolist())
          traj_len = 0
    self.states = np.stack(self.states, axis = 0) # self.states.shape = (sample_num, state_dim)
    self.next_states = np.stack(self.next_states, axis = 0) # self.next_states.shape = (sample_num, state_dim)
    self.rewards = np.expand_dims(self.rewards, axis = -1) # self.rewards.shape = (sample_num, 1)
    self.actions = np.expand_dims(self.actions, axis = -1) # self.actions.shape = (sample_num, 1)
    self.returns_to_go = np.expand_dims(self.returns_to_go, axis = -1) # self.returns_to_go.shape = (sample_num, 1)
    self.dones = np.expand_dims(self.dones, axis = -1) # self.dones.shape = (sample_num, 1)
  def discount_cumsum(self, x, gamma = 0.8):
    x = np.array(x)
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
      discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum
  def __len__(self):
    return len(self.states)
  def __getitem__(self, index):
    return {
      'state': self.states[i],
      'next_state': self.next_states[i],
      'reward': self.rewards[i],
      'action': self.actions[i],
      'returns_to_go': self.returns_to_go[i],
      'done': self.dones[i]
    }

if __name__ == "__main__":
  data = EpisodeReplayBuffer('../../../data/trajectory/trajectory_data.csv')
  for example in data:
    print(example)
    if example['done'] != 0: break
