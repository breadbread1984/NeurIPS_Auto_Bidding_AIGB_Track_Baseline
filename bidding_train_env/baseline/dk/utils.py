#!/usr/bin/python3

import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np
from bisect import bisect

class EpisodeReplayBuffer(Dataset):
  def __init__(self, csv_path, chunksize = 1000):
    super(EpisodeReplayBuffer, self).__init__()
    self.states, self.actions, self.returns_to_go, self.dones = list(), list(), list(), list()
    rewards = list()
    def safe_literal_eval(val):
      if pd.isna(val):
        return val
      try:
        return ast.literal_eval(val)
      except (ValueError, SyntaxError):
        print(ValueError)
        return val
    traj_len = 0
    for chunk in pd.read_csv(csv_path, chunksize = chunksize):
      chunk['state'] = chunk['state'].apply(safe_literal_eval)
      for index, row in chunk.iterrows():
        self.states.append(np.array(row['state']))
        self.actions.append(row['action'])
        self.dones.append(row['done'])
        rewards.append(row['reward'])
        traj_len += 1
        if row['done'] != 0:
          returns_to_go = self.discount_cumsum(rewards[-traj_len:])
          self.returns_to_go.extend(returns_to_go.tolist())
          traj_len = 0
    self.states = np.stack(self.states, axis = 0) # self.states.shape = (sample_num, state_dim)
    self.actions = np.expand_dims(self.actions, axis = -1) # self.actions.shape = (sample_num, 1)
    self.returns_to_go = np.expand_dims(self.returns_to_go, axis = -1) # self.returns_to_go.shape = (sample_num, 1)
    self.dones = np.expand_dims(self.dones, axis = -1) # self.dones.shape = (sample_num, 1)
    assert self.states.shape[0] == self.actions.shape[0] == self.returns_to_go.shape[0] == self.dones.shape[0]
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
      'states': self.states[index].astype(numpy.float32),
      'actions': self.actions[index].astype(numpy.float32),
      'returns_to_go': self.returns_to_go[index].astype(numpy.float32),
      'dones': self.dones[index]
    }

if __name__ == "__main__":
  data = EpisodeReplayBuffer('../../../data/trajectory/trajectory_data.csv')
  for example in data:
    print(example)
    if example['done'] != 0: break
