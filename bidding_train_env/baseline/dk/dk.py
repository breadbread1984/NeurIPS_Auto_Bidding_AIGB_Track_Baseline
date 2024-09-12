#!/usr/bin/python3

import random
from os import mkdir, makedirs
from os.path import join, exists, isdir
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class DecisionKAN(nn.Module):
  def __init__(self, state_dim, act_dim, gamma = 0.8, lr = 1e-4):
    super(DecisionKAN, self).__init__()
    self.gamma = gamma
    self.Q = nn.Sequential(
      nn.Linear(state_dim + act_dim, 8),
      nn.GELU(),
      nn.LayerNorm((8,)),
      nn.Linear(8, 4),
      nn.GELU(),
      nn.LayerNorm((4,)),
      nn.Linear(4, 1),
      nn.ReLU())
    self.pi = nn.Sequential(
      nn.LayerNorm((state_dim,)),
      nn.Linear(state_dim, 8),
      nn.GELU(),
      nn.LayerNorm((8,)),
      nn.Linear(8, 4),
      nn.GELU(),
      nn.LayerNorm((4,)),
      nn.Linear(4, act_dim),
      nn.ReLU())
    self.criterion = nn.MSELoss()
    self.optimizer = Adam(list(self.Q.parameters()) + list(self.pi.parameters()), lr = lr)
    self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0 = 5, T_mult = 2)
  def forward(self, states):
    actions_next = self.pi(states) # actions_next.shape = (batch, act_dim)
    return actions_next
  def get_action(self, states):
    states = torch.Tensor(states).to(torch.float32)
    states = torch.unsqueeze(states, dim = 0).to(next(self.pi.parameters()).device) # states.shape = (batch, act_dim)
    actions_next = self.forward(states)
    actions_next = actions_next.detach().cpu().numpy()[0]
    return actions_next
  def step(self, states, actions, returns_to_go, step):
    # s_t, a_t -> Q(s_t, a_t)
    inputs = torch.cat([states, actions], dim = -1) # inputs.shape = (batch, state_dim + act_dim)
    returns_to_go_pred = self.Q(inputs) # q_preds.shape = (batch, 1)
    q_loss = self.criterion(returns_to_go_pred, returns_to_go)
    # s_t -> pi(s_t)
    # s_t, pi(s_t) -> Q(s_t, pi(s_t))
    actions_pred = self.pi(states) # actions_pred.shape = (batch, act_dim)
    inputs = torch.cat([states, actions_pred], dim = -1) # inputs.shape = (batch, state_dim + act_dim)
    returns_to_go_best = self.Q(inputs) # returns_to_go_best.shape = (batch, 1)
    pi_loss = -torch.mean(returns_to_go_best)
    loss = q_loss + pi_loss
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.detach().cpu().item()
  def take_actions(self, state):
    self.eval()
    action = self.get_action(state)
    return action
  def save_net(self, save_path):
    if not exists(save_path):
      mkdir(save_path)
    file_path = join(save_path, 'dk.pt')
    torch.save(self.state_dict(), file_path)
  def save_jit(self, save_path):
    if not isdir(save_path):
      mkdir(save_path)
    jit_model = torch.jit.script(self.cpu())
    torch.jit.save(jit_model, f'{save_path}/dk_model.pth')
  def load_net(self, load_path = 'save_model/DKtest/dk.pt', device = 'cpu'):
    file_path = load_path
    self.load_state_dict(torch.load(file_path, map_location = device))
    print(f"Model loaded from {device}.")

if __name__ == "__main__":
  pass
