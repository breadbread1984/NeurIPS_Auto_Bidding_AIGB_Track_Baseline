#!/usr/bin/python3

from os import mkdir
from os.path import join, exists, isdir
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def b_spline(x, grid, k = 0, extend = True, device = torch.device('cuda')):
  def extend_grid(grid, k_extend=0):
    # pad k to left and right
    # grid shape: (batch, grid)
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
    for i in range(k_extend):
      grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
      grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
    grid = grid.to(device)
    return grid
  if extend == True:
     grid = extend_grid(grid, k_extend=k)
  grid = grid.unsqueeze(dim = 2).to(device)
  x = x.unsqueeze(dim = 1).to(device)
  if k == 0:
    value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
  else:
    B_km1 = b_spline(x[:, 0], grid = grid[:, :, 0], k = k - 1, extend = False, device = device)
    value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
  return value

def coef2curve(x_eval, grid, coef, k):
  # get curve(x) values of given x values, curve is defined by coef
  # x_eval.shape = (spline number, sample number)
  y_eval = torch.einsum('ij,ijk->ik', coef, b_spline(x_eval, grid, k, device = x_eval.device))
  # y_eval.shape = (spline number, sample number)
  return y_eval

def curve2coef(x_eval, y_eval, grid, k):
  # estimate coef of curve(x) by giving (x,y) samples
  # x_eval.shape = (spline number, sample number)
  mat = b_spline(x_eval, grid, k, device = x_eval.device).permute(0, 2, 1)
  coef = torch.linalg.lstsq(mat, torch.unsqueeze(y_eval, dim = 2), driver = 'gels' if mat.is_cuda else 'gelsy').solution[:,:,0]
  # coef.shape = (spline number, grid + 1)
  return coef

class KANLayer(nn.Module):
  def __init__(self, channel_in, channel_out, grid = 5, k = 3, base_fun = 'silu', scale_base = 1.0, scale_sp = 1.0, base_trainable = True, sp_trainable = True, grid_eps = 0.02):
    # grid: the number of grid intervals
    # k: the order of piecewise polynomial
    super(KANLayer, self).__init__()
    self.channel_in = channel_in
    self.channel_out = channel_out
    self.grid = grid
    self.k = k
    self.base_fun = {'silu': nn.SiLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}[base_fun]
    self.grid_eps = grid_eps
    # 1) initialize coefficients from random curve
    # grid_x is updated in forward pass
    # coef is updated in both forward & backward pass
    # scale_base is updated in backward pass
    # scale_sp is updated in backward pass
    self.grid_x = nn.Parameter(torch.tile(torch.unsqueeze(torch.linspace(start = -1, end = 1, steps = self.grid + 1), dim = 0), (channel_in * channel_out, 1)), requires_grad = False) # samples.shape = (channel_in * channel_out, grid + 1)
    y = (torch.rand(channel_in * channel_out, self.grid_x.shape[1]) - 1 / 2) * 0.1 / self.grid # y.shape = (channel_in * channel_out, grid + 1)
    self.coef = nn.Parameter(curve2coef(self.grid_x, y, self.grid_x, k), requires_grad = True) # coef.shape = (channel_in * channel_out, grid + 1)
    self.scale_base = nn.Parameter(torch.ones(1, channel_in * channel_out) * scale_base, requires_grad = base_trainable) # scale_base.shape = (1, channel_in * channel_out)
    self.scale_sp = nn.Parameter(torch.ones(1, channel_in * channel_out) * scale_sp, requires_grad = sp_trainable) # scale_sp.shape = (channel_in * channel_out)
  def forward(self, x, do_train = False):
    # NOTE: x.shape = (batch, channel_in)
    # every channel is processed by a single activation function (spline)
    batch  = x.shape[0]
    tiled_x = torch.einsum('ij,k->ikj', x, torch.ones(self.channel_out,).to(x.device)).reshape(batch, self.channel_out * self.channel_in) # tiled_x.shape = (batch, channel_out * channel_in)
    if do_train == True:
      # NOTE: update grid_x to just cover the value range of the samples
      # sort x for every spline ascendly
      x_pos = torch.transpose(torch.sort(tiled_x, dim = 0)[0], 0, 1) # x_pos.shape = (spline number = channel_out * channel_in, sample number = batch)
      y_eval = coef2curve(x_pos, self.grid_x, self.coef, self.k) # y_eval.shape = (spline number = channel_out * channel_in, sample number = batch)
      # grid_adaptive: a grid which is uniformly sampled by index (heavy weight)
      # grid_uniform: a grid which is uniformly sampled by value (light weight)
      grid_adaptive = x_pos[:, torch.linspace(start = 0, end = batch - 1, steps = self.grid + 1, dtype = torch.int32)] # grid_adaptive.shape = (channel_out * channel_in, grid + 1)
      grid_uniform = torch.cat([grid_adaptive[:, [0]] - 0.01 + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * 0.01) * a for a in torch.linspace(0, 1, self.grid + 1)], dim = 1) # grid_uniform.shape = (channel_out * channel_in, grid + 1)
      self.grid_x.data = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive # grid_x.shape = (channel_out * channel_in, grid + 1)
      self.coef.data = curve2coef(x_pos, y_eval, self.grid_x, self.k) # coef.shape = (channel_in * channel_out, grid + 1)
    preacts = torch.reshape(tiled_x, (batch, self.channel_out, self.channel_in)) # preacts.shape = (batch, channel_out, channel_in)
    base = self.base_fun(tiled_x) # tiled_x.shape = (batch, channel_out * channel_in)
    y = coef2curve(x_eval = torch.transpose(tiled_x, 0, 1), grid = self.grid_x, coef = self.coef, k = self.k) # y.shape = (channel_out * channel_in, batch)
    y = torch.transpose(y, 0, 1) # y.shape = (batch, channel_out * channel_in)
    # postspline is curve(x)
    # postacts is a merge of curve(x) and base_fun(x), merge weight is learnable
    # y is KAN layer output
    postspline = torch.reshape(y, (batch, self.channel_out, self.channel_in)) # postspline.shape = (batch, channel_out, channel_in)
    postacts = torch.reshape(self.scale_base * base + self.scale_sp * y, (batch, self.channel_out, self.channel_in)) # y.shape = (batch, channel_out, channel_in)
    y = torch.sum(postacts, dim = 2) # y.shape = (batch, channel_out)
    return y, preacts, postacts, postspline

class KAN(nn.Module):
  def __init__(self, channels = None, grid = 5, k = 3, base_fun = 'silu', grid_eps = 0.02, lamb_l1 = 1., lamb_entropy = 2., lamb_coef = 0.1, lamb_coefdiff = 0.1):
    super(KAN, self).__init__()
    self.lamb_l1 = lamb_l1
    self.lamb_entropy = lamb_entropy
    self.lamb_coef = lamb_coef
    self.lamb_coefdiff = lamb_coefdiff
    acts = list()
    biases = list()
    for i in range(len(channels) - 1):
      cin, cout = channels[i], channels[i + 1]
      scale_base = 1 / torch.sqrt(torch.Tensor([cin])) + (torch.randn(cin * cout,) * 2 - 1) * 0.1
      acts.append(KANLayer(channel_in = cin, channel_out = cout, grid = grid, k = k, base_fun = base_fun, scale_base = scale_base, scale_sp = 1.0, base_trainable = True, sp_trainable = True, grid_eps = grid_eps))
      biases.append(nn.Parameter(torch.zeros((1, cout)), requires_grad = True)) # bias.shape = (1, cout)
    self.acts = nn.ModuleList(acts)
    self.biases = nn.ParameterList(biases)
  def forward(self, x, do_train = False):
    res = x
    acts_scale = list()
    for act, bias in zip(self.acts,self.biases):
      res, preacts, postacts, postspline = act(res, do_train)
      res = res + bias
      # activation scale
      grid_reshape = act.grid_x.reshape(act.channel_out,act.channel_in,-1) # grid_reshape.shape = (channel_out,channel_in,grid+1)
      input_range = grid_reshape[:,:,-1] - grid_reshape[:,:,0] + 1e-4 # input value range, shape = (channel_out, channel_in)
      output_range = torch.mean(torch.abs(postacts), dim = 0) # output absolute value range, shape = (batch, channel_out, channel_in)
      acts_scale.append(output_range / input_range) # acts_scale.shape = (batch, channel_out, channel_in)
    # get regularizer
    regularizer = 0
    for scale in acts_scale:
      vec = scale.reshape(-1,) # vec.shape = (batch * channel_out * channel_in)
      p = vec / torch.sum(vec) # p.shape = (batch * channel_out * channel_in)
      nonlinear = lambda x,th,factor: (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)
      l1 = torch.sum(nonlinear(vec, 1e-16,1.))
      entropy = - torch.sum(p * torch.log2(p + 1e-4))
      regularizer += self.lamb_l1 * l1 + self.lamb_entropy * entropy
    for act in self.acts:
      coeff_l1 = torch.sum(torch.mean(torch.abs(act.coef), dim = 1))
      coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(act.coef)), dim = 1))
      regularizer += self.lamb_coef * coeff_l1 + self.lamb_coefdiff * coeff_diff_l1
    return res, regularizer

class DecisionKAN(nn.Module):
  def __init__(self, state_dim, act_dim, gamma = 0.8, lr = 1e-4, weight_decay = 1e-4):
    super(DecisionKAN, self).__init__()
    self.gamma = gamma
    self.Q = KAN(channels = [state_dim + act_dim, 8, 4, 1], grid = 7, k = 3)
    self.pi = KAN(channels = [state_dim, 8, 4, act_dim], grid = 7, k = 3)
    self.criterion = nn.MSELoss()
    self.optimizer = AdamW(self.parameters(), lr = lr, weight_decay = weight_decay)
    self.scheduler = LambdaLR(self.optimizer, lambda steps: min((steps + 1) / self.warmup_steps, 1))
  def forward(self, states):
    actions_next, regularizer = self.pi(states) # actions_next.shape = (batch, 1)
    return actions_next
  def get_action(self, states):
    actions_next = self.forward(states)
    return actions_next
  def step(self, states, next_states, rewards, actions, returns_to_go, dones):
    # s_t, a_t -> Q(s_t, a_t)
    inputs = torch.cat([states, actions], dim = -1) # inputs.shape = (batch, state_dim + act_dim)
    returns_to_go_pred, regularizer1 = self.Q(inputs) # q_preds.shape = (batch, 1)
    q_loss = criterion(returns_to_go_pred, returns_to_go)
    # s_t -> pi(s_t)
    # s_t, pi(s_t) -> Q(s_t, pi(s_t))
    actions_pred, regularizer2 = self.pi(states) # actions_pred.shape = (batch, act_dim)
    inputs = torch.cat([states, actions_pred], dim = -1) # inputs.shape = (batch, state_dim + act_dim)
    returns_to_go_best, regularizer3 = self.Q(inputs) # returns_to_go_best.shape = (batch, 1)
    pi_loss = -torch.mean(returns_to_go_best)
    loss = q_loss + pi_loss + regularizer1 + regularizer1 + regularizer2 + regularizer3
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.parameters(), .25)
    self.optimizer.step()
    return loss.detach().cpu().item()
  def take_actions(self, state):
    self.eval()
    action = self.get_action(state)
    action = action.detach().cpu().numpy()
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
  def load_net(self, load_path = 'saved_model/DKtest', device = 'cpu'):
    file_path = load_path
    self.load_state_dict(torch.load(file_path, map_location = device))
    print(f"Model loaded from {self.device}.")

if __name__ == "__main__":
  layer = KANLayer(3,2)
  inputs = torch.randn(2048, 3)
  y, preacts, postacts, postspline = layer(inputs)
  print(y.shape, preacts.shape, postacts.shape, postspline.shape)
  kan = KAN(channels = (1331, 4, 1))
  inputs = torch.randn(2048, 1331)
  y = kan(inputs, do_train = False)
  print(y.shape)
  torch.save(kan, 'kan.torch')
  print(kan.parameters())
