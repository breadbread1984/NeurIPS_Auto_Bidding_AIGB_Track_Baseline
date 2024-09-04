#!/usr/bin/python3

import torch
from torch import nn
from transformers import GPT2LMHeadModel

class DecisionGPT(nn.Module):
  def __init__(self, state_dim, act_dim, max_ep_len = 96):
    super(DecisionGPT, self).__init__()
    self.gpt2 = GPT2LMHeadModel.from_pretrained('openai-community/gpt2').transformer
    self.embed_timestep = nn.Embedding(max_ep_len, self.gpt2.config.n_ctx)
    self.input_dense = nn.Linear(state_dim + act_dim + 2, self.gpt2.config.n_ctx)
    self.predict_return = nn.Linear(self.gpt2.config.n_ctx, 1)
    self.predict_state = nn.Linear(self.gpt2.config.n_ctx, state_dim)
    self.predict_action = nn.Linear(self.gpt2.config.n_ctx, act_dim)
  def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask = None):
    # states.shape = (batch, hist_len, state_dim) also known as s_{t-1}
    # actions.shape = (batch, hist_len, act_dim) also known as a_{t-1}
    # returns_to_go.shape = (batch, hist_len, 1) also known as V(s_{t-1})
    # rewards.shape = (batch, hist_len, 1) also known as r_{t-1}
    # timesteps.shape = (batch, hist_len)
    if attention_mask is None:
      attention_mask = torch.ones((states.shape[0], states.shape[1]), dtype = torch.long)
    input_embeddings = torch.cat([states, actions, returns_to_go, rewards], dim = -1) # input_embeddings.shape = (batch, hist_len, state_dim + act_dim + 2)
    input_embeddings = self.input_dense(input_embeddings) # input_embeddings.shape = (batch, hist_len, n_ctx)
    time_embeddings = self.embed_timestep(timesteps) # time_embeddings.shape = (batch, hist_len, n_ctx)
    inputs_embeds = input_embeddings + time_embeddings # inputs_embeds.shape = (batch, hist_len, n_ctx)
    results = self.gpt2(inputs_embeds = inputs_embeds, attention_mask = attention_mask, return_dict = True)
    hidden_state = results.hidden_state # hidden_state.shape = (batch, hist_len, n_ctx)
    hidden_state = self.output_dense(hidden_state) # hidden_state.shape = (batch, hist_len, 5)
    return_preds = self.predict_return(hidden_state) # return_preds.shape = (batch, hist_len, 1)
    state_preds = self.predict_state(hidden_state) # state_preds.shape = (batch, hist_len, state_dim)
    action_preds = self.predict_action(hidden_state) # action_preds.shape = (batch, hist_len, act_dim)
    # return tensors:
    # s_t, a_t, V(s_t)
    return state_preds, action_preds, return_preds, None
  def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
    # states.shape = (batch, hist_len, state_dim)
    # actions.shape = (batch, hist_len, act_dim)
    # returns_to_go.shape = returns_to_go.reshape(1, hist_len, 1)
    # rewards.shape = (batch, hist_len, 1)
    # timesteps.shape = (batch, hist_len)
    _, action_preds, return_preds, reward_preds = self.forward(states, actions, rewards, returns_to_go, timesteps)
    return action_preds[0, -1] # (batch, act_dim)
  def step(self, states, actions, rewards, dones, returns_to_go, timesteps, attention_mask):
    # states.shape = (batch, hist_len, state_dim) also known as s_{t-1}
    rewards_target, actions_target, rtg_target = torch.clone(rewards), torch.clone(actiones), torch.clone(returns_to_go)
    state_preds, action_preds, return_preds, reward_preds = self.forward(states, actions, rewards, returns_to_go[:,:-1],  timesteps, attention_mask = attention_mask)
    
