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
  def forward(self, states, actions, rewards, returns_to_go, timesteps):
    # states.shape = (batch, hist_len, state_dim)
    # actions.shape = (batch, hist_len, 1)
    # returns_to_go.shape = (batch, hist_len, 1)
    # rewards.shape = (batch, hist_len, 1)
    # timesteps.shape = (batch, hist_len, 1)
    input_embeddings = torch.cat([states, actions, returns_to_go, rewards], dim = -1) # input_embeddings.shape = (batch, hist_len, state_dim + act_dim + 2)
    input_embeddings = self.input_dense(input_embeddings) # input_embeddings.shape = (batch, hist_len, n_ctx)
    time_embeddings = self.embed_timestep(timesteps) # time_embeddings.shape = (batch, hist_len, n_ctx)
    inputs_embeds = input_embeddings + time_embeddings # inputs_embeds.shape = (batch, hist_len, n_ctx)
    results = self.gpt2(inputs_embeds = inputs_embeds, return_dict = True)
    hidden_state = results.hidden_state # hidden_state.shape = (batch, hist_len, n_ctx)
    hidden_state = self.output_dense(hidden_state) # hidden_state.shape = (batch, hist_len, 5)
    return_preds = self.predict_return(hidden_state) # return_preds.shape = (batch, hist_len, 1)
    state_preds = self.predict_state(hidden_state) # state_preds.shape = (batch, hist_len, state_dim)
    action_preds = self.predict_action(hidden_state) # action_preds.shape = (batch, hist_len, act_dim)

