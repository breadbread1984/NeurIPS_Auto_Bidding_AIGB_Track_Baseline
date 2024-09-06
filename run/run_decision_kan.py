#!/usr/bin/python3

from os import listdir
from os.path import join, exists
from math import ceil
import numpy as np
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dk.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.dk.dk import DecisionKAN
from torch.utils.data import DataLoader, ConcatDataset
import logging
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_dk():
    train_model()

def train_model():
    datasets = list()
    for f in listdir(join('data', 'trajectory')):
        csv_path = join('data', 'trajectory', f)
        replay_buffer = EpisodeReplayBuffer(csv_path)
        datasets.append(replay_buffer)
    replay_buffer = ConcateDataset(datasets)
    logger.info(f"Replay buffer size: {len(replay_buffer)}")
    model = DecisionKAN(state_dim = 16, act_dim = 1).to("cuda")
    dataloader = DataLoader(replay_buffer, batch_size = 32, shuffle = True, num_workers = 32)
    model.train()
    for epoch in range(100):
      i = 0
      for triplet in dataloader:
          states, next_states, rewards, actions, returns_to_go, dones = \
                triplet['states'].to('cuda'), triplet['next_states'].to('cuda'), triplet['rewards'].to('cuda'), \
                triplet['actions'].to('cuda'), triplet['returns_to_go'].to('cuda'), triplet['dones'].to('cuda')
          train_loss = model.step(states, next_states, rewards, actions, returns_to_go, dones)
          i += 1
          logger.info(f"Epoch: {epoch} Step: {i} Action loss: {np.mean(train_loss)}")
          model.scheduler.step()
      model.save_net("save_model/DKtest")

def load_model():
    model = DecisionKAN(state_dim = 16, act_dim = 1).to("cuda")
    model.load_net("Model/DK/saved_model")

if __name__ == "__main__":
    run_dk()
