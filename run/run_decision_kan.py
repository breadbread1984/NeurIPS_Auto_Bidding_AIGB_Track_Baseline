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
    replay_buffer = ConcatDataset(datasets)
    logger.info(f"Replay buffer size: {len(replay_buffer)}")
    model = DecisionKAN(state_dim = 16, act_dim = 1, lr = 1e-2).to("cuda")
    if exists(join('save_model','DKtest','dk.pt')):
      model.load_net(join('save_model', 'DKtest', 'dk.pt'), "cuda")
    dataloader = DataLoader(replay_buffer, batch_size = 4550, shuffle = True, num_workers = 256)
    model.train()
    for epoch in range(100):
      i = 0
      for step, triplet in enumerate(dataloader):
          states, actions, returns_to_go = triplet['states'].to('cuda'), triplet['actions'].to('cuda'), triplet['returns_to_go'].to('cuda')
          train_loss = model.step(states, actions, returns_to_go, epoch * len(dataloader) + step)
          i += 1
          logger.info(f"Epoch: {epoch} Step: {i} Action loss: {np.mean(train_loss)}")
          if step % 1000 == 0: model.save_net("save_model/DKtest")
      model.scheduler.step()
      model.save_net("save_model/DKtest")

def load_model():
    model = DecisionKAN(state_dim = 16, act_dim = 1).to("cuda")
    model.load_net("save_model/DKtest")

if __name__ == "__main__":
    run_dk()
