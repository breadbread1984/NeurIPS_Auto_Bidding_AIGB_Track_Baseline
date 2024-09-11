import time
import gin
import numpy as np
import os
from os.path import realpath, dirname, join
import psutil
# from saved_model.DTtest.dt import DecisionKAN
from bidding_train_env.baseline.dk.dk import DecisionKAN
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
import torch
import pickle

class DkBiddingStrategy(BaseBiddingStrategy):
  def __init__(self, budget=100, name="Decision-Transformer-PlayerStrategy", cpa=2, category=1):
    super(DkBiddingStrategy, self).__init__(budget, name, cpa, category)
    file_name = dirname(realpath(__file__))
    dir_name = dirname(file_name)
    dir_name = dirname(dir_name)
    model_path = join(dir_name, "save_model", "DKtest", "dk.pt")
    self.model = DecisionKAN(state_dim = 16, act_dim = 1)
    self.model.load_net(model_path)
  def reset(self):
    self.remaining_budge = self.budget
  def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
              historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
    time_left = (48 - timeStepIndex) / 48
    budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
    history_xi = [result[:, 0] for result in historyAuctionResult]
    history_pValue = [result[:, 0] for result in historyPValueInfo]
    history_conversion = [result[:, 1] for result in historyImpressionResult]

    historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0

    historical_conversion_mean = np.mean(
        [np.mean(reward) for reward in history_conversion]) if history_conversion else 0

    historical_LeastWinningCost_mean = np.mean(
        [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0

    historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0

    historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

    def mean_of_last_n_elements(history, n):
        last_three_data = history[max(0, n - 3):n]
        if len(last_three_data) == 0:
            return 0
        else:
            return np.mean([np.mean(data) for data in last_three_data])

    last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
    last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
    last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
    last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
    last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

    current_pValues_mean = np.mean(pValues)
    current_pv_num = len(pValues)

    historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
    last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
    last_three_pv_num_total = sum(
        [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

    test_state = np.array([
        time_left, budget_left, historical_bid_mean, last_three_bid_mean,
        historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
        historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
        last_three_conversion_mean, last_three_xi_mean,
        current_pValues_mean, current_pv_num, last_three_pv_num_total,
        historical_pv_num_total
    ])

    bids = self.model.take_actions(test_state)
    return bids
