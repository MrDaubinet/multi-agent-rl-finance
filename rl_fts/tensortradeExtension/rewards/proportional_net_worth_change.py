from tensortrade.env.default.rewards import TensorTradeRewardScheme
from tensortrade.env.generic import TradingEnv

from rl_fts.tensortradeExtension.actions.buy_sell_hold import BSH
from enum import Enum


class PriceDirection(Enum):
    UP = True
    DOWN = False


class PNWC(TensorTradeRewardScheme):
    """
       At every time step, reward the agent on the proportional networth change.
            preportion_net_worth_change = (net_worth - self.previous_net_worth) / self.previous_net_worth
    """

    def __init__(self, starting_value: float):
        self.net_worth_history = []
        self.starting_value = starting_value
        self.previous_net_worth = starting_value

    def reward(self, env: TradingEnv) -> float:
        return self.get_reward(env.action_scheme)

    def get_reward(self, action_scheme: BSH) -> float:
        """
        Parameters
        ----------
        action_scheme : `BSH | PBSH `
            The action scheme used for managing simple or proportion buy sell hold.
        Returns
            reward = (new_net_worth - old_net_worth) / old_net_worth
        -------
        float
            The difference in networth as profit / loss 
        """
        asset_balance = action_scheme.asset.balance.convert(
            action_scheme.exchange_pair)
        cash_balance = action_scheme.cash.balance
        net_worth = (asset_balance + cash_balance).as_float()
        if self.previous_net_worth != net_worth:
            preportion_net_worth_change = (net_worth - self.previous_net_worth) / self.previous_net_worth
            self.previous_net_worth = net_worth
        else:
            preportion_net_worth_change = 0
        self.net_worth_history.append(net_worth)
        return preportion_net_worth_change

    def reset(self) -> None:
        """Resets the history and previous net worth of the reward scheme."""
        self.net_worth_history = []
        self.previous_net_worth = self.starting_value

# Helper Functions
def get_reward_clipping(price_stream):
    biggest_gain = float('-inf')
    biggest_loss = float('inf')

    # at every time step
    for t in range(0, len(price_stream)-1):
        # Check the proportion difference in values
        proportional_price_difference = (
            price_stream[t+1] - price_stream[t]) / price_stream[t]
        # update biggest gain
        if proportional_price_difference > biggest_gain:
            biggest_gain = proportional_price_difference
        # update biggest loss
        if proportional_price_difference < biggest_loss:
            biggest_loss = proportional_price_difference
    return (biggest_loss, biggest_gain)

def get_max_episode_reward(data_stream):
    price_stream = data_stream
    episode_max_reward = 0
    for t in range(0, len(price_stream)-1):
        # if the price is going up
        if price_stream[t+1] > price_stream[t]:
            # get the reward
            proportional_price_difference = (price_stream[t+1] - price_stream[t]) / price_stream[t]
            # add the reward to the max
            episode_max_reward += proportional_price_difference
    return episode_max_reward
