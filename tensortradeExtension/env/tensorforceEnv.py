from typing import Union
import tensortrade.env.default as default
from tensortrade.env.generic import TradingEnv
from tensortrade.feed.core import DataFeed
from tensortrade.oms.wallets import Portfolio

def create(
  portfolio: 'Portfolio',
  action_scheme: 'Union[actions.TensorTradeActionScheme, str]',
  reward_scheme: 'Union[rewards.TensorTradeRewardScheme, str]',
  feed: 'DataFeed',
  window_size: int = 1,
  min_periods: int = None,
  max_episode_timesteps: int = None,
  **kwargs) -> TradingEnv:
  """Creates the default `TradingEnv` of the project to be used in training
  RL agents.

  Parameters
  ----------
  portfolio : `Portfolio`
      The portfolio to be used by the environment.
  action_scheme : `actions.TensorTradeActionScheme` or str
      The action scheme for computing actions at every step of an episode.
  reward_scheme : `rewards.TensorTradeRewardScheme` or str
      The reward scheme for computing rewards at every step of an episode.
  feed : `DataFeed`
      The feed for generating observations to be used in the look back
      window.
  window_size : int
      The size of the look back window to use for the observation space.
  min_periods : int, optional
      The minimum number of steps to warm up the `feed`.
  **kwargs : keyword arguments
      Extra keyword arguments needed to build the environment.

  Returns
  -------
  `TradingEnv`
      The default trading environment.
  """
  env = default.create(portfolio=portfolio,
    action_scheme=action_scheme,
    reward_scheme=reward_scheme,
    feed=feed,
    window_size=window_size,
    min_periods=min_periods,
    **kwargs)
  
  env.max_episode_timesteps = lambda: max_episode_timesteps
  return env 