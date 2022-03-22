'''
  Strategy 1:
    Data: Generated Sinewave
    DRL: DQN with a custom MLP architecture
    Action Space: 
      price values,
      price -> rolling mean (10 data points),
      price -> rolling mean (50 data points),
      price -> rolliong mean (100 data points),
      price -> log difference
'''

# tensortrade
from tensortrade.oms.instruments import Instrument
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import PBR
from tensortrade.env import default
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio

# stable baseline
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# other libraries
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# Custom code
import sys
sys.path.append("..")

from tensortradeExtension.env.generic.components.renderer.positionChangeChart import PositionChangeChart
from tensortradeExtension.data.sine import SineWaveDataGenerator


# run configuration
window_size = 30
n_steps = 10000
evaluation_freq = 100
model_path = "./strategy_1_model"
log_path = "./strategy_1_logs"

# Create the environement
def create_env():
  # extract data
  sine_wave_data_generator = SineWaveDataGenerator()
  # create dataframe
  dataframe = sine_wave_data_generator.fetch()
  # create price stream
  price_stream = Stream.source(list(dataframe['price']), dtype="float").rename("USD-TTC")
  # create exchange
  sinewavee_xchange = Exchange("sine-wave", service=execute_order, options=ExchangeOptions(commission=0.00075))(
      price_stream
  )
  # setup financial instruments
  USD = Instrument("USD", 2, "U.S. Dollar")
  TTC = Instrument("TTC", 2, "TensorTrade Coin")
  cash = Wallet(sinewavee_xchange, 100 * USD)
  asset = Wallet(sinewavee_xchange, 0 * TTC)

  # creat portfolio
  portfolio = Portfolio(USD, [
      cash,
      asset
  ])
  # create data feed
  feed = DataFeed([
      price_stream,
      price_stream.rolling(window=10).mean().rename("fast"),
      price_stream.rolling(window=50).mean().rename("medium"),
      price_stream.rolling(window=100).mean().rename("slow"),
      price_stream.log().diff().fillna(0).rename("lr")
  ])
  # set reward scheme
  reward_scheme = PBR(price=price_stream)
  # set action scheme
  action_scheme = BSH(
      cash=cash,
      asset=asset
  ).attach(reward_scheme)
  # create the render feed
  renderer_feed = DataFeed([
      Stream.source(dataframe["price"], dtype="float").rename("price"),
      Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
  ])
  # create the environment
  environment = default.create(
      feed=feed,
      portfolio=portfolio,
      action_scheme=action_scheme,
      reward_scheme=reward_scheme,
      renderer_feed=renderer_feed,
      renderer=PositionChangeChart(),
      window_size=window_size,
      max_allowed_loss=0.6
  )
  return environment

def train():
  '''
    Train the RL agent for this strategy
  '''

  # Instantiate the environment
  env = create_env()
  # wr_env = Monitor(env)
  env.reset()

  #environment details
  print("Action Space: "+str(env.action_space))
  print("State Space: "+str(env.observation_space.shape))
  print("Next observation")
  print(env.observer.feed.next())

  # get the optimal batch size
  batch_size = get_optimal_batch_size()

  # Create the agent
  agent = DQN(
    "MlpPolicy", 
    env=env, 
    batch_size=batch_size, 
    verbose=0, 
    exploration_initial_eps=0.9,
    target_update_interval=1000,
    buffer_size=1000,
    learning_starts=0
  )
  
  # set the evaluation model
  eval_callback = EvalCallback(
    eval_env=env,
    best_model_save_path=model_path,
    log_path=log_path,
    eval_freq=evaluation_freq,
    deterministic=True, 
    render=True
  )

  agent.learn(total_timesteps=n_steps, callback=eval_callback)

  agent.save(model_path)


def evaluate():
  '''
    Evaluate the RL agent for this strategy
  '''
  # Instantiate the environment
  env = create_env()
  env.reset()

  # load the model
  agent = DQN.load(model_path)
  
  # Run until episode ends
  episode_reward = 0
  done = False

  while not done:
      action, _state = agent.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      episode_reward += reward

  env.render()

# Helper Function
def get_optimal_batch_size(window_size=30, n_steps=1000, batch_factor=4, stride=1):
    """
    lookback = 30          # Days of past data (also named window_size).
    batch_factor = 4       # batch_size = (sample_size - lookback - stride) // batch_factor
    stride = 1             # Time series shift into the future.
    """
    lookback = window_size
    sample_size = n_steps
    batch_size = ((sample_size - lookback - stride) // batch_factor)
    return batch_size