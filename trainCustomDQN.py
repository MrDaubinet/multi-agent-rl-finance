from tensortrade.oms.instruments import Instrument
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import PBR
from tensortrade.env import default

from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio

from tensortradeExtension.env.generic.components.renderer.positionChangeChart import PositionChangeChart

from tensortrade.agents import DQNAgent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup Instruments
USD = Instrument("USD", 2, "U.S. Dollar")
TTC = Instrument("TTC", 8, "TensorTrade Coin")


n_steps = 10
n_episodes = 10
window_size = 30
memory_capacity = n_steps * 10
save_path = "models/tests/sinewave/dqn"

# Generate the data stream
def generate_data_stream():
  # generate x values over 1001
  x = np.arange(0, 2*np.pi, 2*np.pi / 1001)
  # generate y values from x values
  y = 50*np.sin(3*x) + 100
  # reset x values over 1000
  x = np.arange(0, 2*np.pi, 2*np.pi / 1000)

  price_stream = Stream.source(y, dtype="float").rename("USD-TTC")

  return [price_stream, y]

# Create the environement
def create_env():
    price_stream, y = generate_data_stream()
    sine_wave_exchange = Exchange("sinewave", service=execute_order)(
        price_stream
    )
    cash = Wallet(sine_wave_exchange, 100000 * USD)
    asset = Wallet(sine_wave_exchange, 0 * TTC)

    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    feed = DataFeed([
        price_stream,
        price_stream.rolling(window=10).mean().rename("fast"),
        price_stream.rolling(window=50).mean().rename("medium"),
        price_stream.rolling(window=100).mean().rename("slow"),
        price_stream.log().diff().fillna(0).rename("lr")
    ])

    action_scheme = BSH(
        cash=cash,
        asset=asset
    )

    reward_scheme = PBR(price=price_stream)

    renderer_feed = DataFeed([
        Stream.source(y, dtype="float").rename("price"),
        Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
    ])

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

# calculate the batch size
def get_optimal_batch_size(window_size=window_size, n_steps=1000, batch_factor=4, stride=1):
    """
    lookback = 30          # Days of past data (also named window_size).
    batch_factor = 4       # batch_size = (sample_size - lookback - stride) // batch_factor
    stride = 1             # Time series shift into the future.
    """
    lookback = window_size
    sample_size = n_steps
    batch_size = ((sample_size - lookback - stride) // batch_factor)
    return batch_size

generate_data_stream()

env = create_env()


batch_size = get_optimal_batch_size()

agent = DQNAgent(env)

agent.train(batch_size=batch_size, 
            n_steps=n_steps, 
            n_episodes=n_episodes, 
            memory_capacity=memory_capacity, 
            save_path=save_path)

# Run until episode ends
episode_reward = 0
done = False
obs = env.reset()

while not done:
    action = agent.get_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward

env.render()