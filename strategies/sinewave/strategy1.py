'''
  Strategy 1:
    Data: Generated Sinewave
    DRL: DQN with a custom MLP architecture
    Observation Space: 
      price values,
      price -> rolling mean (10 data points),
      price -> rolling mean (50 data points),
      price -> rolliong mean (100 data points),
      price -> log difference
    Action Space: 
      Buy -> purchase 100% of the asset
      sell -> sell 100% of the asset
      hold -> do nothing
    Reward Strategy:
      Position Based Returns
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

from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

# Custom code
import sys
sys.path.append("..")

from tensortradeExtension.env.generic.components.renderer.bhsPositionChangeChart import PositionChangeChart
from tensortradeExtension.data.sine import SineWaveDataGenerator

# from stablebaseline3Extension.common.callbacks.RenderBSHCallback import RenderCallback
# from stablebaseline3Extension.common.callbacks.RenderHyperParameters import RenderHyperParameterCallback

# run configuration
window_size = 30
n_steps = 2000
evaluation_freq = n_steps / 10
model_path = "./models/data-sinewave-strategy1"
log_path = "./logs/data-sinewave-strategy1"
tensorboard_name = "data-sinewave-strategy1"

# create the data generator
data_generator = SineWaveDataGenerator(y_peaks=5)

def create_train_env(config):
  # get the training data
  dataframe = data_generator.train()
  # create price stream
  price_stream = Stream.source(list(dataframe['price']), dtype="float").rename("USD-TTC")
  # create exchange
  sinewavee_xchange = Exchange("sine-wave", service=execute_order, options=ExchangeOptions(commission=0.00075))(
      price_stream
  )
  # setup financial instruments
  USD = Instrument("USD", 8, "U.S. Dollar")
  TTC = Instrument("TTC", 8, "TensorTrade Coin")
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
      window_size=window_size,
      min_periods=30,
      max_allowed_loss=0.6,
      renderer=PositionChangeChart(),
  )
  return environment

def train():
  '''
    Train the RL agent for this strategy
  '''
  # Instantiate the environment
  # env = create_env(dataframe)
  # env.reset()

  # get the optimal batch size
  # batch_size = get_optimal_batch_size(window_size=window_size, n_steps=n_steps)

  # agent.learn(total_timesteps=n_steps, callback=[eval_callback, render_callback, hyperparameter_callback], tb_log_name=tensorboard_name)

  # agent.save(model_path)

  # Register the environment
  register_env("TradingEnv", create_train_env)

  # Create an RLlib Trainer instance to learn how to act in the above
  # environment.
  trainer = PPOTrainer(
    config={
    "env": "TradingEnv",
    "env_config": {},  # config to pass to env class
    }
  )

  for i in range(n_steps):
    results = trainer.train()
    reward = results['episode_reward_mean']
    # print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
    print(f"Iteration: {i}; episode_reward_mean: {reward}")

def evaluate():
  '''
    Evaluate the RL agent for this strategy
  '''
  # get the training data
  dataframe = data_generator.eval()
  # Instantiate the environment
  env = create_env(dataframe)
  env.reset()

  # load the model
  # agent = DQN.load(model_path+"/best_model")
  
  # Run until episode ends
  episode_reward = 0
  done = False
  obs = env.reset()

  while not done:
      action, _state = agent.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      episode_reward += reward

  env.render()
  print("episode_reward")
  print(episode_reward)

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