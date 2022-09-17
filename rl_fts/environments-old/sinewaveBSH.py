# tensortrade Environment
from tensortrade.oms.instruments import Instrument
from tensortrade.env.default.rewards import PBR
from tensortrade.env import default
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
import matplotlib.pyplot as plt

# --- Custom Code --- #
# Data
from rl_fts.data.sine import SineWaveDataGenerator
# Action Scheme
from rl_fts.tensortradeExtension.actions.buySellHold import BSH
# Renderer
from rl_fts.tensortradeExtension.renderer.bhsPositionChangeChart import PositionChangeChart

def generate_env(dataframe, config):
  # create price stream
  price_stream = Stream.source(list(dataframe['price']), dtype="float").rename("USD-TTC")
  # create exchange
  sinewavee_xchange = Exchange("sine-wave", service=execute_order, options=ExchangeOptions(commission=0.01))(
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
    price_stream.rolling(window=20).mean().rename("medium"),
    price_stream.rolling(window=30).mean().rename("slow"),
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
  renderer = default.renderers.EmptyRenderer()
  if ('render_env' in config and config["render_env"] == True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), clear=True)
    plt.tight_layout()
    fig.suptitle("Performance")
    renderer = PositionChangeChart(ax1, ax2)
  # create the environment
  environment = default.create(
    feed=feed,
    portfolio=portfolio,
    action_scheme=action_scheme,
    reward_scheme=reward_scheme,
    renderer_feed=renderer_feed,
    renderer=renderer,
    window_size=config["window_size"], # number of previous time series datapoints to use as the observation
    min_periods=config["min_periods"], # minimum number of time series datapoints used for warm up
    max_allowed_loss=config["max_allowed_loss"], # % of allowed loss on starting funds
  )
  return environment

# -- HELPER FUNCTIONS --#
def create_env(config):
  # create the data generator
  data_generator = SineWaveDataGenerator(period=config["period"], x_sample=config["trading_days"])
  if config["type"] == "train":
    dataframe = data_generator.train()
  elif config["type"] == "eval":
    dataframe = data_generator.validate()
  else:
    dataframe = data_generator.test()
  return generate_env(dataframe, config)