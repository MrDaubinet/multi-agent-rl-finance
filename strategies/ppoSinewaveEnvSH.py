'''
  Strategy 1:
    DRL: 
      PPO with a custom MLP architecture
    Environment: SinewaveSH
        Data: Generated Sinewave
          Training: 5 peaks
          Evaluation: 2 peaks
          Testing: 3 peaks
    Observation Space: 
      price values,
      price -> rolling mean (10 data points),
      price -> rolling mean (50 data points),
      price -> rolliong mean (100 data points),
      price -> log difference
    Action Space: 
      Enter Short -> short 100% of the asset into cash
      Exit Short -> Exit short, repurchase the asset, profit on cash difference
      hold -> do nothing
    Reward Strategy:
      Position Based Returns
'''

from strategies.strategy import Strategy

from rayExtension.callbacks.recordNetWorthCallback import RecordNetWorthCallback

from ray.rllib.agents import ppo

# Environments
from environments.sinewaveEnvSH import create_env

class PPOSinewaveEnvSH(Strategy):

    def __init__(self) -> None:
      # run configuration
      self.max_epoch = 50
      self.net_worth_threshold = 500
      self.patience = 1
      self.evaluation_frequency = 5
      self.log_name = "PPOSinewaveEnvSH"
      self.log_dir = f"/Users/jordandaubinet/Documents/Repositories/masters/masters-code/logs/{self.log_name}"

      # configure the train environment
      self.env_train_config = {
        "type": "train",
        "window_size": 50,
        "min_periods": 50,
        "max_allowed_loss": 1, # allow for 100% loss of funds
        "period": 10, # the number of periods to generate with the sine wave
      }
      # configure the test environment
      self.env_test_config = {
        "type": "test",
        "window_size": 50,
        "min_periods": 50,
        "render_env": True,
        "max_allowed_loss": 0.1, # allow for 10% loss of funds
        "num_workers": 1,
        "period": 10, # the number of periods to generate with the sine wave
      }
      # configure the test environment
      self.env_eval_config = {
        "type": "train",
        "window_size": 50,
        "min_periods": 50,
        "render_env": True,
        "max_allowed_loss": 1, # allow for 10% loss of funds
        "num_workers": 1,
        "period": 10, # the number of periods to generate with the sine wave
      }
      # Configure the algorithm.
      self.config = {
        "env": "TradingEnv",
        "env_config": self.env_train_config,  # config to pass to env class
        "evaluation_interval": self.evaluation_frequency,
        "evaluation_num_episodes": 1,
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "env_config": self.env_train_config,
            "render_env": True,
            "explore": False,
        },
        "num_workers": 1,
        "batch_mode": "complete_episodes",
        "callbacks": RecordNetWorthCallback,
      }
      ppo_config = ppo.DEFAULT_CONFIG.copy()
      ppo_config.update(self.config)
      self.config = ppo_config
      self.algorithm_name="PPO"
      self.create_env = create_env

      self.agent = ppo.PPOTrainer
      # TODO: send training metrics to tensorboard