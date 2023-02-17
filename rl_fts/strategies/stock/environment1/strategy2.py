'''
	Strategy 1:
		DRL: 
			PPO with a custom MLP model 
		Environment: Stock - NFLX
			Data: 10 years (2012-12-31 - 2022-12-31)
				Training: 5 years
				Evaluation: 3 years
				Testing: 2 years
			Observation Space: 
				price values,
				price -> rolling mean (10 data points),
				price -> rolling mean (20 data points),
				price -> rolliong mean (30 data points),
				price -> log difference
		Action Space: buy-sell-hold
		Reward Strategy: net-worth-change
		horizon: 30 days
'''
# base class
from rl_fts.strategies.strategy import Strategy
# Call backs
from rl_fts.rayExtension.callbacks.recordNetWorthCallback import RecordNetWorthCallback
# RL Agent
from ray.rllib.agents import ppo
from ray.rllib.utils.typing import ModelConfigDict
# Model
from ray.rllib.models import ModelCatalog
# from rl_fts.strategies.sinewave
# Environment
from rl_fts.environments.stock.environment1 import create_env

ModelCatalog.register_custom_model("my_tf_model", MyModelClass)

class PPO_NFLX_BSH_NWC(Strategy):

		def __init__(self):
				# run configuration
				self.max_epoch = 50
				self.net_worth_threshold = 250
				self.patience = 1
				self.evaluation_frequency = 1
				self.log_name = "stock/NFLX/strategy1"
				self.log_dir = "/Users/jordandaubinet/Documents/Repositories/masters/masters-code/logs/"

				# configure the train environment
				env_train_config = {
						"type": "train",
						"window_size": 30,
						"min_periods": 30,
						"max_allowed_loss": 1,  # allow for 100% loss of funds
						"render_env": True,
						# "render_env": False,
						"log_name": self.log_name,
						"log_dir": self.log_dir,
						"horizon": 30,
						"id": "NFLX_BSH_NWC",
				}
				# configure the test environment
				self.env_test_config = {
						"type": "eval",
						"window_size": 30,
						"min_periods": 30,
						"render_env": True,
						"max_allowed_loss": 1,  # allow for 100% loss of funds
						"num_workers": 1,
				}
				# Model details
				model: ModelConfigDict = {
						"_disable_preprocessor_api": False,
						"fcnet_hiddens": [256, 256],
						"fcnet_activation": "relu",
				}
				# variables for configuration
				num_rollout_workers = 1
				# this could probably be made something random (more memory specific) over 30
				sgd_minibatch_size = 30
				# this is a fairly poor way of setting the batch size
				train_batch_size = num_rollout_workers * sgd_minibatch_size
				# train_batch_size = 30
				# Configure the algorithm.
				config = ppo.PPOConfig(
				).training(
						model=model,
						# train_batch_size=500
						train_batch_size=train_batch_size
				).framework(
						framework="tf2",
						eager_tracing=True
				).environment(
						env="TradingEnv",
						env_config=env_train_config,
						render_env=True,
						normalize_actions=True
				).rollouts(
						num_rollout_workers=num_rollout_workers,
						batch_mode="complete_episodes",
						horizon=30,
						soft_horizon=False
				).evaluation(
						evaluation_config=self.env_test_config,
						evaluation_interval=1,
						evaluation_duration_unit="timesteps",
						evaluation_duration=30,
						evaluation_num_workers=1,
						evaluation_parallel_to_training=False
				).callbacks(RecordNetWorthCallback)

				config.rollout_fragment_length = 30
				config.sgd_minibatch_size = sgd_minibatch_size

				self.config = config.to_dict()
				self.algorithm_name = "PPO"
				self.create_env = create_env
				self.agent = ppo.PPOTrainer

def main():
		strategy = PPO_NFLX_BSH_NWC()
		strategy.clearLogs()
		strategy.train()


main()
