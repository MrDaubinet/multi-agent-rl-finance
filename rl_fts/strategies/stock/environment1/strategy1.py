'''
	Strategy 1:
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
		DRL: 
			PPO with a ray MLP architecture
			horizon: Entire training set -> x days
'''
# base class
from rl_fts.strategies.strategy import Strategy
# Call backs
from rl_fts.rayExtension.callbacks.recordNetWorthCallback import RecordNetWorthCallback
# RL Agent
from ray.rllib.agents import ppo
# RL Model
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import ModelConfigDict
from rl_fts.strategies.stock.environment1.models.model1 import KerasBatchNormModel
# Environment
from rl_fts.environments.stock.environment1 import create_env, normalization_info, reward_clipping, max_episode_reward, max_net_worth

class PPO_NFLX_BSH_NWC(Strategy):

		def __init__(self):
				# run configuration
				self.max_epoch = 50
				horizon = 1260
				threshold_potential_max_worth = 0.8
				self.patience = 1
				self.evaluation_frequency = 1
				self.log_name = "stock/NFLX/environment1/strategy1"
				self.log_dir = "/Users/jordandaubinet/Documents/Repositories/masters/masters-code/logs/"

				# register model
				ModelCatalog.register_custom_model(
					"KerasBatchNormModel", KerasBatchNormModel)

				# configure the train environment
				env_train_config = {
					"type": "train",
					"window_size": 30,
					"min_periods": 30,
					"max_allowed_loss": 1,  # allow for 100% loss of funds
					"render_env": True,
					"log_name": self.log_name,
					"log_dir": self.log_dir,
					"horizon": horizon,
					"id": "NFLX_BSH_NWC",
					"starting_cash": 100
				}
				# Configure the algorithm.
				norm_info = normalization_info(env_train_config)
				# Model details
				model: ModelConfigDict = {
					"custom_model": "KerasBatchNormModel",
					"custom_model_config": {
					"mean": norm_info["mean"],
					"var": norm_info["var"]
					}
				}
				# Normalization
				reward_clipping_bounds = reward_clipping(env_train_config)
				print(f'reward_clipping_bounds: {reward_clipping_bounds}')
				episode_reward_bound = max_episode_reward(env_train_config)
				print(f'max possible episode reward: {episode_reward_bound}')
				# strategy should stop at this threshold
				self.net_worth_threshold = threshold_potential_max_worth*max_net_worth(env_train_config)
				# minimum
				print(f'minimum price value: {norm_info["min"]}')
				# maximum 
				print(f'maximum price value: {norm_info["max"]}')
				# mean
				print(f'mean price value: {norm_info["mean"][0]}')
				print(f'net worth threshold: {self.net_worth_threshold}')
				# Configure the algorithm.
				config = ppo.PPOConfig(
				).training(
						model=model,
						train_batch_size=1260
				).framework(
						framework="tf2",
						eager_tracing=False
				).environment(
						env="TradingEnv",
						env_config=env_train_config,
						render_env=True,
						normalize_actions=True,
						clip_rewards=reward_clipping_bounds
				).rollouts(
						num_rollout_workers=4,
						batch_mode="complete_episodes",
						horizon=horizon,
						soft_horizon=False
				).evaluation(
						evaluation_config=env_train_config,
						evaluation_interval=1,
						evaluation_duration_unit="timesteps",
						evaluation_duration=horizon,
						evaluation_num_workers=1,
						evaluation_parallel_to_training=True
				).callbacks(RecordNetWorthCallback)

				# number of times we update the NN with a single batch
				config.num_sgd_iter = 10
				# the size of the minibatch = the size of the batch
				config.sgd_minibatch_size = 126
				# the value function is the expected return from a current
				config.vf_clip_param = episode_reward_bound

				self.config = config.to_dict()
				self.algorithm_name = "PPO"
				self.create_env = create_env
				self.agent = ppo.PPOTrainer

def main():
		strategy = PPO_NFLX_BSH_NWC()
		strategy.clearLogs()
		strategy.train()

main()
