'''
  Strategy 1:
    Data: Generated Sinewave
    DRL: PPO with a custom MLP architecture
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

# TODO: send training metrics to tensorboard
# TODO: load, evaluate and plot evaluation
# TODO: move the environment details to its own class / folder

# RAY Reinforcement Learning
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune import Analysis
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, TrialPlateauStopper

import time

# --- Custom Code --- #
# Environments
from environments.sinewaveEnvironment import create_env

# Tune CallBack
from rayExtension.callbacks.printCallback import PrintCallback
# Tune Stopper
from rayExtension.stoppers.netWorthstopper import NetWorthstopper

# Agent Callback
from rayExtension.callbacks.recordNetWorthCallback import RecordNetWorthCallback

# run configuration
MAX_EPOCH = 50
evaluation_freq = 50
NETWORTH_THRESHOLD = 500
log_name = "strategy1"
model_path = f"./models/sinewave/{log_name}"
log_dir = f"/Users/jordandaubinet/Documents/Repositories/masters/masters-code/logs/sinewave/{log_name}"

# configure the train environment
env_train_config = {
	"type": "train",
  "window_size": 50,
  "min_periods": 50,
  "max_allowed_loss": 1, # allow for 100% loss of funds
  "period": 10, # the number of periods to generate with the sine wave
}
# configure the test environment
env_test_config = {
	"type": "test",
  "window_size": 50,
  "min_periods": 50,
  "render_env": True,
  "max_allowed_loss": 0.1, # allow for 10% loss of funds
	"num_workers": 1,
  "period": 10, # the number of periods to generate with the sine wave
}
# configure the test environment
env_eval_config = {
	"type": "train",
  "window_size": 50,
  "min_periods": 50,
  "render_env": True,
  "max_allowed_loss": 1, # allow for 10% loss of funds
	"num_workers": 1,
  "period": 10, # the number of periods to generate with the sine wave
}
# Configure the algorithm.
config = {
  "env": "TradingEnv",
  "env_config": env_train_config,  # config to pass to env class
  "evaluation_interval": evaluation_freq,
  "evaluation_num_episodes": 1,
  "evaluation_num_workers": 1,
  "evaluation_config": {
      "env_config": env_train_config,
      "render_env": True,
      "explore": True
  },
  "callbacks": RecordNetWorthCallback,
  "num_workers": 4,
  "batch_mode": "complete_episodes",
}
ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update(config)

def train() -> Analysis:
  '''
    Train the RL agent for this strategy
  '''
  # Register the environment
  tune.register_env("TradingEnv", create_env)

  # dashboard
  ray.init(local_mode=True)
  
  start = time.time()

  # Setup stopping conditions
  stopper = CombinedStopper(
      MaximumIterationStopper(max_iter=MAX_EPOCH),
      NetWorthstopper(max_net_worth=NETWORTH_THRESHOLD, patience=3),
      TrialPlateauStopper(metric="net_worth_max")
  )

  # train an agent
  analysis = tune.run(
    "PPO",
    name=log_name,
    config=ppo_config,
    stop=stopper,
    metric="episode_reward_mean",
    mode="max",
    verbose=0,
    local_dir=log_dir,
    checkpoint_at_end=True,
    callbacks=[PrintCallback()]
  )
  print(f"Best Trail log directory: {analysis.best_logdir}")
  ray.shutdown()

  taken = time.time() - start
  print(f"Time taken: {taken:.2f} seconds.")
  return analysis

def evaluate(best_logdir = None):
  '''
    Evaluate the RL agent for this strategy
  '''
  # Register the environment
  tune.register_env("TradingEnv", create_env)

  # Restore agent
  agent = ppo.PPOTrainer(
    env="TradingEnv",
    config=ppo_config
  )
  agent.restore(best_logdir)
	# evaluate an episode
  # agent.evaluate()

  # Instantiate the environment
  env = create_env(env_test_config)
  
  # Run until episode ends
  episode_reward = 0
  done = False
  obs = env.reset()

  while not done:
      action = agent.compute_single_action(obs)
      obs, reward, done, _ = env.step(action)
      episode_reward += reward

  env.render()