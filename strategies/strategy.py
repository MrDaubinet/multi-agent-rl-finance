"""
  A strategy class combines a DRL algorithm and an environment with a specific, 
    action, reward and observation space
"""
import time

import ray
from ray import tune
from ray.tune import Analysis
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, TrialPlateauStopper

# Tune CallBack
from rayExtension.callbacks.printCallback import PrintCallback
from rayExtension.callbacks.renderCallback import RenderCallback
# Tune Stopper
from rayExtension.stoppers.netWorthstopper import NetWorthstopper

class Strategy():

  def __init__(self) -> None:
    self.max_epoch = None
    self.evaluation_frequency = None
    self.net_worth_threshold = None
    self.patience = 0
    
    self.env_test_config = None
    self.agent = None
    raise NotImplementedError

  def train(self) -> Analysis:
    '''
      Train the RL agent for this strategy
    '''
    # Register the environment
    tune.register_env("TradingEnv", self.create_env)

    # dashboard
    ray.init(local_mode=True)
    
    start = time.time()

    # Setup stopping conditions
    stopper = CombinedStopper(
        MaximumIterationStopper(max_iter=self.max_epoch),
        NetWorthstopper(max_net_worth=self.net_worth_threshold, patience=self.patience),
        TrialPlateauStopper(metric="net_worth_max")
    )

    # train an agent
    analysis = tune.run(
      self.algorithm_name,
      name=self.log_name,
      config=self.config,
      stop=stopper,
      metric="episode_reward_mean",
      mode="max",
      verbose=0,
      local_dir=self.log_name,
      checkpoint_at_end=True,
      callbacks=[PrintCallback(), RenderCallback(self.evaluation_frequency)]
    )
    print(f"Best Trail log directory: {analysis.best_logdir}")
    ray.shutdown()

    taken = time.time() - start
    print(f"Time taken: {taken:.2f} seconds.")

    self.best_logdir = analysis.best_trial.checkpoint.value
    return analysis

  def evaluate(self, best_logdir = None):
    '''
      Evaluate the RL agent for this strategy
    '''
    # Register the environment
    tune.register_env("TradingEnv", self.create_env)

    if not best_logdir:
      best_logdir = self.best_logdir

    # Restore agent
    self.agent(
      env="TradingEnv",
      config=self.config
    )
    self.agent.restore(best_logdir)
    # evaluate an episode
    # agent.evaluate()

    # Instantiate the environment
    env = self.create_env(self.env_test_config)
    
    # Run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()

    while not done:
        action = self.agent.compute_single_action(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

    env.render()