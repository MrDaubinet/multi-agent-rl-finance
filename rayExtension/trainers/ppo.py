# Import the RL algorithm (Trainer) we would like to use.
from ray.rllib.agents import ppo


EPOCH = 10
evaluation_freq = 1

# # Configure the algorithm.
# config = {
#     # Environment (RLlib understands openAI gym registered strings).
#     "env": "Taxi-v3",
#     # Use 2 environment workers (aka "rollout workers") that parallelly
#     # collect samples from their own environment clone(s).
#     "num_workers": 2,
#     # Change this to "framework: torch", if you are using PyTorch.
#     # Also, use "framework: tf2" for tf2.x eager execution.
#     "framework": "tf",
#     # Tweak the default model provided automatically by RLlib,
#     # given the environment's observation- and action spaces.
#     "model": {
#         "fcnet_hiddens": [64, 64],
#         "fcnet_activation": "relu",
#     },
#     # Set up a separate evaluation worker set for the
#     # `trainer.evaluate()` call after training (see below).
#     "evaluation_num_workers": 1,
#     # Only for evaluation runs, render the env.
#     "evaluation_config": {
#         "render_env": True,
#     },
# }

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
      "explore": False
  },
  # "evaluation_interval": 1,
	# "create_env_on_driver": True,
  # "num_envs_per_worker": 1,
  "num_workers": 2,
  "batch_mode": "complete_episodes",
  # Size of batches collected from each worker.(allows for parrallel sample collection) 
  # "rollout_fragment_length": 1028,
  # Number of timesteps collected for each SGD round. This defines the size
  # of each SGD epoch.
  # "train_batch_size": 2056,
  # minibatch size within each epoch.
  # "sgd_minibatch_size": 128,
  # Number of SGD iterations in each outer loop (i.e., number of epochs to
  # execute per train batch).
  # "num_sgd_iter": 30
}

ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update(config)

# Create our RLlib Trainer.
trainer = ppo.PPOTrainer(config=ppo_config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(EPOCH):
    print(trainer.train())

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
trainer.evaluate()