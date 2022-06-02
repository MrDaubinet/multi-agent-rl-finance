from strategies.sinewave import strategy1
from environments.sinewaveEnvironment import maximum_reward
from data.sine import SineWaveDataGenerator
import matplotlib.pyplot as plt

def print_reward(name, data_frame, funds):
  max_reward = maximum_reward(data_frame, funds)
  print(f'{name}, maximum reward -> {max_reward}')

def plot(df, title):
  df.plot(title=title)

def print_env_rewards(funds):
  data_gen = SineWaveDataGenerator()
  # generate dataframes
  train_data = data_gen.train()
  validate_data = data_gen.validate()
  test_data = data_gen.test()
  # print rewards
  print_reward("Training Data", train_data, funds)
  print_reward("Validation Data", validate_data, funds)
  print_reward("Test Data", test_data, funds)

def plot_envs():
  data_gen = SineWaveDataGenerator()
  # generate dataframes
  train_data = data_gen.train()
  validate_data = data_gen.validate()
  test_data = data_gen.test()
  # plot price data
  plot(train_data, title="Training Data")
  plot(validate_data, title="Validation Data")
  plot(test_data, title="Testing Data")
  plt.show()

def main():
  plt.close('all')
  print_env_rewards(100)
  plot_envs()
  analysis = strategy1.train()
  # strategy1.evaluate(analysis.best_trial.checkpoint.value)
  # strategy1.evaluate('/Users/jordandaubinet/Documents/Repositories/masters/masters-code/logs/sinewave/strategy1/strategy1/PPOTrainer_TradingEnv_7ab32_00000_0_2022-05-27_11-22-07/checkpoint_000020/checkpoint-20')

  # TODO: create custom callback to render env while training

  # TODO: Update state space to include balance,
  # TODO: Update action to buy, hold, sell, amounts
  # TODO: Update reward

main()