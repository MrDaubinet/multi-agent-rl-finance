from strategies.ppoSinewaveEnvBSH import PPOSinewaveEnvBSH
from strategies.ppoSinewaveEnvSH import PPOSinewaveEnvSH

def main():
  # Strategy 1
  # strategy = PPOSinewaveEnvBSH()
  # strategy.train()
  # strategy.evaluate()
  # Strategy 2
  strategy_2 = PPOSinewaveEnvSH()
  strategy_2.train()
  strategy_2.evaluate()
  # TODO: networth is being tracked in info, lets check if we can print it (and plot it in tensorboard)
    # how is the info resource accessed by tune
main()