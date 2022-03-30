import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union
import gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.logger import Figure

import sys
sys.path.append("..")

from tensortradeExtension.env.generic.components.renderer.positionChangeChart import PositionChangeChart

class RenderCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param env: (Union[gym.Env, VecEnv]) environment object
    """
    def __init__(self, 
      env: Union[gym.Env, VecEnv],
      verbose=0,
      ):
        super(RenderCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        """
        # self.position_change_chart.render(self.env)

        # Run until episode ends
        episode_reward = 0
        done = False
        obs = self.env.reset()

        while not done:
            action, _state = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            episode_reward += reward
        
        self.logGraph()
        self.env.render()
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
    

    # ---- Helper Functions ---- #
    def logGraph(self):
        history = pd.DataFrame(self.env.observer.renderer_history)

        actions = list(history.action)
        p = list(history.price)

        buy = {}
        sell = {}

        for i in range(len(actions) - 1):
            a1 = actions[i]
            a2 = actions[i + 1]

            if a1 != a2:
                if a1 == 0 and a2 == 1:
                    buy[i] = p[i]
                else:
                    sell[i] = p[i]

        # for step in self.env.action_scheme.portfolio.performance:
        net_worth = self.env.action_scheme.portfolio.net_worth
        self.logger.record(key="Net Worth", value=net_worth)

        buy = pd.Series(buy)
        sell = pd.Series(sell)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        fig.suptitle("Performance")

        axs[0].plot(np.arange(len(p)), p, label="price")
        axs[0].scatter(buy.index, buy.values, marker="^", color="green")
        axs[0].scatter(sell.index, sell.values, marker="^", color="red")
        axs[0].set_title("Trading Chart")

        performance_df = pd.DataFrame().from_dict(self.env.action_scheme.portfolio.performance, orient='index')
        performance_df.plot(ax=axs[1])
        axs[1].set_title("Net Worth")
        self.logger.record("Training Performance", Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))