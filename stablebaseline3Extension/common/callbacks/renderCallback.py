from typing import Union
import gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy

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
        # self.position_change_chart = PositionChangeChart()
        # self.env = self.env
        self.env.render()
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass