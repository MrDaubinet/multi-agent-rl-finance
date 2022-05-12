from typing import Union
import gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import json

import sys
sys.path.append("..")

class RenderHyperParameterCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param env: (Union[gym.Env, VecEnv]) environment object
    """
    def __init__(self, 
      env: Union[gym.Env, VecEnv],
      window_size: int,
      n_steps: int,
      verbose=0,
      ):
        super(RenderHyperParameterCallback, self).__init__(verbose)
        self.env = env
        self.window_size = window_size
        self.n_steps = n_steps
    
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        """
        self.logger.record(key="window_size", value=self.window_size)
        self.logger.record(key="n_steps", value=self.n_steps)
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # config = {
        #   "window_size": self.window_size,
        #   "n_steps": self.n_steps
        # }
        # result = config.items()
        # self.logger.record("TrainConfig", tf.convert_to_tensor(np.array(list(config.items()))))
        # test = RenderHyperParameterCallback.pretty_json(config)


    @staticmethod
    def pretty_json(hp):
      json_hp = json.dumps(hp, indent=1)
      return "".join("\t" + line for line in json_hp.splitlines(True))