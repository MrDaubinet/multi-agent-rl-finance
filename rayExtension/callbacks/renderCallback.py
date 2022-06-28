
from ray.tune import Callback
import matplotlib.pyplot as plt

class RenderCallback(Callback):
    def __init__(self, evaluation_frequency) -> None:
        super().__init__()
        self.evaluation_frequency = evaluation_frequency

    def on_trial_result(self, iteration, trials, trial, result, **info):
        # 1. lets clear the last plot
        # plt.cla()
        # 4. plot the graphs
        if iteration % self.evaluation_frequency == 0:
            plt.draw()
            plt.pause(0.001)
