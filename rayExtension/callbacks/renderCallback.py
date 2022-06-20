
from ray.tune import Callback
import matplotlib.pyplot as plt

class RenderCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        # 1. lets clear the last plot
        # plt.cla()
        # 4. plot the graphs
        plt.draw()
        plt.pause(0.001)
