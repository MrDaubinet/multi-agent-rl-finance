
from ray.tune import Callback
import matplotlib.pyplot as plt

class RenderCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"on_checkpoint render callback")
        # 1. lets clear the last plot
        # 2. load the agent
        # 3. run through the evaluation data
        # 4. plot the results
        plt.cla('all')
