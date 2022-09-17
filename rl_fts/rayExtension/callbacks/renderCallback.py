
from ray.tune import Callback
import matplotlib.pyplot as plt
import tensorflow as tf
import io

class RenderCallback(Callback):
    def __init__(self, evaluation_frequency, name, log_dir) -> None:
        super().__init__()
        self.evaluation_frequency = evaluation_frequency
        self.log_dir = log_dir
        self.name = name

    def on_trial_result(self, iteration, trials, trial, result, **info):
        # plot the graphs
        if iteration % self.evaluation_frequency == 0:
            # plt.plot()

            fig = plt.gcf()
            # send image to tensorboard
            writer = tf.summary.create_file_writer(self.log_dir+self.name)
            with writer.as_default():
                data = self.plot_to_image(fig)
                tf.summary.image(self.name, data=data, step=iteration)

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

