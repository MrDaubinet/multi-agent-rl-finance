import pandas as pd
import numpy as np
from math import ceil
from typing import List

class SineWaveDataGenerator:
    """Provides methods for generating sinewave data
    Attributes
    ----------

    Methods
    -------
    fetch(ticker,interval,start,end,period,prepost=False,auto_adjust=False,actions=False)
        Fetches data from yahoo finance.
    """

    def __init__(self, 
        x_sample: int = 1000,
        y_peaks: int = 3,
        y_height: str = 100,
        d_ratio: List[float] = [0.5, 0.3, 0.2]) -> None:
        """
        Parameters
        ----------
        x_range : int
            The number of data points to generate.
        y_height : int
            The height of the sine wave
        d_ratio : List[float]
            The ratio for the [train, test, evaluation] split.
        """
        self.x_sample = x_sample
        self.y_peaks = y_peaks
        self.y_height = y_height
        self.d_ratio = d_ratio
        self.data = self.generate()

    def generate(self):
        """
        Generates training, evaluation and testing data for a sinwave. 
        Parameters
        ----------
        x_range : int
            The number of data points to generate.
        y_height : int
            The height of the sine wave
        Returns
        -------
        `pd.DataFrame`:
            Price.
        """
        # generate x values to 1001
        x = np.arange(0, 2*np.pi, 2*np.pi / (self.x_sample + 1))
        # generate y values from x values
        y = 50*np.sin(2*x*self.y_peaks) + self.y_height
        # reset x values over 1000
        x = np.arange(0, 2*np.pi, 2*np.pi / self.x_sample)
        # generate the dataframe
        return pd.DataFrame(y, columns =['price'])

    def train(self):
        """
        Retrieve the training data from the data source.
        Returns
        -------
        `pd.DataFrame`:
            Price.
        """
        train_start_index = 0
        # Get the ending dataframe row from our train, test, evaluation ratio's
        train_end_index = ceil(len(self.data) * self.d_ratio[0])
        # get the dataframe with rows indexed for training
        dataframe = self.data.iloc[train_start_index:train_end_index]
        return dataframe
    
    def eval(self):
        """
        Retrieve the evaluation data from the data source.
        Returns
        -------
        `pd.DataFrame`:
            Price.
        """
        eval_start_index = ceil(len(self.data) * self.d_ratio[0])
        # Get the ending dataframe row from our train, test, evaluation ratio's
        eval_end_index = ceil(len(self.data) * (self.d_ratio[0] + self.d_ratio[1]))
        # get the dataframe with rows indexed for training
        dataframe = self.data.iloc[eval_start_index:eval_end_index]
        return dataframe

    def test(self):
        """
        Retrieve the testing data from the data source.
        Returns
        -------
        `pd.DataFrame`:
            Price.
        """
        test_start_index = ceil(len(self.data) * (self.d_ratio[0] + self.d_ratio[1]))
        # Get the ending dataframe row from our train, test, evaluation ratio's
        test_end_index = len(self.data)
        # get the dataframe with rows indexed for training
        dataframe = self.data.iloc[test_start_index:test_end_index]
        return dataframe