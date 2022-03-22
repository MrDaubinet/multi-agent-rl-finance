import pandas as pd
import numpy as np

class SineWaveDataGenerator:
    """Provides methods for generating sinewave data
    Attributes
    ----------

    Methods
    -------
    fetch(ticker,interval,start,end,period,prepost=False,auto_adjust=False,actions=False)
        Fetches data from yahoo finance.
    """

    def fetch(self,
              x_range: int = 1000,
              y_height: str = 100) -> pd.DataFrame:
        """Fetches data for different exchanges and cryptocurrency pairs.
        Parameters
        ----------
        x_range : int
            The number of data points to generate.
        y_height : int
            The height of the sine wave
        Returns
        -------
        `pd.DataFrame`
            A open, high, low, close and volume, dividends and splits for the specified stock ticker.
        """
        # generate x values over 1001
        x = np.arange(0, 2*np.pi, 2*np.pi / (x_range + 1))
        # generate y values from x values
        y = 50*np.sin(3*x) + y_height
        # reset x values over 1000
        x = np.arange(0, 2*np.pi, 2*np.pi / x_range)
        dataframe = pd.DataFrame(y, columns =['price'])
        return dataframe
