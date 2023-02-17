import yfinance as yf
import pandas as pd
from typing import List
from math import ceil

class StockDataGenerator:
    """Provides methods for retrieving data on different stocks from yahoo finance using the yfinance python module
    Attributes
    ----------
    ticker : str
        The ticker symbol of the stock.
    interval : str
        The data interval (intraday data cannot extend last 60 days) Valid intervals are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    start : str
        If not using period - Download start date string (YYYY-MM-DD) or datetime.
    end : str
        If not using period - Download end date string (YYYY-MM-DD) or datetime.
    period : str
        data period to download (Either Use period parameter or use start and end) Valid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    prepost : str
        Include Pre and Post market data in results? (Default is False)
    auto_adjust : str
        Adjust all OHLC automatically? (Default is True)
    actions : str
        Download stock dividends and stock splits events? (Default is True)


    Methods
    -------
    generate()
        Fetches data from yahoo finance.
    train()
        Generates the training data.
    eval()
        Generates the evaluation data.
    test()
        Generates the testing data.
    """

    def __init__(self,
              ticker: str,
              interval: str,
              start: str = None,
              end: str = None,
              period: str = "5y",
              prepost: bool = False,
              auto_adjust: bool = True,
              actions: bool = True,
              d_ratio: List[float] = [0.5, 0.3, 0.2]) -> pd.DataFrame:
        """Fetches data for different exchanges and cryptocurrency pairs.
        Parameters
        ----------
        ticker : str
            The ticker symbol of the stock.
        interval : str
            The data interval (intraday data cannot extend last 60 days) Valid intervals are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        start : str
            If not using period - Download start date string (YYYY-MM-DD) or datetime.
        end : str
            If not using period - Download end date string (YYYY-MM-DD) or datetime.
        period : str
            data period to download (Either Use period parameter or use start and end) Valid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        prepost : str
            Include Pre and Post market data in results? (Default is False)
        auto_adjust : str
            Adjust all OHLC automatically? (Default is True)
        actions : str
            Download stock dividends and stock splits events? (Default is True)
        Returns
        -------
        """
        self.ticker = ticker
        self.interval = interval
        self.start = start
        self.end = end
        self.period = period
        self.prepost = prepost
        self.auto_adjust = auto_adjust
        self.actions = actions
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
        msft = yf.Ticker(self.ticker)
        if self.start:
            return msft.history(
                interval=self.interval, 
                start=self.start, 
                end=self.end, 
                prepost=self.prepost, 
                auto_adjust=self.auto_adjust, 
                actions=self.actions
            ) 
        else:
            return msft.history(
                interval=self.interval, 
                period=self.period, 
                prepost=self.prepost, 
                auto_adjust=self.auto_adjust, 
                actions=self.actions
            )  
    

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
    
    def validate(self):
        """
        Retrieve the evaluation data from the data source.
        Returns
        -------
        `pd.DataFrame`:
            Price.
        """
        valid_start_index = ceil(len(self.data) * self.d_ratio[0])
        # Get the ending dataframe row from our train, test, evaluation ratio's
        valid_end_index = ceil(len(self.data) * (self.d_ratio[0] + self.d_ratio[1]))
        # get the dataframe with rows indexed for training
        dataframe = self.data.iloc[valid_start_index:valid_end_index]
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