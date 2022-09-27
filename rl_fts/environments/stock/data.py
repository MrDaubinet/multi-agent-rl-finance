import yfinance as yf
import pandas as pd

class StockDataGenerator:
    """Provides methods for retrieving data on different stocks from yahoo finance using the yfinance python module
    Attributes
    ----------

    Methods
    -------
    fetch(ticker,interval,start,end,period,prepost=False,auto_adjust=False,actions=False)
        Fetches data from yahoo finance.
    """

    def fetch(self,
              ticker: str,
              interval: str,
              start: str = None,
              end: str = None,
              period: str = "max",
              prepost: bool = False,
              auto_adjust: bool = True,
              actions: bool = True) -> pd.DataFrame:
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
        `pd.DataFrame`
            A open, high, low, close and volume, dividends and splits for the specified stock ticker.
        """
        msft = yf.Ticker(ticker)
        if start:
            df = msft.history(interval=interval, start=start, end=end, prepost=prepost, auto_adjust=auto_adjust, actions=actions)  
        else:
            df = msft.history(period, interval=interval, prepost=prepost, auto_adjust=auto_adjust, actions=actions)
        return df
