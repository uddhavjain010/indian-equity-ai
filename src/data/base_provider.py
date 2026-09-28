"""Base Data Provider Abstraction.

Defines the contract for fetching historical and near-real-time market data
for Indian equities, allowing seamless swapping of market data vendors.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseDataProvider(ABC):
    """Abstract Base Class for Indian Stock Data Providers."""

    @abstractmethod
    def fetch_historical_data(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Args:
            symbol: Ticker symbol (e.g. 'RELIANCE.NS' or '500325.BO')
            period: Time window (e.g. '1y', '2y', '5y', '10y')
            interval: Bar resolution (e.g. '1d')

        Returns:
            pd.DataFrame: Standardized OHLCV dataframe with DatetimeIndex and columns:
                          ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        pass

    @abstractmethod
    def fetch_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch the latest available near-real-time market quote.

        Args:
            symbol: Ticker symbol (e.g. 'RELIANCE.NS')

        Returns:
            Dict containing:
                - symbol: str
                - current_price: float
                - previous_close: float
                - change: float
                - change_pct: float
                - volume: int
                - timestamp: str
                - data_status: str (e.g. "Latest available market data")
        """
        pass

    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Verify whether a symbol exists and is tradeable on the provider."""
        pass
