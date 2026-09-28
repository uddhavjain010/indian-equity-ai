"""Yahoo Finance Data Provider for Indian Equities (NSE/BSE).

Implements BaseDataProvider using yfinance to fetch historical daily OHLCV
and near-real-time quotes without requiring paid API infrastructure.
"""
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import yfinance as yf

from src.data.base_provider import BaseDataProvider
from src.utils.logger import get_logger

logger = get_logger("yfinance_provider")


class YFinanceProvider(BaseDataProvider):
    """Data provider accessing market data via Yahoo Finance."""

    def __init__(self):
        super().__init__()

    def fetch_historical_data(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for an Indian equity.

        Args:
            symbol: Symbol with exchange extension (e.g. 'RELIANCE.NS' or 'TCS.BO')
            period: Duration string ('1y', '2y', '5y', '10y')
            interval: Bar resolution ('1d')

        Returns:
            pd.DataFrame: Cleaned OHLCV data with DatetimeIndex
        """
        logger.info("Fetching %s historical data for %s (interval=%s)", period, symbol, interval)
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=False)

        if df.empty:
            # Fallback to yf.download in case Ticker.history returned empty
            logger.warning("Ticker.history returned empty for %s, trying yf.download", symbol)
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)

        if df.empty:
            raise ValueError(f"No historical data found for symbol '{symbol}' with period '{period}'.")

        # Flatten multi-level columns if present (from recent yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            # Take the level with Price names
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # Standardize required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            # Look for case-insensitive matching
            col_map = {c.lower(): c for c in df.columns}
            renames = {}
            for req in missing_cols:
                if req.lower() in col_map:
                    renames[col_map[req.lower()]] = req
            df = df.rename(columns=renames)
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Data for '{symbol}' is missing required OHLCV columns: {missing_cols}")

        df = df[required_cols].copy()

        # Handle timezone in index: localize or strip to naive UTC/IST
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.index.name = "Date"
        df = df.sort_index()

        # Convert numeric types
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop entirely NaN rows
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        logger.info("Successfully fetched %d records for %s", len(df), symbol)
        return df

    def fetch_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch the latest available near-real-time market quote.

        Args:
            symbol: Ticker symbol (e.g. 'RELIANCE.NS')

        Returns:
            Dict containing current price, previous close, change, etc.
        """
        logger.info("Fetching latest quote for %s", symbol)
        ticker = yf.Ticker(symbol)

        current_price = None
        previous_close = None
        volume = 0
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

        # Method 1: Try fast_info
        try:
            fast_info = getattr(ticker, "fast_info", None)
            if fast_info is not None:
                current_price = getattr(fast_info, "last_price", None)
                previous_close = getattr(fast_info, "previous_close", None)
                volume = int(getattr(fast_info, "last_volume", 0) or 0)
        except Exception as e:
            logger.debug("fast_info lookup failed: %s", e)

        # Method 2: Fallback to recent 5-day intraday or daily history
        if current_price is None or pd.isna(current_price):
            try:
                hist = ticker.history(period="5d", interval="1d")
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                if not hist.empty and len(hist) >= 1:
                    current_price = float(hist["Close"].iloc[-1])
                    previous_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else float(hist["Open"].iloc[-1])
                    volume = int(hist["Volume"].iloc[-1])
                    last_date = hist.index[-1]
                    timestamp_str = f"{last_date.strftime('%Y-%m-%d')} (Market Close/EOD)"
            except Exception as e:
                logger.warning("History fallback failed for quote: %s", e)

        if current_price is None or pd.isna(current_price):
            raise RuntimeError(f"Could not retrieve latest price quote for {symbol}.")

        if previous_close is None or pd.isna(previous_close) or previous_close == 0:
            previous_close = current_price

        change = current_price - previous_close
        change_pct = (change / previous_close) * 100.0 if previous_close else 0.0

        return {
            "symbol": symbol,
            "current_price": round(float(current_price), 2),
            "previous_close": round(float(previous_close), 2),
            "change": round(float(change), 2),
            "change_pct": round(float(change_pct), 2),
            "volume": int(volume),
            "timestamp": timestamp_str,
            "data_status": "Latest available market data (near-real-time / educational)",
            "provider": "yfinance",
        }

    def validate_symbol(self, symbol: str) -> bool:
        """Verify whether a symbol has valid market data."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            return not hist.empty
        except Exception:
            return False
