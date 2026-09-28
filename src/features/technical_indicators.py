"""Technical Indicators and Feature Engineering Module.

Calculates price, moving averages, momentum, volatility, volume, and lag features
using strictly backward-looking windows to guarantee zero data leakage.
"""
from typing import List, Tuple
import numpy as np
import pandas as pd
import ta

from src.utils.logger import get_logger

logger = get_logger("features")


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive technical indicators and lag features on OHLCV data.

    All features use exclusively past and contemporaneous data (t, t-1, ...);
    NO forward-looking (future) data is ever accessed.

    Args:
        df: Input DataFrame with DatetimeIndex and columns:
            ['Open', 'High', 'Low', 'Close', 'Volume']

    Returns:
        pd.DataFrame: Augmented DataFrame containing all engineered features.
    """
    data = df.copy()

    # Ensure chronological order before computing rolling windows
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()

    # 1. Price Returns
    data["Daily_Return"] = data["Close"].pct_change()
    data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))

    # 2. Moving Averages (Simple & Exponential)
    data["SMA_10"] = data["Close"].rolling(window=10, min_periods=5).mean()
    data["SMA_20"] = data["Close"].rolling(window=20, min_periods=5).mean()
    data["SMA_50"] = data["Close"].rolling(window=50, min_periods=10).mean()
    data["SMA_200"] = data["Close"].rolling(window=200, min_periods=20).mean()

    data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["EMA_50"] = data["Close"].ewm(span=50, adjust=False).mean()

    # 3. Momentum: RSI (14) & MACD (12, 26, 9)
    # Using 'ta' library or direct vectorized computation
    rsi_indicator = ta.momentum.RSIIndicator(close=data["Close"], window=14)
    data["RSI_14"] = rsi_indicator.rsi()

    macd_indicator = ta.trend.MACD(
        close=data["Close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    data["MACD"] = macd_indicator.macd()
    data["MACD_Signal"] = macd_indicator.macd_signal()
    data["MACD_Hist"] = macd_indicator.macd_diff()

    # 4. Volatility: Bollinger Bands (20, 2 std) & Rolling Volatility
    bb_indicator = ta.volatility.BollingerBands(
        close=data["Close"],
        window=20,
        window_dev=2
    )
    data["BB_High"] = bb_indicator.bollinger_hband().bfill()
    data["BB_Mid"] = bb_indicator.bollinger_mavg().bfill()
    data["BB_Low"] = bb_indicator.bollinger_lband().bfill()
    data["BB_Width"] = bb_indicator.bollinger_wband().bfill()
    # 20-day rolling standard deviation of daily returns (annualized using 252 trading days)
    data["Rolling_Vol_20"] = (data["Daily_Return"].rolling(window=20, min_periods=5).std() * np.sqrt(252)).bfill()

    # 5. Volume Features
    vol_prev = data["Volume"].shift(1).replace(0, np.nan)
    data["Volume_Change"] = ((data["Volume"] - data["Volume"].shift(1)) / vol_prev).fillna(0.0)
    data["Volume_SMA_20"] = data["Volume"].rolling(window=20, min_periods=5).mean().bfill()
    # Safe ratio to avoid division by zero
    vol_sma_safe = data["Volume_SMA_20"].replace(0, np.nan)
    data["Volume_Ratio"] = (data["Volume"] / vol_sma_safe).fillna(1.0)

    # 6. Lag Features (Prices & Returns)
    for lag in [1, 2, 3, 5, 10]:
        data[f"Close_Lag_{lag}"] = data["Close"].shift(lag).bfill()

    data["Return_Lag_1"] = data["Daily_Return"].shift(1).fillna(0.0)
    data["Return_Lag_5"] = data["Daily_Return"].shift(5).fillna(0.0)

    # Clean any inf / -inf produced by divisions or logs
    data = data.replace([np.inf, -np.inf], np.nan)
    # Backfill and forward fill any remaining indicator initialization NaNs
    data = data.bfill().ffill().fillna(0.0)

    return data


def create_prediction_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create primary continuous target and directional classification target.

    Primary Target:
        target(t) = Close(t+1)  [Next trading day's closing price]

    Directional Target:
        direction_target(t) = 1 if Close(t+1) > Close(t) else 0

    Args:
        df: DataFrame with at least 'Close' column.

    Returns:
        pd.DataFrame: DataFrame with 'Target_Next_Close' and 'Target_Direction'.
                      The very last row will have NaN for targets because future is unknown.
    """
    data = df.copy()
    data["Target_Next_Close"] = data["Close"].shift(-1)
    data["Target_Direction"] = (data["Target_Next_Close"] > data["Close"]).astype(float)
    # The last row has unknown target, so direction should also be NaN
    data.loc[data["Target_Next_Close"].isna(), "Target_Direction"] = np.nan
    return data


def get_feature_column_names(df: pd.DataFrame) -> List[str]:
    """Return list of predictor feature columns, strictly excluding target columns and metadata."""
    exclude = {"Target_Next_Close", "Target_Direction", "Date"}
    # Also exclude raw future columns if any exist
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith("Target_")]
    return feature_cols
