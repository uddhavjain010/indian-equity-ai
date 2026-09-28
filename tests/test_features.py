"""Unit Tests for Technical Indicator Calculation and Target Generation."""
import pytest
import pandas as pd
import numpy as np

from src.features.technical_indicators import (
    calculate_technical_indicators,
    create_prediction_targets,
    get_feature_column_names,
)


@pytest.fixture
def sample_ohlcv_data():
    """Generate 60 days of synthetic geometric Brownian motion OHLCV data."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 1000.0 * np.cumprod(1.0 + returns)
    
    highs = prices * (1.0 + np.abs(np.random.normal(0.005, 0.002, n)))
    lows = prices * (1.0 - np.abs(np.random.normal(0.005, 0.002, n)))
    opens = (prices + lows) / 2.0
    volume = np.random.randint(500000, 2000000, n)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": prices,
        "Volume": volume
    }, index=dates)
    return df


def test_indicator_computation_completeness(sample_ohlcv_data):
    """Verify all technical indicators and lag features are calculated."""
    df_feat = calculate_technical_indicators(sample_ohlcv_data)

    expected_indicators = [
        "Daily_Return", "Log_Return",
        "SMA_10", "SMA_20", "SMA_50", "SMA_200",
        "EMA_10", "EMA_20", "EMA_50",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_High", "BB_Mid", "BB_Low", "BB_Width", "Rolling_Vol_20",
        "Volume_Change", "Volume_SMA_20", "Volume_Ratio",
        "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5", "Close_Lag_10",
        "Return_Lag_1", "Return_Lag_5"
    ]

    for col in expected_indicators:
        assert col in df_feat.columns, f"Indicator column {col} is missing."
        assert not df_feat[col].isna().any(), f"Indicator column {col} contains unexpected NaN."
        assert not np.isinf(df_feat[col]).any(), f"Indicator column {col} contains unexpected Inf."


def test_target_creation(sample_ohlcv_data):
    """Verify target(t) = Close(t+1) and direction_target logic."""
    df_feat = calculate_technical_indicators(sample_ohlcv_data)
    df_tagged = create_prediction_targets(df_feat)

    assert "Target_Next_Close" in df_tagged.columns
    assert "Target_Direction" in df_tagged.columns

    # Check for row 0: Target_Next_Close should equal Close of row 1
    assert df_tagged["Target_Next_Close"].iloc[0] == pytest.approx(sample_ohlcv_data["Close"].iloc[1])

    # Check direction: 1 if next close > current close, else 0
    for i in range(len(sample_ohlcv_data) - 1):
        actual_up = 1.0 if sample_ohlcv_data["Close"].iloc[i+1] > sample_ohlcv_data["Close"].iloc[i] else 0.0
        assert df_tagged["Target_Direction"].iloc[i] == actual_up

    # Last row target must be NaN
    assert np.isnan(df_tagged["Target_Next_Close"].iloc[-1])
    assert np.isnan(df_tagged["Target_Direction"].iloc[-1])
