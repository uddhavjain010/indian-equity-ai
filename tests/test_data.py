"""Unit Tests for Data Validation & Provider Interface."""
import pytest
import pandas as pd
import numpy as np

from src.data.download_data import validate_ohlcv_data
from src.data.base_provider import BaseDataProvider
from src.data import get_data_provider


def create_synthetic_raw_data() -> pd.DataFrame:
    """Helper to create dummy OHLCV dataframe with clean baseline values."""
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "Open": [100.0, 102.0, 101.0, 105.0, 103.0, 104.0, 106.0, 107.0, 108.0, 109.0],
        "High": [105.0, 106.0, 104.0, 108.0, 106.0, 107.0, 109.0, 110.0, 112.0, 113.0],
        "Low": [98.0, 100.0, 99.0, 102.0, 101.0, 102.0, 104.0, 105.0, 106.0, 107.0],
        "Close": [102.0, 101.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        "Volume": [1000, 1500, 1200, 1800, 500, 100, 2000, 2200, 2100, 2500],
    }, index=dates)
    return df


def test_validation_removes_negative_volume():
    """Verify negative volume is sanitized."""
    df = create_synthetic_raw_data()
    clean_df, report = validate_ohlcv_data(df, symbol="TEST")
    assert (clean_df["Volume"] >= 0).all()


def test_validation_removes_impossible_prices():
    """Verify impossible prices (e.g. High < Low or Close < 0) are removed."""
    df = create_synthetic_raw_data()
    # Inject impossible price
    df.loc[df.index[2], "High"] = 90.0  # High is below Low (99.0)
    df.loc[df.index[5], "Close"] = -10.0  # Negative price

    clean_df, report = validate_ohlcv_data(df, symbol="TEST")
    assert report["impossible_price_rows_removed"] >= 2
    assert len(clean_df) <= len(df) - 2


def test_validation_deduplication():
    """Verify duplicate date records are pruned."""
    df = create_synthetic_raw_data()
    # Duplicate first row
    dup_row = df.iloc[[0]]
    df_with_dup = pd.concat([df, dup_row])
    clean_df, report = validate_ohlcv_data(df_with_dup, symbol="TEST")
    assert report["duplicate_rows_removed"] == 1
    assert len(clean_df) == len(df)


def test_data_provider_factory():
    """Verify default provider conforms to BaseDataProvider."""
    provider = get_data_provider()
    assert isinstance(provider, BaseDataProvider)
