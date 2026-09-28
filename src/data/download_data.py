"""Historical Data Downloader and Rigorous Data Validator.

Fetches historical market data, runs comprehensive financial data validation,
and persists sanitized datasets in data/raw/.
"""
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from config.settings import RAW_DATA_DIR
from src.data import get_data_provider
from src.utils.logger import get_logger

logger = get_logger("download_data")


def sanitize_filename(symbol: str) -> str:
    """Convert symbol like 'RELIANCE.NS' to safe filename 'RELIANCE_NS.csv'."""
    clean = symbol.replace(".", "_").replace(":", "_").replace("^", "")
    return f"{clean}.csv"


def validate_ohlcv_data(df: pd.DataFrame, symbol: str = "ASSET") -> Tuple[pd.DataFrame, dict]:
    """Perform rigorous quantitative validation on historical OHLCV data.

    Validates:
    - Chronological ordering and monotonic date index.
    - Duplicate timestamps.
    - Missing OHLCV fields.
    - Impossible prices: High < Low, Open/High/Low/Close <= 0,
      High < Open, High < Close, Low > Open, Low > Close.
    - Negative or zero volume handling.

    Returns:
        Tuple[pd.DataFrame, dict]: Cleaned DataFrame and validation report dict.
    """
    report = {
        "symbol": symbol,
        "initial_rows": len(df),
        "duplicate_rows_removed": 0,
        "nan_rows_dropped": 0,
        "impossible_price_rows_removed": 0,
        "zero_volume_rows": 0,
        "is_chronological": True,
        "start_date": None,
        "end_date": None,
        "final_rows": 0,
    }

    if df.empty:
        raise ValueError(f"Empty DataFrame supplied for validation of {symbol}.")

    # 1. Deduplicate by index (Date)
    initial_len = len(df)
    df = df[~df.index.duplicated(keep="first")]
    report["duplicate_rows_removed"] = initial_len - len(df)

    # 2. Sort chronologically
    if not df.index.is_monotonic_increasing:
        report["is_chronological"] = False
        df = df.sort_index()

    # 3. Check and drop NaN in OHLC
    nan_mask = df[["Open", "High", "Low", "Close"]].isna().any(axis=1)
    nan_count = int(nan_mask.sum())
    if nan_count > 0:
        logger.warning("Dropping %d rows with NaN in OHLC prices for %s", nan_count, symbol)
        df = df[~nan_mask]
        report["nan_rows_dropped"] = nan_count

    # 4. Check impossible prices
    # Rule A: All prices > 0
    positive_mask = (df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)
    # Rule B: High must be highest of the bar
    high_valid = (df["High"] >= df["Low"]) & (df["High"] >= df["Open"]) & (df["High"] >= df["Close"])
    # Rule C: Low must be lowest of the bar
    low_valid = (df["Low"] <= df["Open"]) & (df["Low"] <= df["Close"])

    valid_prices = positive_mask & high_valid & low_valid
    invalid_price_count = int((~valid_prices).sum())
    if invalid_price_count > 0:
        logger.warning(
            "Found %d rows with impossible price logic (e.g. High < Low or Non-positive price) for %s; removing.",
            invalid_price_count, symbol
        )
        df = df[valid_prices]
        report["impossible_price_rows_removed"] = invalid_price_count

    # 5. Volume validation: non-negative
    if "Volume" in df.columns:
        # Negative volumes are invalid
        df = df[df["Volume"] >= 0]
        # Count zero volume days
        zero_vol = int((df["Volume"] == 0).sum())
        report["zero_volume_rows"] = zero_vol
        # Replace zero volume with small epsilon or forward fill if needed, but keep intact with positive min
        df["Volume"] = df["Volume"].clip(lower=0)

    # Update summary report
    report["final_rows"] = len(df)
    if not df.empty:
        report["start_date"] = str(df.index[0])
        report["end_date"] = str(df.index[-1])

    logger.info("Validation complete for %s: %d clean records retained.", symbol, report["final_rows"])
    return df, report


def download_and_save_stock_data(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    output_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Download historical stock data from provider, validate it, and persist to raw CSV.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE.NS')
        period: Timeframe ('1y', '2y', '5y', '10y')
        interval: Bar resolution ('1d')
        output_dir: Target directory (defaults to RAW_DATA_DIR)
        force_refresh: Whether to re-download if file already exists

    Returns:
        Tuple[pd.DataFrame, dict]: Sanitized DataFrame and validation report.
    """
    save_dir = output_dir or RAW_DATA_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = sanitize_filename(symbol)
    filepath = save_dir / filename

    # Check local cache if not forcing refresh
    expected_min_rows = {
        "1y": 200,
        "2y": 450,
        "5y": 1150,
        "10y": 2300,
    }.get(period, 200)

    if filepath.exists() and not force_refresh:
        logger.info("Checking existing cached data at %s", filepath)
        try:
            df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
            if len(df) >= expected_min_rows:
                clean_df, report = validate_ohlcv_data(df, symbol=symbol)
                logger.info("Using cached data (%d rows) for %s", len(clean_df), symbol)
                return clean_df, report
            else:
                logger.info(
                    "Cached data for %s has %d rows (less than expected %d for %s). Re-fetching from provider.",
                    symbol, len(df), expected_min_rows, period
                )
        except Exception as e:
            logger.warning("Failed to read cached file %s: %s. Re-downloading.", filepath, e)

    provider = get_data_provider()
    raw_df = provider.fetch_historical_data(symbol, period=period, interval=interval)
    clean_df, report = validate_ohlcv_data(raw_df, symbol=symbol)

    # Save to CSV
    clean_df.to_csv(filepath, index=True)
    logger.info("Saved validated data to %s (%d rows)", filepath, len(clean_df))

    return clean_df, report