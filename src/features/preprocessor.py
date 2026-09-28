"""Chronological Data Preprocessing & Anti-Leakage Pipeline.

Guarantees zero data leakage by:
- Enforcing strictly chronological temporal splits (No random shuffling).
- Fitting feature and target scalers SOLELY on the training portion.
- Transforming validation and testing sets strictly using train-fitted parameters.
- Constructing causal, strictly backward-looking sequence windows for LSTM.
"""
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.settings import ModelConfig
from src.features.technical_indicators import (
    calculate_technical_indicators,
    create_prediction_targets,
    get_feature_column_names,
)
from src.utils.logger import get_logger

logger = get_logger("preprocessor")


@dataclass
class ProcessedDataset:
    """Container holding strictly partitioned time-series datasets."""
    # Tabular (XGBoost)
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    
    # LSTM Sequences (3D: [samples, lookback, features])
    X_train_seq: np.ndarray
    y_train_seq: np.ndarray
    X_val_seq: np.ndarray
    y_val_seq: np.ndarray
    X_test_seq: np.ndarray
    y_test_seq: np.ndarray
    
    # Associated dates for evaluation & plotting
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    test_current_close: pd.Series
    
    # Scalers (Fitted solely on Training data)
    feature_scaler: Any
    target_scaler: Any
    
    # Feature list
    feature_names: List[str]
    lookback: int


def prepare_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators, add targets, and drop initial warm-up NaNs safely.

    Args:
        df: Raw cleaned OHLCV DataFrame.

    Returns:
        pd.DataFrame: Complete feature & target dataset.
    """
    df_feat = calculate_technical_indicators(df)
    df_tagged = create_prediction_targets(df_feat)
    return df_tagged


def split_chronologically(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into Train, Validation, and Test partitions strictly in chronological order.

    No shuffling or random sampling is allowed.

    Args:
        df: Ordered DataFrame.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        "Chronological Split: Total=%d | Train=%d (%s to %s) | Val=%d (%s to %s) | Test=%d (%s to %s)",
        n,
        len(train_df), str(train_df.index[0])[:10], str(train_df.index[-1])[:10],
        len(val_df), str(val_df.index[0])[:10], str(val_df.index[-1])[:10],
        len(test_df), str(test_df.index[0])[:10], str(test_df.index[-1])[:10],
    )
    return train_df, val_df, test_df


def create_lstm_sequences(
    X_scaled: np.ndarray,
    y_scaled: np.ndarray,
    lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct causal 3D sequence arrays for recurrent models.

    For index i, sequence is X[i - lookback : i], predicting target y[i].
    Strictly past-to-present; zero forward leakage.

    Args:
        X_scaled: 2D array of normalized feature values [samples, n_features].
        y_scaled: 1D or 2D array of normalized target values [samples].
        lookback: Temporal window length (e.g. 60 bars).

    Returns:
        Tuple of (X_seq, y_seq) where X_seq has shape [samples - lookback, lookback, n_features].
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback : i])
        y_seq.append(y_scaled[i])
    return np.array(X_seq), np.array(y_seq)


def preprocess_stock_data(
    df: pd.DataFrame,
    config: Optional[ModelConfig] = None
) -> ProcessedDataset:
    """Full chronological preprocessing pipeline with zero leakage guarantees.

    1. Engineers backward-looking indicators.
    2. Drops rows with indicator warm-up NaNs (e.g. SMA 200).
    3. Retains last row separately for next-day live inference.
    4. Chronologically partitions historical supervised dataset.
    5. Fits StandardScaler/MinMaxScaler strictly on training set.
    6. Normalizes validation and test sets using training parameters.
    7. Generates causal sequences for LSTM.

    Args:
        df: Validated raw OHLCV DataFrame.
        config: ModelConfig parameters.

    Returns:
        ProcessedDataset bundle.
    """
    cfg = config or ModelConfig()
    df_prepared = prepare_features_and_targets(df)

    feature_cols = get_feature_column_names(df_prepared)
    
    # We drop NaN rows caused by rolling windows and drop the very last row where Target_Next_Close is NaN
    # (The last row is used exclusively for future live prediction)
    supervised_df = df_prepared.dropna(subset=feature_cols + ["Target_Next_Close"]).copy()

    if len(supervised_df) < (cfg.lookback + 50):
        raise ValueError(
            f"Insufficient historical data ({len(supervised_df)} rows) after feature calculation "
            f"and lookback window ({cfg.lookback}). Please select a longer historical period (e.g., 2y or 5y)."
        )

    # Chronological partition
    train_df, val_df, test_df = split_chronologically(
        supervised_df,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio
    )

    X_train_raw = train_df[feature_cols]
    y_train_raw = train_df["Target_Next_Close"]
    X_val_raw = val_df[feature_cols]
    y_val_raw = val_df["Target_Next_Close"]
    X_test_raw = test_df[feature_cols]
    y_test_raw = test_df["Target_Next_Close"]

    # ANTI-LEAKAGE SCALER: FIT ONLY ON TRAINING DATA
    feature_scaler = StandardScaler()
    feature_scaler.fit(X_train_raw)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(y_train_raw.values.reshape(-1, 1))

    # Transform without re-fitting
    X_train_scaled = feature_scaler.transform(X_train_raw)
    X_val_scaled = feature_scaler.transform(X_val_raw)
    X_test_scaled = feature_scaler.transform(X_test_raw)

    y_train_scaled = target_scaler.transform(y_train_raw.values.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val_raw.values.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test_raw.values.reshape(-1, 1)).flatten()

    # Construct LSTM sequences
    X_train_seq, y_train_seq = create_lstm_sequences(X_train_scaled, y_train_scaled, cfg.lookback)
    X_val_seq, y_val_seq = create_lstm_sequences(X_val_scaled, y_val_scaled, cfg.lookback)
    X_test_seq, y_test_seq = create_lstm_sequences(X_test_scaled, y_test_scaled, cfg.lookback)

    # Dates aligned with target for evaluation
    train_dates = train_df.index[cfg.lookback:]
    val_dates = val_df.index[cfg.lookback:]
    test_dates = test_df.index[cfg.lookback:]
    test_current_close = test_df["Close"].iloc[cfg.lookback:]

    logger.info(
        "Preprocessing complete: Train seq shape=%s, Val seq shape=%s, Test seq shape=%s",
        X_train_seq.shape, X_val_seq.shape, X_test_seq.shape
    )

    return ProcessedDataset(
        X_train=X_train_raw,
        y_train=y_train_raw,
        X_val=X_val_raw,
        y_val=y_val_raw,
        X_test=X_test_raw,
        y_test=y_test_raw,
        X_train_seq=X_train_seq,
        y_train_seq=y_train_seq,
        X_val_seq=X_val_seq,
        y_val_seq=y_val_seq,
        X_test_seq=X_test_seq,
        y_test_seq=y_test_seq,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        test_current_close=test_current_close,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_names=feature_cols,
        lookback=cfg.lookback,
    )


def integrate_live_tick_into_history(
    df_daily: pd.DataFrame,
    live_price: float,
    live_volume: Optional[int] = None
) -> pd.DataFrame:
    """Safely integrate live streaming tick/LTP into historical daily OHLCV series.

    Maintains complete feature schema alignment with models trained on daily bars.
    Does NOT fabricate daily indicators from a solitary tick. Instead, updates the
    latest market state bar (Close = LTP, High = max(High, LTP), Low = min(Low, LTP))
    and causally recalculates technical indicators across the complete historical series.

    Args:
        df_daily: Historical daily OHLCV dataframe.
        live_price: Latest Traded Price (LTP) from WebSocket or quote.
        live_volume: Optional latest cumulative traded volume.

    Returns:
        pd.DataFrame: Feature-engineered DataFrame reflecting the latest market state.
    """
    if df_daily.empty:
        raise ValueError("Cannot integrate live tick into empty historical dataframe.")

    df_live = df_daily.copy()
    if live_price is not None and live_price > 0:
        # Update the most recent day bar's price
        idx = df_live.index[-1]
        df_live.loc[idx, "Close"] = float(live_price)
        df_live.loc[idx, "High"] = max(float(df_live.loc[idx, "High"]), float(live_price))
        df_live.loc[idx, "Low"] = min(float(df_live.loc[idx, "Low"]), float(live_price))
        if live_volume is not None and live_volume > 0:
            df_live.loc[idx, "Volume"] = max(int(df_live.loc[idx, "Volume"]), int(live_volume))

    # Recalculate indicators causally
    df_features = calculate_technical_indicators(df_live)
    return df_features

