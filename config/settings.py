"""Application, Model, and Signal Configuration Settings."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BACKTEST_DATA_DIR = DATA_DIR / "backtests"
MODELS_DIR = BASE_DIR / "models"
LSTM_MODEL_DIR = MODELS_DIR / "lstm"
XGBOOST_MODEL_DIR = MODELS_DIR / "xgboost"
LOGS_DIR = BASE_DIR / "logs"

# Ensure runtime directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, BACKTEST_DATA_DIR, LSTM_MODEL_DIR, XGBOOST_MODEL_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

@dataclass
class AppConfig:
    """Core Application Settings."""
    provider: str = os.getenv("YFINANCE_PROVIDER", "default")
    upstox_client_id: str = os.getenv("UPSTOX_CLIENT_ID", os.getenv("UPSTOX_API_KEY", ""))
    upstox_client_secret: str = os.getenv("UPSTOX_CLIENT_SECRET", os.getenv("UPSTOX_API_SECRET", ""))
    upstox_access_token: str = os.getenv("UPSTOX_ACCESS_TOKEN", "")
    upstox_redirect_uri: str = os.getenv("UPSTOX_REDIRECT_URI", "https://127.0.0.1:5000/")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    default_symbol: str = "RELIANCE.NS"
    default_period: str = "2y"
    default_interval: str = "1d"
    supported_periods: List[str] = field(default_factory=lambda: ["1y", "2y", "5y", "10y"])
    supported_intervals: List[str] = field(default_factory=lambda: ["1d"])

@dataclass
class ModelConfig:
    """ML Model Hyperparameters and Dataset Splits."""
    # Data Splitting (strictly chronological, no shuffling)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Sequence settings for LSTM
    lookback: int = 60
    lstm_units_1: int = 64
    lstm_units_2: int = 32
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 25
    batch_size: int = 32
    early_stopping_patience: int = 6
    
    # XGBoost regression parameters
    xgb_n_estimators: int = 150
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_random_state: int = 42

@dataclass
class SignalConfig:
    """Signal Generation Engine Thresholds and Rule Parameters."""
    # Return thresholds for educational signals (e.g. 0.5% predicted move)
    min_predicted_return_buy: float = 0.005    # +0.5% predicted return
    min_predicted_return_sell: float = -0.005  # -0.5% predicted return
    
    # RSI thresholds
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # Signal labels
    SIGNAL_BUY: str = "BUY"
    SIGNAL_HOLD: str = "HOLD"
    SIGNAL_SELL: str = "SELL"
