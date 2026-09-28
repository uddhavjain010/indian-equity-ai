"""Unit Tests for Quantitative Signal Engine Rules."""
import pytest
import pandas as pd
from config.settings import SignalConfig
from src.signals.signal_engine import SignalEngine


@pytest.fixture
def signal_engine():
    return SignalEngine()


def test_buy_signal_rule_confluence(signal_engine):
    """Test that strong bullish forecast with supportive momentum yields BUY."""
    features = pd.Series({
        "RSI_14": 52.0,            # Neutral, < 70
        "MACD": 1.5,
        "MACD_Signal": 0.8,        # Bullish cross
        "MACD_Hist": 0.7,
        "EMA_20": 2400.0,
    })

    current_price = 2500.0  # Above EMA 20
    # +2.0% forecast (> +0.5% threshold)
    lstm_pred = 2550.0
    xgb_pred = 2548.0

    signal = signal_engine.generate_signal(
        symbol="RELIANCE.NS",
        current_price=current_price,
        lstm_prediction=lstm_pred,
        xgb_prediction=xgb_pred,
        latest_features=features
    )

    assert signal.signal == "BUY"
    assert "MACD is in bullish alignment" in " ".join(signal.reasons)
    assert signal.price_vs_ema == "Above"


def test_sell_signal_rule_confluence(signal_engine):
    """Test that bearish forecast with declining momentum yields SELL."""
    features = pd.Series({
        "RSI_14": 45.0,            # > 30
        "MACD": -2.0,
        "MACD_Signal": -1.0,       # Bearish
        "MACD_Hist": -1.0,
        "EMA_20": 2600.0,
    })

    current_price = 2500.0  # Below EMA 20
    # -2.0% forecast (< -0.5% threshold)
    lstm_pred = 2450.0
    xgb_pred = 2445.0

    signal = signal_engine.generate_signal(
        symbol="RELIANCE.NS",
        current_price=current_price,
        lstm_prediction=lstm_pred,
        xgb_prediction=xgb_pred,
        latest_features=features
    )

    assert signal.signal == "SELL"
    assert "MACD is in bearish alignment" in " ".join(signal.reasons)
    assert signal.price_vs_ema == "Below"


def test_hold_on_model_conflict(signal_engine):
    """Test that conflicting model directions enforce a HOLD signal."""
    features = pd.Series({
        "RSI_14": 50.0,
        "MACD": 0.5,
        "MACD_Signal": 0.2,
        "MACD_Hist": 0.3,
        "EMA_20": 2480.0,
    })

    current_price = 2500.0
    # LSTM predicts UP (+2%), XGBoost predicts DOWN (-2%)
    lstm_pred = 2550.0
    xgb_pred = 2450.0

    signal = signal_engine.generate_signal(
        symbol="RELIANCE.NS",
        current_price=current_price,
        lstm_prediction=lstm_pred,
        xgb_prediction=xgb_pred,
        latest_features=features
    )

    assert signal.signal == "HOLD"
    assert "conflicting" in " ".join(signal.reasons).lower()
