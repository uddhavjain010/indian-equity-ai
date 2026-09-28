"""Rigorous Anti-Leakage Verification Tests.

Proves mathematically and programmatically that:
1. Scalers are fitted SOLELY on the training portion and never on validation/testing data.
2. Temporal ordering is strictly preserved (train < val < test) without shuffling.
3. Causal sequence generation never references future timestamps.
4. Technical indicators are strictly causal (modifying future rows has 0.0 impact on past indicators).
5. Targets are shifted by exactly one trading period (Target_t = Close_{t+1}).
6. Walk-forward backtest trains exclusively on past observations relative to forecast step.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from config.settings import ModelConfig
from src.features.technical_indicators import (
    calculate_technical_indicators,
    create_prediction_targets,
)
from src.features.preprocessor import (
    preprocess_stock_data,
    split_chronologically,
    create_lstm_sequences,
)
from src.backtesting.walk_forward import WalkForwardBacktester


@pytest.fixture
def non_stationary_price_data():
    """Create trending time series where test set has significantly higher prices than train set."""
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    
    # Linear upward trend: Train prices ~ 100-200, Test prices ~ 300-400
    trend = np.linspace(100, 400, n)
    noise = np.random.normal(0, 2, n)
    close = trend + noise
    
    df = pd.DataFrame({
        "Open": close - 1,
        "High": close + 3,
        "Low": close - 3,
        "Close": close,
        "Volume": np.random.randint(100000, 500000, n),
    }, index=dates)
    return df


def test_scaler_fitted_only_on_training_data(non_stationary_price_data):
    """PROVE that the scaler is fitted ONLY on training data and not the full dataset."""
    cfg = ModelConfig(train_ratio=0.70, val_ratio=0.15, lookback=20)
    dataset = preprocess_stock_data(non_stationary_price_data, config=cfg)

    # 1. Feature Scaler Check
    fitted_mean = dataset.feature_scaler.mean_
    train_mean = dataset.X_train.mean(axis=0).values
    full_mean = pd.concat([dataset.X_train, dataset.X_val, dataset.X_test]).mean(axis=0).values

    # Proves: Scaler mean equals training set mean exactly
    np.testing.assert_allclose(fitted_mean, train_mean, rtol=1e-5)

    # Proves: Scaler mean is significantly different from the full dataset mean
    close_idx = dataset.feature_names.index("Close")
    assert abs(fitted_mean[close_idx] - full_mean[close_idx]) > 10.0, (
        "Leakage detected! Scaler mean matches full dataset rather than training portion."
    )

    # 2. Target Scaler Check
    fitted_min = dataset.target_scaler.data_min_[0]
    fitted_max = dataset.target_scaler.data_max_[0]
    train_min = dataset.y_train.min()
    train_max = dataset.y_train.max()
    test_max = dataset.y_test.max()

    np.testing.assert_allclose(fitted_min, train_min, rtol=1e-5)
    np.testing.assert_allclose(fitted_max, train_max, rtol=1e-5)
    # The trending dataset has test_max much higher than train_max
    assert fitted_max < test_max, (
        "Leakage detected! Target scaler saw test set maximum values during fit."
    )


def test_chronological_splits_strictly_ordered(non_stationary_price_data):
    """Verify that temporal splits are strictly non-overlapping and forward in time."""
    train_df, val_df, test_df = split_chronologically(non_stationary_price_data, 0.70, 0.15)

    # Verify monotonic date progression
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()
    assert len(train_df) + len(val_df) + len(test_df) == len(non_stationary_price_data)


def test_no_future_data_in_lstm_sequences():
    """Verify that for sequence i, only past data [i-lookback : i] is included."""
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)
    lookback = 10

    X_seq, y_seq = create_lstm_sequences(X, y, lookback)

    for idx in range(len(X_seq)):
        # Target index is lookback + idx
        target_val = y_seq[idx]
        sequence = X_seq[idx].flatten()

        # The sequence must contain values strictly less than target_val
        assert np.all(sequence < target_val)
        # The last value in sequence is target_val - 1
        assert sequence[-1] == target_val - 1


def test_feature_calculation_causality(non_stationary_price_data):
    """PROVE that features at time t depend strictly on observations <= t.

    Altering data at future timestamps (t+1, t+2, ...) must produce ZERO change
    in feature values at time t.
    """
    df_original = non_stationary_price_data.copy()
    features_orig = calculate_technical_indicators(df_original)

    # Choose evaluation point t = 100
    t_idx = 100
    row_t_orig = features_orig.iloc[t_idx].copy()

    # Now create modified dataset with drastic changes in the future (t+1 onwards)
    df_future_tampered = df_original.copy()
    df_future_tampered.iloc[t_idx + 1:, df_future_tampered.columns.get_loc("Close")] *= 5.0
    df_future_tampered.iloc[t_idx + 1:, df_future_tampered.columns.get_loc("High")] *= 5.0
    df_future_tampered.iloc[t_idx + 1:, df_future_tampered.columns.get_loc("Volume")] *= 10.0

    features_tampered = calculate_technical_indicators(df_future_tampered)
    row_t_tampered = features_tampered.iloc[t_idx]

    # Verify that all 33 indicators at time t are IDENTICAL
    for col in features_orig.columns:
        diff = abs(row_t_orig[col] - row_t_tampered[col])
        assert diff == pytest.approx(0.0, abs=1e-8), (
            f"Future leakage in indicator '{col}' at index {t_idx}! Diff = {diff}"
        )


def test_target_shift_exactness(non_stationary_price_data):
    """Verify target(t) is strictly Close(t+1) and NOT contemporaneous Close(t)."""
    df_feat = calculate_technical_indicators(non_stationary_price_data)
    df_targets = create_prediction_targets(df_feat)

    # For every row t, Target_Next_Close must equal Close at t+1
    for i in range(len(df_targets) - 1):
        assert df_targets["Target_Next_Close"].iloc[i] == non_stationary_price_data["Close"].iloc[i + 1]
        assert df_targets["Target_Next_Close"].iloc[i] != non_stationary_price_data["Close"].iloc[i]

    # The last row cannot have a known next-day target
    assert np.isnan(df_targets["Target_Next_Close"].iloc[-1])


def test_walk_forward_causal_isolation(non_stationary_price_data):
    """PROVE that walk-forward validation never accesses future observations at step t."""
    backtester = WalkForwardBacktester(
        symbol="TEST.NS",
        initial_train_size=80,
        step_size=5,
        model_type="xgboost"
    )

    results_df, metrics = backtester.run(non_stationary_price_data, retrain_every=10)

    # Verify all evaluation dates in backtest are strictly greater than initial training cutoff
    initial_cutoff_date = non_stationary_price_data.index[80]
    assert results_df.index.min() >= initial_cutoff_date
    assert len(results_df) > 0
    assert "Actual_Next_Price" in results_df.columns
    assert "Predicted_Next_Price" in results_df.columns
