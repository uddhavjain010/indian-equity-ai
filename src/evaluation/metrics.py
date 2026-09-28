"""Evaluation Metrics for Financial Time-Series Models.

Calculates RMSE, MAE, MAPE, and Directional Accuracy.
Directional Accuracy evaluates whether the model correctly anticipates
the sign of the price movement relative to current close.
"""
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    current_prices: Union[np.ndarray, pd.Series],
    model_name: str = "Model"
) -> Dict[str, Any]:
    """Calculate key financial regression and directional accuracy metrics.

    Args:
        y_true: Ground truth next-period closing prices.
        y_pred: Model predicted next-period closing prices.
        current_prices: Current period closing prices at prediction time t.
        model_name: Identifier for reporting.

    Returns:
        Dict containing:
            - Model: str
            - RMSE: float
            - MAE: float
            - MAPE (%): float
            - Directional_Accuracy (%): float
            - Total_Predictions: int
    """
    y_t = np.asarray(y_true, dtype=float).flatten()
    y_p = np.asarray(y_pred, dtype=float).flatten()
    c_p = np.asarray(current_prices, dtype=float).flatten()

    if len(y_t) != len(y_p) or len(y_t) != len(c_p):
        raise ValueError(
            f"Mismatched array shapes: y_true={len(y_t)}, y_pred={len(y_p)}, current={len(c_p)}"
        )

    # Filter any NaNs
    valid_mask = (~np.isnan(y_t)) & (~np.isnan(y_p)) & (~np.isnan(c_p))
    y_t = y_t[valid_mask]
    y_p = y_p[valid_mask]
    c_p = c_p[valid_mask]

    if len(y_t) == 0:
        return {
            "Model": model_name,
            "RMSE": 0.0,
            "MAE": 0.0,
            "MAPE (%)": 0.0,
            "Directional_Accuracy (%)": 0.0,
            "Total_Predictions": 0,
        }

    # Continuous Price Metrics
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    mape = np.mean(np.abs((y_t - y_p) / np.where(y_t == 0, 1e-6, y_t))) * 100.0

    # Directional Accuracy (t to t+1)
    # Predicted Up/Down: y_pred > current_price
    # Actual Up/Down:    y_true > current_price
    predicted_dir = (y_p > c_p).astype(int)
    actual_dir = (y_t > c_p).astype(int)
    directional_accuracy = np.mean(predicted_dir == actual_dir) * 100.0

    return {
        "Model": model_name,
        "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 2),
        "MAPE (%)": round(float(mape), 2),
        "Directional_Accuracy (%)": round(float(directional_accuracy), 2),
        "Total_Predictions": int(len(y_t)),
    }


def compare_models(
    y_true: Union[np.ndarray, pd.Series],
    lstm_preds: Union[np.ndarray, pd.Series],
    xgb_preds: Union[np.ndarray, pd.Series],
    current_prices: Union[np.ndarray, pd.Series]
) -> pd.DataFrame:
    """Produce a standardized comparison table across LSTM and XGBoost."""
    metrics_lstm = calculate_metrics(y_true, lstm_preds, current_prices, model_name="LSTM")
    metrics_xgb = calculate_metrics(y_true, xgb_preds, current_prices, model_name="XGBoost")

    df_comp = pd.DataFrame([metrics_lstm, metrics_xgb])
    return df_comp
