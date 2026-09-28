"""Walk-Forward Backtesting Engine for Financial Machine Learning.

Implements rigorous, causal walk-forward testing (rolling or expanding windows)
without data leakage. Simulates real-world sequential forecasting where models
are fitted strictly on past observations to predict the unseen next period.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.settings import BACKTEST_DATA_DIR, ModelConfig
from src.features.technical_indicators import (
    calculate_technical_indicators,
    create_prediction_targets,
    get_feature_column_names,
)
from src.evaluation.metrics import calculate_metrics
from src.models.xgboost_model import StockXGBoostModel
from src.utils.logger import get_logger

logger = get_logger("walk_forward")


class WalkForwardBacktester:
    """Walk-forward time-series backtester ensuring strict causal temporal boundaries."""

    def __init__(
        self,
        symbol: str,
        initial_train_size: int = 150,
        step_size: int = 1,
        model_type: str = "xgboost",
        config: Optional[ModelConfig] = None,
    ):
        self.symbol = symbol
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.model_type = model_type.lower()
        self.config = config or ModelConfig()
        self.results_df: Optional[pd.DataFrame] = None
        self.summary_metrics: Optional[Dict[str, Any]] = None

    def run(
        self,
        df: pd.DataFrame,
        retrain_every: int = 20,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute walk-forward validation across the dataset.

        Args:
            df: Cleaned OHLCV DataFrame.
            retrain_every: Frequency of full model re-fitting (e.g. every 20 days/monthly).

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Detailed per-step predictions and summary metrics.
        """
        logger.info(
            "Starting Walk-Forward Backtest for %s | Initial Train=%d | Step=%d | Retrain Every=%d",
            self.symbol, self.initial_train_size, self.step_size, retrain_every
        )

        # 1. Feature Engineering
        df_feat = calculate_technical_indicators(df)
        df_full = create_prediction_targets(df_feat)
        feature_cols = get_feature_column_names(df_full)

        # Drop warm-up rows and the last row (where Target_Next_Close is NaN)
        data = df_full.dropna(subset=feature_cols + ["Target_Next_Close"]).copy()

        total_bars = len(data)
        if total_bars <= self.initial_train_size:
            raise ValueError(
                f"Dataset length ({total_bars}) is too short for initial training size ({self.initial_train_size})."
            )

        records = []
        model = None
        scaler = None

        # Walk forward from initial_train_size to the end
        for t in range(self.initial_train_size, total_bars, self.step_size):
            # Training slice: strictly 0 up to t
            train_slice = data.iloc[:t]
            # Test step: index t (predicting t+1 price using features at t)
            test_row = data.iloc[[t]]

            # Re-train model periodically or on first step
            if model is None or (t - self.initial_train_size) % retrain_every == 0:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(train_slice[feature_cols])
                y_train = train_slice["Target_Next_Close"].values

                # Fit XGBoost model on current expanding training window
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    n_estimators=self.config.xgb_n_estimators,
                    max_depth=self.config.xgb_max_depth,
                    learning_rate=self.config.xgb_learning_rate,
                    subsample=self.config.xgb_subsample,
                    colsample_bytree=self.config.xgb_colsample_bytree,
                    random_state=self.config.xgb_random_state,
                    n_jobs=-1,
                )
                model.fit(X_train_scaled, y_train)

            # Transform test row using scaler fitted ONLY on historical data up to t
            X_test_scaled = scaler.transform(test_row[feature_cols])
            pred_price = float(model.predict(X_test_scaled)[0])

            actual_next_price = float(test_row["Target_Next_Close"].iloc[0])
            current_price = float(test_row["Close"].iloc[0])
            eval_date = test_row.index[0]

            pred_dir = 1 if pred_price > current_price else 0
            actual_dir = 1 if actual_next_price > current_price else 0
            err = pred_price - actual_next_price
            abs_err = abs(err)

            # Hypothetical strategy return: if predicted UP, capture actual day return; else 0
            actual_return = (actual_next_price - current_price) / current_price
            strat_return = actual_return if pred_dir == 1 else 0.0

            records.append({
                "Date": eval_date,
                "Symbol": self.symbol,
                "Model": self.model_type.upper(),
                "Current_Price": round(current_price, 2),
                "Actual_Next_Price": round(actual_next_price, 2),
                "Predicted_Next_Price": round(pred_price, 2),
                "Error": round(err, 2),
                "Abs_Error": round(abs_err, 2),
                "Predicted_Direction": pred_dir,
                "Actual_Direction": actual_dir,
                "Direction_Correct": int(pred_dir == actual_dir),
                "Market_Return": actual_return,
                "Strategy_Return": strat_return,
            })

        results_df = pd.DataFrame(records)
        results_df.set_index("Date", inplace=True)

        # Cumulative returns for hypothetical educational strategy vs buy-and-hold
        results_df["Cumulative_Market"] = (1.0 + results_df["Market_Return"]).cumprod()
        results_df["Cumulative_Strategy"] = (1.0 + results_df["Strategy_Return"]).cumprod()

        self.results_df = results_df

        # Calculate standard summary metrics
        metrics = calculate_metrics(
            y_true=results_df["Actual_Next_Price"],
            y_pred=results_df["Predicted_Next_Price"],
            current_prices=results_df["Current_Price"],
            model_name=f"{self.model_type.upper()}-WalkForward"
        )

        total_strat_return = float((results_df["Cumulative_Strategy"].iloc[-1] - 1.0) * 100.0) if len(results_df) > 0 else 0.0
        total_market_return = float((results_df["Cumulative_Market"].iloc[-1] - 1.0) * 100.0) if len(results_df) > 0 else 0.0

        metrics["Hypothetical_Strategy_Return (%)"] = round(total_strat_return, 2)
        metrics["Benchmark_Market_Return (%)"] = round(total_market_return, 2)
        metrics["Start_Date"] = str(results_df.index[0])[:10]
        metrics["End_Date"] = str(results_df.index[-1])[:10]

        self.summary_metrics = metrics
        logger.info(
            "Walk-Forward Backtest completed: %d steps | RMSE: %.2f | Direction Acc: %.2f%%",
            len(results_df), metrics["RMSE"], metrics["Directional_Accuracy (%)"]
        )

        return results_df, metrics

    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """Save backtesting records to CSV."""
        if self.results_df is None:
            raise RuntimeError("No backtest results to save. Call run() first.")

        target_dir = output_dir or BACKTEST_DATA_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        clean_sym = self.symbol.replace(".", "_")
        filepath = target_dir / f"{clean_sym}_{self.model_type}_walk_forward.csv"

        self.results_df.to_csv(filepath, index=True)
        logger.info("Saved walk-forward backtest data to %s", filepath)
        return filepath
