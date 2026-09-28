"""Long Short-Term Memory (LSTM) Model Implementation for Stock Price Prediction.

Uses TensorFlow/Keras with stacked LSTM layers, dropout regularization,
EarlyStopping, ModelCheckpoint, and anti-leakage inverse scaling.
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from config.settings import LSTM_MODEL_DIR, ModelConfig
from src.features.preprocessor import ProcessedDataset
from src.utils.logger import get_logger

logger = get_logger("lstm_model")


class StockLSTMModel:
    """Configurable Stacked LSTM Deep Learning Model for next-period price regression."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model: Optional[Sequential] = None
        self.feature_scaler: Optional[Any] = None
        self.target_scaler: Optional[Any] = None
        self.feature_names: list = []
        self.symbol: str = ""
        self.history: Optional[Dict[str, list]] = None

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Construct the Keras sequential architecture.

        Input -> LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16) -> Dense(1)
        """
        model = Sequential([
            Input(shape=input_shape),
            LSTM(
                units=self.config.lstm_units_1,
                return_sequences=True,
                name="lstm_layer_1"
            ),
            Dropout(self.config.dropout_rate, name="dropout_1"),
            LSTM(
                units=self.config.lstm_units_2,
                return_sequences=False,
                name="lstm_layer_2"
            ),
            Dropout(self.config.dropout_rate, name="dropout_2"),
            Dense(16, activation="relu", name="dense_intermediate"),
            Dense(1, name="output_price")
        ])

        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        self.model = model
        return model

    def train(
        self,
        dataset: ProcessedDataset,
        symbol: str = "STOCK",
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 1
    ) -> Dict[str, list]:
        """Train the LSTM model chronologically using pre-split data with early stopping.

        Args:
            dataset: ProcessedDataset instance.
            symbol: Ticker symbol string.
            epochs: Optional epoch override.
            batch_size: Optional batch size override.
            verbose: Keras verbosity level.

        Returns:
            Dict containing training and validation loss history.
        """
        self.symbol = symbol
        self.feature_scaler = dataset.feature_scaler
        self.target_scaler = dataset.target_scaler
        self.feature_names = dataset.feature_names

        X_train_seq = dataset.X_train_seq
        y_train_seq = dataset.y_train_seq
        X_val_seq = dataset.X_val_seq
        y_val_seq = dataset.y_val_seq

        num_samples, lookback, num_features = X_train_seq.shape
        logger.info(
            "Building LSTM with lookback=%d, features=%d for %s (train samples=%d)",
            lookback, num_features, symbol, num_samples
        )

        self.build_model(input_shape=(lookback, num_features))

        # Checkpoint directory
        save_path = LSTM_MODEL_DIR / f"{symbol.replace('.', '_')}_best.keras"

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            ModelCheckpoint(
                filepath=str(save_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=0
            )
        ]

        actual_epochs = epochs or self.config.epochs
        actual_batch = batch_size or self.config.batch_size

        logger.info("Training LSTM for up to %d epochs...", actual_epochs)
        fit_history = self.model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=actual_epochs,
            batch_size=actual_batch,
            callbacks=callbacks,
            shuffle=False,  # Strictly preserve temporal mini-batch sequence
            verbose=verbose
        )

        self.history = fit_history.history
        logger.info("LSTM training completed. Best Val Loss: %.6f", min(self.history["val_loss"]))
        return self.history

    def predict_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        """Predict prices given 3D input sequences and inverse-transform to original ₹ scale."""
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
        if self.target_scaler is None:
            raise RuntimeError("Target scaler missing for inverse transformation.")

        scaled_preds = self.model.predict(X_seq, verbose=0)
        # Invert scaling back to actual rupee prices
        unscaled_preds = self.target_scaler.inverse_transform(scaled_preds).flatten()
        return unscaled_preds

    def predict_next_day(self, recent_feature_df: pd.DataFrame) -> float:
        """Predict the next trading day's closing price from the most recent historical window.

        Args:
            recent_feature_df: DataFrame containing at least `lookback` rows with all features.

        Returns:
            float: Predicted next-day closing price in INR (₹).
        """
        if len(recent_feature_df) < self.config.lookback:
            raise ValueError(
                f"Need at least {self.config.lookback} rows of feature history, got {len(recent_feature_df)}"
            )

        # Take the most recent lookback rows
        window = recent_feature_df[self.feature_names].iloc[-self.config.lookback:].copy()
        scaled_window = self.feature_scaler.transform(window)
        # Reshape to (1, lookback, num_features)
        X_live = np.expand_dims(scaled_window, axis=0)

        preds = self.predict_sequences(X_live)
        return float(preds[0])

    def save(self, directory: Optional[Path] = None, prefix: Optional[str] = None) -> Path:
        """Persist model, scalers, and metadata to disk."""
        target_dir = directory or LSTM_MODEL_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        sym_clean = (prefix or self.symbol or "default").replace(".", "_")

        model_path = target_dir / f"{sym_clean}_model.keras"
        self.model.save(str(model_path))

        scaler_path = target_dir / f"{sym_clean}_scalers.joblib"
        joblib.dump(
            {
                "feature_scaler": self.feature_scaler,
                "target_scaler": self.target_scaler,
                "feature_names": self.feature_names,
            },
            scaler_path
        )

        metadata_path = target_dir / f"{sym_clean}_meta.json"
        meta = {
            "symbol": self.symbol,
            "lookback": self.config.lookback,
            "lstm_units_1": self.config.lstm_units_1,
            "lstm_units_2": self.config.lstm_units_2,
            "dropout_rate": self.config.dropout_rate,
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved LSTM artifacts for %s to %s", self.symbol, target_dir)
        return model_path

    def load(self, directory: Optional[Path] = None, prefix: str = "RELIANCE_NS") -> None:
        """Load model, scalers, and configuration from disk."""
        target_dir = directory or LSTM_MODEL_DIR
        sym_clean = prefix.replace(".", "_")

        model_path = target_dir / f"{sym_clean}_model.keras"
        if not model_path.exists():
            # Check for _best.keras fallback
            best_path = target_dir / f"{sym_clean}_best.keras"
            if best_path.exists():
                model_path = best_path
            else:
                raise FileNotFoundError(f"LSTM model not found at {model_path}")

        self.model = load_model(str(model_path))

        scaler_path = target_dir / f"{sym_clean}_scalers.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler bundle not found at {scaler_path}")

        scalers_bundle = joblib.load(scaler_path)
        self.feature_scaler = scalers_bundle["feature_scaler"]
        self.target_scaler = scalers_bundle["target_scaler"]
        self.feature_names = scalers_bundle["feature_names"]
        self.symbol = prefix

        logger.info("Loaded LSTM model and scalers for %s", prefix)
