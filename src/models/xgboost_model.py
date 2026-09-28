"""XGBoost Regression Model for Stock Price Prediction.

Leverages gradient boosted decision trees over engineered technical indicators
and lag features with chronological validation and anti-leakage safeguards.
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

from config.settings import XGBOOST_MODEL_DIR, ModelConfig
from src.features.preprocessor import ProcessedDataset
from src.utils.logger import get_logger

logger = get_logger("xgboost_model")


class StockXGBoostModel:
    """Gradient Boosted Trees Regression Model for next-day Indian equity price prediction."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model: Optional[XGBRegressor] = None
        self.feature_names: List[str] = []
        self.feature_scaler: Optional[Any] = None
        self.symbol: str = ""

    def build_model(self) -> XGBRegressor:
        """Instantiate XGBRegressor with robust quantitative finance parameters."""
        self.model = XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            random_state=self.config.xgb_random_state,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
        )
        return self.model

    def train(
        self,
        dataset: ProcessedDataset,
        symbol: str = "STOCK",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Train XGBoost chronologically with validation evaluation.

        Args:
            dataset: ProcessedDataset instance containing X_train, y_train, X_val, y_val.
            symbol: Stock symbol.
            verbose: Verbosity boolean.

        Returns:
            Dict containing evaluation results.
        """
        self.symbol = symbol
        self.feature_names = dataset.feature_names
        self.feature_scaler = dataset.feature_scaler

        # In XGBoost, tree models are scale-invariant, but using scaled or unscaled features consistently is crucial.
        # We use the scaled features prepared by the anti-leakage scaler for absolute feature alignment.
        X_train = dataset.feature_scaler.transform(dataset.X_train)
        y_train = dataset.y_train.values

        X_val = dataset.feature_scaler.transform(dataset.X_val)
        y_val = dataset.y_val.values

        logger.info(
            "Training XGBoost on %d samples (val samples=%d) with %d features for %s",
            len(X_train), len(X_val), len(self.feature_names), symbol
        )

        self.build_model()
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=verbose
        )

        evals_result = self.model.evals_result()
        logger.info("XGBoost training complete.")
        return evals_result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict next-day prices given a DataFrame of features.

        Args:
            X: DataFrame with column names matching self.feature_names.

        Returns:
            np.ndarray: Predicted prices in ₹.
        """
        if self.model is None:
            raise RuntimeError("XGBoost model is not trained or loaded.")

        X_aligned = X[self.feature_names]
        X_scaled = self.feature_scaler.transform(X_aligned)
        return self.model.predict(X_scaled)

    def predict_next_day(self, recent_feature_df: pd.DataFrame) -> float:
        """Predict the single next-day closing price from the most recent feature bar."""
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        latest_row = recent_feature_df[self.feature_names].iloc[[-1]]
        preds = self.predict(latest_row)
        return float(preds[0])

    def get_feature_importances(self) -> pd.DataFrame:
        """Return sorted DataFrame of feature importances."""
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame()
        df_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": self.model.feature_importances_
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        return df_imp

    def save(self, directory: Optional[Path] = None, prefix: Optional[str] = None) -> Path:
        """Persist XGBoost model and artifacts to disk."""
        target_dir = directory or XGBOOST_MODEL_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        sym_clean = (prefix or self.symbol or "default").replace(".", "_")

        model_path = target_dir / f"{sym_clean}_model.joblib"
        bundle = {
            "model": self.model,
            "feature_scaler": self.feature_scaler,
            "feature_names": self.feature_names,
            "symbol": self.symbol,
            "config": self.config,
        }
        joblib.dump(bundle, model_path)

        metadata_path = target_dir / f"{sym_clean}_meta.json"
        meta = {
            "symbol": self.symbol,
            "n_estimators": self.config.xgb_n_estimators,
            "max_depth": self.config.xgb_max_depth,
            "learning_rate": self.config.xgb_learning_rate,
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved XGBoost model artifacts for %s to %s", self.symbol, model_path)
        return model_path

    def load(self, directory: Optional[Path] = None, prefix: str = "RELIANCE_NS") -> None:
        """Load XGBoost model and configuration from disk."""
        target_dir = directory or XGBOOST_MODEL_DIR
        sym_clean = prefix.replace(".", "_")
        model_path = target_dir / f"{sym_clean}_model.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found at {model_path}")

        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_scaler = bundle["feature_scaler"]
        self.feature_names = bundle["feature_names"]
        self.symbol = bundle.get("symbol", prefix)

        logger.info("Loaded XGBoost model for %s from %s", prefix, model_path)
