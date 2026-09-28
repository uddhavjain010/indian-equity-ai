"""Quantitative Signal Generation Engine.

Produces transparent, rule-based research signals (BUY / HOLD / SELL)
by fusing multi-model ML price forecasts with technical momentum, trend,
and volatility filters.

Signals are strictly educational/research-oriented and do not constitute
financial or trading advice.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from config.settings import SignalConfig
from src.utils.logger import get_logger

logger = get_logger("signal_engine")


@dataclass
class MarketSignal:
    """Encapsulates generated signal decision, indicators, and rule breakdown."""
    signal: str                        # "BUY", "HOLD", or "SELL"
    symbol: str
    current_price: float
    predicted_price: float
    predicted_return_pct: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    ema_20: float
    price_vs_ema: str                  # "Above", "Below", "Equal"
    model_agreement: str               # "Strong (Both Agree)", "Partial", "Single Model"
    reasons: List[str]
    disclaimer: str = "Model Signal — Educational / Research Use Only. Not Investment Advice."


class SignalEngine:
    """Generates explainable quantitative signals combining ML predictions and technical filters."""

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()

    def generate_signal(
        self,
        symbol: str,
        current_price: float,
        lstm_prediction: Optional[float] = None,
        xgb_prediction: Optional[float] = None,
        latest_features: Optional[pd.Series] = None,
    ) -> MarketSignal:
        """Compute transparent educational BUY/HOLD/SELL signal based on multi-indicator consensus.

        Args:
            symbol: Ticker symbol.
            current_price: Latest market price at timestamp t.
            lstm_prediction: Predicted next-day close from LSTM (or None).
            xgb_prediction: Predicted next-day close from XGBoost (or None).
            latest_features: Series containing latest row of engineered technical indicators.

        Returns:
            MarketSignal object.
        """
        reasons: List[str] = []
        
        # Determine aggregate predicted price and model agreement
        preds = []
        if lstm_prediction is not None and not np.isnan(lstm_prediction):
            preds.append(("LSTM", float(lstm_prediction)))
        if xgb_prediction is not None and not np.isnan(xgb_prediction):
            preds.append(("XGBoost", float(xgb_prediction)))

        if not preds:
            raise ValueError("At least one valid model prediction (LSTM or XGBoost) is required.")

        if len(preds) == 2:
            pred_lstm = preds[0][1]
            pred_xgb = preds[1][1]
            avg_pred = (pred_lstm + pred_xgb) / 2.0
            
            # Check agreement in direction
            lstm_dir = "UP" if pred_lstm > current_price else "DOWN"
            xgb_dir = "UP" if pred_xgb > current_price else "DOWN"
            if lstm_dir == xgb_dir:
                agreement = f"Concordant: Both models forecast {lstm_dir}"
                models_agree = True
            else:
                agreement = f"Conflicting: LSTM says {lstm_dir}, XGBoost says {xgb_dir}"
                models_agree = False
        else:
            avg_pred = preds[0][1]
            model_name = preds[0][0]
            agreement = f"Single Model ({model_name})"
            models_agree = True

        pred_return = (avg_pred - current_price) / current_price
        pred_return_pct = pred_return * 100.0

        # Technical Indicators extraction
        rsi = float(latest_features.get("RSI_14", 50.0)) if latest_features is not None else 50.0
        macd = float(latest_features.get("MACD", 0.0)) if latest_features is not None else 0.0
        macd_signal = float(latest_features.get("MACD_Signal", 0.0)) if latest_features is not None else 0.0
        macd_hist = float(latest_features.get("MACD_Hist", 0.0)) if latest_features is not None else 0.0
        ema_20 = float(latest_features.get("EMA_20", current_price)) if latest_features is not None else current_price

        price_above_ema = current_price >= ema_20
        price_vs_ema = "Above" if current_price > ema_20 else ("Below" if current_price < ema_20 else "Equal")
        macd_bullish = (macd > macd_signal) or (macd_hist > 0)
        macd_bearish = (macd < macd_signal) or (macd_hist < 0)

        # Rule evaluation
        # Condition 1: Return Threshold
        return_buy_ok = pred_return >= self.config.min_predicted_return_buy
        return_sell_ok = pred_return <= self.config.min_predicted_return_sell

        # Condition 2: Momentum (RSI)
        rsi_buy_ok = rsi < self.config.rsi_overbought  # Not overbought
        rsi_sell_ok = rsi > self.config.rsi_oversold   # Not oversold

        signal = self.config.SIGNAL_HOLD

        # Evaluate BUY Conditions
        if return_buy_ok and macd_bullish and price_above_ema and rsi_buy_ok and models_agree:
            signal = self.config.SIGNAL_BUY
            reasons.append(f"Predicted return (+{pred_return_pct:.2f}%) exceeds buy threshold (+{self.config.min_predicted_return_buy*100:.1f}%).")
            reasons.append("MACD is in bullish alignment (MACD line above Signal).")
            reasons.append(f"Current price (₹{current_price:.2f}) is trading above EMA 20 (₹{ema_20:.2f}).")
            reasons.append(f"RSI ({rsi:.1f}) is within acceptable range below overbought threshold ({self.config.rsi_overbought}).")
            if len(preds) == 2:
                reasons.append("Both LSTM and XGBoost models agree on upward direction.")

        # Evaluate SELL Conditions
        elif return_sell_ok and macd_bearish and (not price_above_ema) and rsi_sell_ok and models_agree:
            signal = self.config.SIGNAL_SELL
            reasons.append(f"Predicted return ({pred_return_pct:.2f}%) falls below sell threshold ({self.config.min_predicted_return_sell*100:.1f}%).")
            reasons.append("MACD is in bearish alignment (MACD line below Signal).")
            reasons.append(f"Current price (₹{current_price:.2f}) is trading below EMA 20 (₹{ema_20:.2f}).")
            reasons.append(f"RSI ({rsi:.1f}) is within acceptable range above oversold threshold ({self.config.rsi_oversold}).")
            if len(preds) == 2:
                reasons.append("Both LSTM and XGBoost models agree on downward direction.")

        # Otherwise HOLD
        else:
            signal = self.config.SIGNAL_HOLD
            if not models_agree:
                reasons.append("Model consensus is conflicting between LSTM and XGBoost.")
            elif abs(pred_return) < self.config.min_predicted_return_buy:
                reasons.append(f"Forecast return ({pred_return_pct:+.2f}%) is within neutral band (±{self.config.min_predicted_return_buy*100:.1f}%).")
            if not rsi_buy_ok and pred_return > 0:
                reasons.append(f"RSI ({rsi:.1f}) indicates an overbought condition; holding off on new entry.")
            if not rsi_sell_ok and pred_return < 0:
                reasons.append(f"RSI ({rsi:.1f}) indicates an oversold condition; holding off on new exit.")
            if not macd_bullish and pred_return > 0:
                reasons.append("MACD trend filter does not confirm bullish momentum.")
            if not macd_bearish and pred_return < 0:
                reasons.append("MACD trend filter does not confirm bearish momentum.")
            if not reasons:
                reasons.append("Technical conditions do not satisfy confluence entry/exit criteria.")

        logger.info("Signal for %s: %s (Reasons: %d)", symbol, signal, len(reasons))

        return MarketSignal(
            signal=signal,
            symbol=symbol,
            current_price=round(current_price, 2),
            predicted_price=round(avg_pred, 2),
            predicted_return_pct=round(pred_return_pct, 2),
            rsi=round(rsi, 2),
            macd=round(macd, 2),
            macd_signal=round(macd_signal, 2),
            macd_hist=round(macd_hist, 2),
            ema_20=round(ema_20, 2),
            price_vs_ema=price_vs_ema,
            model_agreement=agreement,
            reasons=reasons,
        )
