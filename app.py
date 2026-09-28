"""Streamlit Dashboard: Real-Time Indian Stock Price Prediction & Signal Platform.

End-to-End Quantitative Machine Learning Dashboard for NSE/BSE Equities
using Stacked LSTM Networks and XGBoost Regressors with Walk-Forward Backtesting.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set base path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.stocks import NIFTY_50_UNIVERSE, get_stock_info, get_all_symbols, get_display_name
from config.settings import AppConfig, ModelConfig, SignalConfig, RAW_DATA_DIR, BACKTEST_DATA_DIR
from src.data import get_data_provider
from src.data.download_data import download_and_save_stock_data, sanitize_filename
from src.features.technical_indicators import calculate_technical_indicators
from src.features.preprocessor import preprocess_stock_data, integrate_live_tick_into_history
from src.models.xgboost_model import StockXGBoostModel
from src.models.lstm_model import StockLSTMModel
from src.evaluation.metrics import calculate_metrics, compare_models
from src.signals.signal_engine import SignalEngine
from src.backtesting.walk_forward import WalkForwardBacktester
from src.utils.logger import get_logger

logger = get_logger("streamlit_app")

# Page configuration
st.set_page_config(
    page_title="Indian Stock ML Predictor & Signal Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Number and Price Formatting Helpers
def format_inr(val: Optional[float]) -> str:
    """Format numeric rupee value with currency symbol and 2 decimal points."""
    if val is None or np.isnan(val):
        return "₹0.00"
    return f"₹{val:,.2f}"

def format_volume_num(vol: Optional[float]) -> str:
    """Format traded volume cleanly with human-readable Indian crore / lakh / million suffixes."""
    if vol is None or np.isnan(vol) or vol <= 0:
        return "0"
    v = int(vol)
    if v >= 10_000_000:
        return f"{v / 10_000_000:.2f}M ({v:,})"
    elif v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M ({v:,})"
    elif v >= 100_000:
        return f"{v / 100_000:.2f}L ({v:,})"
    elif v >= 1_000:
        return f"{v / 1_000:.1f}K ({v:,})"
    return f"{v:,}"

# Custom Styling (Dark Glassmorphism UI with fixed number wrapping)
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
        min-height: 110px;
    }
    .metric-label {
        font-size: 0.80rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 4px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-variant-numeric: tabular-nums;
    }
    .metric-delta-pos {
        font-size: 0.88rem;
        font-weight: 600;
        color: #10b981;
        white-space: nowrap;
    }
    .metric-delta-neg {
        font-size: 0.88rem;
        font-weight: 600;
        color: #ef4444;
        white-space: nowrap;
    }
    
    /* Signal Badges */
    .signal-box-buy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(5, 150, 105, 0.1));
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .signal-box-sell {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .signal-box-hold {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(217, 119, 6, 0.1));
        border: 2px solid #f59e0b;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .signal-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin: 0;
    }
    .signal-subtitle {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-top: 6px;
        font-style: italic;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# Caching layer for market data
@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(symbol: str, period: str, interval: str, force_refresh: bool = False):
    """Cached loader for validated historical OHLCV data."""
    return download_and_save_stock_data(
        symbol=symbol,
        period=period,
        interval=interval,
        force_refresh=force_refresh
    )


@st.cache_data(ttl=30, show_spinner=False)
def load_latest_quote(symbol: str):
    """Cached loader for latest market quote from active provider."""
    provider = get_data_provider()
    return provider.fetch_latest_quote(symbol)


def get_cached_models(symbol: str):
    """Load pre-trained models from disk if available."""
    xgb_model = StockXGBoostModel()
    lstm_model = StockLSTMModel()
    
    xgb_loaded = False
    lstm_loaded = False
    
    try:
        xgb_model.load(prefix=symbol)
        xgb_loaded = True
    except Exception:
        pass

    try:
        lstm_model.load(prefix=symbol)
        lstm_loaded = True
    except Exception:
        pass

    return (xgb_model if xgb_loaded else None), (lstm_model if lstm_loaded else None)


# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
logo_file = Path(__file__).resolve().parent / "assets" / "logo.svg"
if logo_file.exists():
    st.sidebar.image(str(logo_file), width=72)
else:
    st.sidebar.markdown("### 📈 **Indian Equity AI**")

st.sidebar.title("Indian Equity AI")
st.sidebar.caption("NSE / BSE Machine Learning Platform")

all_symbols = get_all_symbols()
symbol_options = {sym: get_display_name(sym) for sym in all_symbols}

selected_symbol = st.sidebar.selectbox(
    "Select Stock",
    options=all_symbols,
    format_func=lambda s: symbol_options.get(s, s),
    index=0
)

stock_info = get_stock_info(selected_symbol)

# Exchange Selector
exchange_choice = st.sidebar.radio(
    "Exchange",
    options=["NSE (.NS)", "BSE (.BO)"],
    index=0,
    horizontal=True
)

if exchange_choice == "BSE (.BO)" and stock_info and stock_info.get("bse_symbol"):
    active_symbol = stock_info["bse_symbol"]
    active_exchange = "BSE"
else:
    active_symbol = selected_symbol
    active_exchange = "NSE"

selected_period = st.sidebar.selectbox(
    "Historical Period",
    options=["1y", "2y", "5y", "10y"],
    index=1
)

selected_model_type = st.sidebar.selectbox(
    "Model Evaluation",
    options=["Both (LSTM & XGBoost)", "XGBoost Only", "LSTM Only"],
    index=0
)

# Buttons
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    btn_refresh = st.button("🔄 Refresh Data", use_container_width=True)
with col_btn2:
    btn_train = st.button("⚡ Retrain Models", use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown(
    """
    **Disclaimers:**
    - ⚠️ For **academic / research** use only.
    - ⚠️ Signals are **model-generated hypothesis**.
    - ⚠️ Not financial or investment advice.
    """
)


# ==========================================
# DATA LOADING & PREPARATION
# ==========================================
with st.spinner("Fetching market data and calculating technical features..."):
    try:
        quote = load_latest_quote(active_symbol)
    except Exception as e:
        logger.warning("Quote fetch error: %s", e)
        quote = {
            "symbol": active_symbol,
            "current_price": 0.0,
            "previous_close": 0.0,
            "change": 0.0,
            "change_pct": 0.0,
            "volume": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "data_status": "LATEST AVAILABLE — yfinance",
            "is_live": False,
        }

    try:
        df_raw, val_report = load_market_data(
            symbol=active_symbol,
            period=selected_period,
            interval="1d",
            force_refresh=btn_refresh
        )
    except Exception as e:
        st.error(f"Failed to fetch market data for {active_symbol}: {e}")
        st.stop()

# Synchronize latest quote price if valid
if quote["current_price"] == 0.0 and len(df_raw) > 0:
    quote["current_price"] = float(df_raw["Close"].iloc[-1])
    quote["previous_close"] = float(df_raw["Close"].iloc[-2]) if len(df_raw) > 1 else quote["current_price"]
    quote["change"] = quote["current_price"] - quote["previous_close"]
    quote["change_pct"] = (quote["change"] / quote["previous_close"]) * 100.0 if quote["previous_close"] else 0.0

# LIVE PRICE PIPELINE: Safely integrate live quote into historical daily series
# Maintains complete feature schema alignment without fabricating single-tick daily bars
df_feat = integrate_live_tick_into_history(df_raw, quote["current_price"], quote.get("volume"))


# Check/Train models if requested
xgb_model, lstm_model = get_cached_models(active_symbol)

if btn_train or (xgb_model is None and "XGBoost" in selected_model_type):
    with st.spinner(f"Training ML models on {len(df_raw)} trading bars for {active_symbol}..."):
        try:
            cfg = ModelConfig()
            dataset = preprocess_stock_data(df_raw, config=cfg)
            
            # Train XGBoost
            xgb_model = StockXGBoostModel(config=cfg)
            xgb_model.train(dataset, symbol=active_symbol)
            xgb_model.save()
            
            # Train LSTM if requested
            if "LSTM" in selected_model_type or "Both" in selected_model_type:
                lstm_model = StockLSTMModel(config=cfg)
                lstm_model.train(dataset, symbol=active_symbol, epochs=15, verbose=0)
                lstm_model.save()

            st.sidebar.success("✓ Models trained and cached successfully!")
        except Exception as e:
            st.sidebar.error(f"Training error: {e}")


# ==========================================
# MAIN APPLICATION TABS
# ==========================================
tab_dashboard, tab_perf, tab_backtest, tab_about = st.tabs([
    "📈 Live Dashboard",
    "🤖 Model Performance",
    "🔄 Walk-Forward Backtesting",
    "ℹ️ Methodology & About"
])


# -----------------------------------------------------------------------------
# TAB 1: LIVE DASHBOARD
# -----------------------------------------------------------------------------
with tab_dashboard:
    # DATA SOURCE STATUS & HEADER
    is_live = quote.get("is_live", False)
    status_color = "#10b981" if is_live else "#f59e0b"
    status_icon = "🟢" if is_live else "🟡"
    status_text = quote.get("data_status", "LATEST AVAILABLE — yfinance")

    st.subheader(f"{stock_info['company_name'] if stock_info else active_symbol} ({active_symbol})")

    col_stat1, col_stat2 = st.columns([1.8, 1.2])
    with col_stat1:
        st.markdown(
            f"""<div style="background: rgba(255, 255, 255, 0.03); border-left: 4px solid {status_color}; padding: 8px 14px; border-radius: 6px; margin-bottom: 12px;">
                <span style="font-weight: 600; color: #94a3b8; font-size: 0.85rem;">DATA SOURCE:</span> 
                <span style="color: {status_color}; font-weight: 700; font-size: 0.95rem;">{status_icon} {status_text}</span>
            </div>""",
            unsafe_allow_html=True
        )
    with col_stat2:
        st.markdown(
            f"""<div style="background: rgba(255, 255, 255, 0.03); border-left: 4px solid #38bdf8; padding: 8px 14px; border-radius: 6px; margin-bottom: 12px; text-align: right;">
                <span style="color: #94a3b8; font-size: 0.82rem;">LAST UPDATED:</span> 
                <span style="color: #f8fafc; font-weight: 600; font-size: 0.90rem;">{quote.get('timestamp', 'N/A')}</span>
            </div>""",
            unsafe_allow_html=True
        )

    # TOP METRIC CARDS
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Latest Price</div>
                <div class="metric-value">{format_inr(quote['current_price'])}</div>
                <div class="{'metric-delta-pos' if quote['change'] >= 0 else 'metric-delta-neg'}">
                    {'▲' if quote['change'] >= 0 else '▼'} {format_inr(abs(quote['change']))} ({quote['change_pct']:+.2f}%)
                </div>
            </div>""",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Previous Close</div>
                <div class="metric-value">{format_inr(quote['previous_close'])}</div>
                <div class="metric-label" style="margin-top:4px;">Official Prev Close</div>
            </div>""",
            unsafe_allow_html=True
        )

    with c3:
        high_24h = float(df_raw["High"].iloc[-1])
        low_24h = float(df_raw["Low"].iloc[-1])
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Day Range</div>
                <div class="metric-value" style="font-size:1.35rem;">{format_inr(low_24h)} - {format_inr(high_24h)}</div>
                <div class="metric-label" style="margin-top:4px;">Low / High</div>
            </div>""",
            unsafe_allow_html=True
        )

    with c4:
        vol = quote["volume"] or int(df_raw["Volume"].iloc[-1])
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Traded Volume</div>
                <div class="metric-value">{format_volume_num(vol)}</div>
                <div class="metric-label" style="margin-top:4px;">Shares Traded</div>
            </div>""",
            unsafe_allow_html=True
        )

    with c5:
        rsi_val = float(df_feat["RSI_14"].iloc[-1])
        rsi_label = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">RSI (14)</div>
                <div class="metric-value">{rsi_val:.1f}</div>
                <div class="metric-label" style="margin-top:4px;">Condition: {rsi_label}</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PREDICTION & SIGNAL SECTION
    # --------------------------------------------------------
    st.markdown("### 🎯 Machine Learning Next-Period Forecast & Research Signal")

    curr_price = quote["current_price"]
    lstm_pred = None
    xgb_pred = None

    if xgb_model is not None:
        try:
            xgb_pred = xgb_model.predict_next_day(df_feat)
        except Exception as e:
            logger.warning("XGBoost prediction error: %s", e)

    if lstm_model is not None:
        try:
            lstm_pred = lstm_model.predict_next_day(df_feat)
        except Exception as e:
            logger.warning("LSTM prediction error: %s", e)

    if xgb_pred is None and lstm_pred is None:
        st.info("ℹ️ No trained model weights found for this stock. Click **'⚡ Retrain Models'** in the sidebar to train LSTM and XGBoost.")

    col_pred_l, col_pred_m, col_signal = st.columns([1.1, 1.1, 1.4])

    with col_pred_l:
        st.markdown("**🤖 XGBoost Regressor**")
        if xgb_pred is not None:
            xgb_change = xgb_pred - curr_price
            xgb_pct = (xgb_change / curr_price) * 100.0
            direction = "UP" if xgb_change >= 0 else "DOWN"
            st.metric(
                label="Predicted Next-Day Close",
                value=format_inr(xgb_pred),
                delta=f"{xgb_pct:+.2f}% ({direction})"
            )
            st.caption(f"Estimated Difference: {format_inr(xgb_change)} vs current {format_inr(curr_price)}")
        else:
            st.warning("XGBoost model not trained yet.")

    with col_pred_m:
        st.markdown("**🧠 Stacked LSTM Network**")
        if lstm_pred is not None:
            lstm_change = lstm_pred - curr_price
            lstm_pct = (lstm_change / curr_price) * 100.0
            direction = "UP" if lstm_change >= 0 else "DOWN"
            st.metric(
                label="Predicted Next-Day Close",
                value=format_inr(lstm_pred),
                delta=f"{lstm_pct:+.2f}% ({direction})"
            )
            st.caption(f"Estimated Difference: {format_inr(lstm_change)} vs current {format_inr(curr_price)}")
        else:
            st.warning("LSTM model not trained yet.")

    with col_signal:
        st.markdown("**⚡ Research / Educational Signal**")
        if xgb_pred is not None or lstm_pred is not None:
            engine = SignalEngine()
            signal_obj = engine.generate_signal(
                symbol=active_symbol,
                current_price=curr_price,
                lstm_prediction=lstm_pred,
                xgb_prediction=xgb_pred,
                latest_features=df_feat.iloc[-1]
            )

            box_class = (
                "signal-box-buy" if signal_obj.signal == "BUY"
                else ("signal-box-sell" if signal_obj.signal == "SELL" else "signal-box-hold")
            )
            color_hex = (
                "#10b981" if signal_obj.signal == "BUY"
                else ("#ef4444" if signal_obj.signal == "SELL" else "#f59e0b")
            )

            st.markdown(
                f"""<div class="{box_class}">
                    <div class="signal-title" style="color: {color_hex};">{signal_obj.signal}</div>
                    <div class="signal-subtitle">{signal_obj.disclaimer}</div>
                </div>""",
                unsafe_allow_html=True
            )

            with st.expander("🔍 Why this signal? View transparent rule breakdown", expanded=False):
                st.write(f"- **Consensus Forecast Return:** `{signal_obj.predicted_return_pct:+.2f}%`")
                st.write(f"- **Model Consensus:** `{signal_obj.model_agreement}`")
                st.write(f"- **Price vs EMA(20):** `{signal_obj.price_vs_ema}` (EMA: {format_inr(signal_obj.ema_20)})")
                st.write(f"- **RSI (14):** `{signal_obj.rsi:.1f}`")
                st.write(f"- **MACD Line / Signal / Hist:** `{signal_obj.macd:.2f} / {signal_obj.macd_signal:.2f} / {signal_obj.macd_hist:+.2f}`")
                st.markdown("**Rule triggers:**")
                for r in signal_obj.reasons:
                    st.markdown(f"• {r}")
        else:
            st.info("Train at least one model to generate signals.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # INTERACTIVE CANDLESTICK & INDICATOR CHARTS (PLOTLY)
    # --------------------------------------------------------
    st.markdown("### 📊 Interactive Price Action & Indicator Analysis")

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        subplot_titles=("Price & Moving Averages (INR ₹)", "Volume", "RSI (14)", "MACD (12, 26, 9)")
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_feat.index,
            open=df_feat["Open"],
            high=df_feat["High"],
            low=df_feat["Low"],
            close=df_feat["Close"],
            name="OHLC",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444"
        ),
        row=1, col=1
    )

    # Moving Average Overlays
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["SMA_20"], mode="lines", name="SMA 20", line=dict(color="#38bdf8", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["SMA_50"], mode="lines", name="SMA 50", line=dict(color="#fbbf24", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["EMA_20"], mode="lines", name="EMA 20", line=dict(color="#c084fc", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["BB_High"], mode="lines", name="BB Upper", line=dict(color="rgba(148, 163, 184, 0.4)", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["BB_Low"], mode="lines", name="BB Lower", line=dict(color="rgba(148, 163, 184, 0.4)", width=1), fill="tonexty", fillcolor="rgba(148, 163, 184, 0.05)"), row=1, col=1)

    # Volume Subplot
    vol_colors = ["#10b981" if c >= o else "#ef4444" for c, o in zip(df_feat["Close"], df_feat["Open"])]
    fig.add_trace(go.Bar(x=df_feat.index, y=df_feat["Volume"], name="Volume", marker_color=vol_colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["Volume_SMA_20"], mode="lines", name="Vol SMA 20", line=dict(color="#f97316", width=1)), row=2, col=1)

    # RSI Subplot
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["RSI_14"], mode="lines", name="RSI (14)", line=dict(color="#a855f7", width=1.8)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#10b981", line_width=1, row=3, col=1)

    # MACD Subplot
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["MACD"], mode="lines", name="MACD", line=dict(color="#38bdf8", width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["MACD_Signal"], mode="lines", name="Signal", line=dict(color="#f43f5e", width=1.5)), row=4, col=1)
    macd_colors = ["#10b981" if val >= 0 else "#ef4444" for val in df_feat["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df_feat.index, y=df_feat["MACD_Hist"], name="Histogram", marker_color=macd_colors), row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=920,
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: MODEL PERFORMANCE COMPARISON
# -----------------------------------------------------------------------------
with tab_perf:
    st.subheader("Model Evaluation & Head-to-Head Comparison")
    st.markdown(
        """
        Below is the side-by-side performance of **Stacked LSTM** versus **XGBoost Regressor**.
        To ensure complete methodological transparency, **Hold-Out Test Evaluation** is strictly
        distinguished from **Walk-Forward Out-Of-Sample Evaluation**.
        """
    )

    try:
        cfg = ModelConfig()
        dataset = preprocess_stock_data(df_raw, config=cfg)
        
        test_dates = dataset.test_dates
        test_curr = dataset.test_current_close.values
        test_y_unscaled = dataset.target_scaler.inverse_transform(dataset.y_test_seq.reshape(-1, 1)).flatten()

        comparison_records = []
        chart_data = {"Date": test_dates, "Actual_Close": test_y_unscaled}

        if xgb_model is not None:
            X_test_aligned = dataset.X_test.loc[test_dates]
            xgb_test_preds = xgb_model.predict(X_test_aligned)
            metrics_xgb = calculate_metrics(
                y_true=test_y_unscaled,
                y_pred=xgb_test_preds,
                current_prices=test_curr,
                model_name="XGBoost"
            )
            metrics_xgb["Sample_Size"] = f"{len(test_dates)} bars"
            metrics_xgb["Evaluation_Method"] = "Hold-Out Test Split (15%)"
            comparison_records.append(metrics_xgb)
            chart_data["XGBoost_Pred"] = xgb_test_preds

        if lstm_model is not None:
            lstm_test_preds = lstm_model.predict_sequences(dataset.X_test_seq)
            metrics_lstm = calculate_metrics(
                y_true=test_y_unscaled,
                y_pred=lstm_test_preds,
                current_prices=test_curr,
                model_name="LSTM"
            )
            metrics_lstm["Sample_Size"] = f"{len(test_dates)} bars"
            metrics_lstm["Evaluation_Method"] = "Hold-Out Test Split (15%)"
            comparison_records.append(metrics_lstm)
            chart_data["LSTM_Pred"] = lstm_test_preds

        st.markdown("#### A. Hold-Out Test Evaluation (Static 15% Out-of-Sample Partition)")
        st.caption(
            f"Evaluated on a strictly hold-out test partition of **{len(test_dates)} trading days** "
            f"({str(test_dates[0])[:10]} to {str(test_dates[-1])[:10]}). "
            "Neither model was trained or fitted on this partition."
        )

        if comparison_records:
            df_metrics = pd.DataFrame(comparison_records)
            cols_order = ["Model", "RMSE", "MAE", "MAPE (%)", "Directional_Accuracy (%)", "Sample_Size", "Evaluation_Method"]
            st.dataframe(
                df_metrics[cols_order].style.format({
                    "RMSE": "₹{:.2f}",
                    "MAE": "₹{:.2f}",
                    "MAPE (%)": "{:.2f}%",
                    "Directional_Accuracy (%)": "{:.2f}%"
                }),
                use_container_width=True
            )

            # Actual vs Predicted Plotly Chart
            st.markdown("#### Test Partition Price Trajectory: Actual vs Predicted")
            df_plot = pd.DataFrame(chart_data).set_index("Date")

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Actual_Close"], mode="lines+markers", name="Actual Close", line=dict(color="#f8fafc", width=2.5)))
            if "XGBoost_Pred" in df_plot.columns:
                fig_comp.add_trace(go.Scatter(x=df_plot.index, y=df_plot["XGBoost_Pred"], mode="lines+markers", name="XGBoost Prediction", line=dict(color="#38bdf8", width=2, dash="dash")))
            if "LSTM_Pred" in df_plot.columns:
                fig_comp.add_trace(go.Scatter(x=df_plot.index, y=df_plot["LSTM_Pred"], mode="lines+markers", name="LSTM Prediction", line=dict(color="#ec4899", width=2, dash="dot")))

            fig_comp.update_layout(
                template="plotly_dark",
                height=480,
                xaxis_title="Date",
                yaxis_title="Stock Price (INR ₹)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")
        st.markdown("#### B. Walk-Forward Out-Of-Sample Evaluation (Dynamic Expanding Windows)")
        clean_sym = active_symbol.replace(".", "_")
        backtest_file = BACKTEST_DATA_DIR / f"{clean_sym}_xgboost_walk_forward.csv"
        
        if backtest_file.exists():
            df_wf_loaded = pd.read_csv(backtest_file, parse_dates=["Date"], index_col="Date")
            wf_summary = calculate_metrics(
                y_true=df_wf_loaded["Actual_Next_Price"],
                y_pred=df_wf_loaded["Predicted_Next_Price"],
                current_prices=df_wf_loaded["Current_Price"],
                model_name="XGBoost Walk-Forward"
            )
            wf_summary["Sample_Size"] = f"{len(df_wf_loaded)} sequential steps"
            wf_summary["Evaluation_Method"] = "Expanding Window Walk-Forward"

            df_wf_table = pd.DataFrame([wf_summary])[["Model", "RMSE", "MAE", "MAPE (%)", "Directional_Accuracy (%)", "Sample_Size", "Evaluation_Method"]]
            st.dataframe(
                df_wf_table.style.format({
                    "RMSE": "₹{:.2f}",
                    "MAE": "₹{:.2f}",
                    "MAPE (%)": "{:.2f}%",
                    "Directional_Accuracy (%)": "{:.2f}%"
                }),
                use_container_width=True
            )
        else:
            st.info("Walk-forward simulation has not been run for this stock yet. Go to the **'🔄 Walk-Forward Backtesting'** tab to generate it.")

        if xgb_model is not None:
            st.markdown("#### Feature Importances (XGBoost)")
            df_imp = xgb_model.get_feature_importances().head(10)
            fig_imp = go.Figure(go.Bar(
                x=df_imp["Importance"][::-1],
                y=df_imp["Feature"][::-1],
                orientation="h",
                marker_color="#38bdf8"
            ))
            fig_imp.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(l=40, r=40, t=20, b=40),
                xaxis_title="Importance Score",
                yaxis_title="Feature"
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.error(f"Error evaluating models on test partition: {e}")


# -----------------------------------------------------------------------------
# TAB 3: WALK-FORWARD BACKTESTING
# -----------------------------------------------------------------------------
with tab_backtest:
    st.subheader("Causal Walk-Forward Backtesting Engine")
    st.markdown(
        r"""
        Walk-forward validation simulates real-world sequential trading. At every step $t$,
        the model is trained strictly on historical data $0 \dots t$ and tasked with forecasting $t+1$.
        The test window rolls forward, guaranteeing **zero data leakage**.
        """
    )

    col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
    with col_b1:
        initial_train_bars = st.slider("Initial Training Window (Bars)", min_value=80, max_value=250, value=140, step=10)
    with col_b2:
        retrain_freq = st.slider("Retraining Frequency (Bars)", min_value=10, max_value=60, value=25, step=5)
    with col_b3:
        btn_run_wf = st.button("▶ Run Walk-Forward Simulation", use_container_width=True)

    clean_sym = active_symbol.replace(".", "_")
    backtest_file = BACKTEST_DATA_DIR / f"{clean_sym}_xgboost_walk_forward.csv"

    if btn_run_wf or backtest_file.exists():
        if btn_run_wf:
            with st.spinner("Executing walk-forward backtest simulation..."):
                backtester = WalkForwardBacktester(
                    symbol=active_symbol,
                    initial_train_size=initial_train_bars,
                    step_size=1,
                    model_type="xgboost"
                )
                df_wf, wf_metrics = backtester.run(df_raw, retrain_every=retrain_freq)
                backtester.save_results()
        else:
            df_wf = pd.read_csv(backtest_file, parse_dates=["Date"], index_col="Date")
            wf_metrics = calculate_metrics(
                y_true=df_wf["Actual_Next_Price"],
                y_pred=df_wf["Predicted_Next_Price"],
                current_prices=df_wf["Current_Price"],
                model_name="XGBoost-WalkForward"
            )
            wf_metrics["Hypothetical_Strategy_Return (%)"] = round(float((df_wf["Cumulative_Strategy"].iloc[-1] - 1.0) * 100.0), 2)
            wf_metrics["Benchmark_Market_Return (%)"] = round(float((df_wf["Cumulative_Market"].iloc[-1] - 1.0) * 100.0), 2)
            wf_metrics["Start_Date"] = str(df_wf.index[0])[:10]
            wf_metrics["End_Date"] = str(df_wf.index[-1])[:10]

        # Parameters and Configuration Card
        st.markdown(
            f"""<div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;">
                <span style="color: #94a3b8; font-size: 0.85rem;">BACKTEST CONFIGURATION:</span> &nbsp;
                <b>Initial Window:</b> <code>{initial_train_bars} bars</code> &nbsp;|&nbsp;
                <b>Retraining Frequency:</b> <code>Every {retrain_freq} bars</code> &nbsp;|&nbsp;
                <b>Evaluation Period:</b> <code>{wf_metrics['Start_Date']} to {wf_metrics['End_Date']}</code>
            </div>""",
            unsafe_allow_html=True
        )

        # Summary Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Out-of-Sample Predictions", f"{len(df_wf)} days")
        m2.metric("RMSE", format_inr(wf_metrics['RMSE']))
        m3.metric("MAE", format_inr(wf_metrics['MAE']))
        m4.metric("Directional Accuracy", f"{wf_metrics['Directional_Accuracy (%)']:.2f}%")
        strat_ret = wf_metrics["Hypothetical_Strategy_Return (%)"]
        m5.metric("Hypothetical Strategy", f"{strat_ret:+.2f}%", delta=f"{strat_ret - wf_metrics['Benchmark_Market_Return (%)']:+.2f}% vs B&H")

        # Cumulative Returns Chart
        st.markdown("#### Cumulative Returns: Hypothetical Strategy vs Buy & Hold Benchmark")
        fig_strat = go.Figure()
        fig_strat.add_trace(go.Scatter(x=df_wf.index, y=df_wf["Cumulative_Strategy"], mode="lines", name="Hypothetical Model Strategy", line=dict(color="#10b981", width=2.5)))
        fig_strat.add_trace(go.Scatter(x=df_wf.index, y=df_wf["Cumulative_Market"], mode="lines", name="Buy & Hold Benchmark", line=dict(color="#94a3b8", width=1.5, dash="dash")))
        fig_strat.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_title="Date",
            yaxis_title="Cumulative Equity Growth (Base = 1.0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_strat, use_container_width=True)

        # Actual vs Predicted & Error Distribution
        col_w1, col_w2 = st.columns([1.6, 1])
        with col_w1:
            st.markdown("#### Walk-Forward Actual vs Predicted Daily Prices")
            fig_wf_p = go.Figure()
            fig_wf_p.add_trace(go.Scatter(x=df_wf.index, y=df_wf["Actual_Next_Price"], mode="lines", name="Actual Next Price", line=dict(color="#f8fafc", width=1.5)))
            fig_wf_p.add_trace(go.Scatter(x=df_wf.index, y=df_wf["Predicted_Next_Price"], mode="lines", name="Predicted Next Price", line=dict(color="#38bdf8", width=1.5, dash="dot")))
            fig_wf_p.update_layout(
                template="plotly_dark",
                height=350,
                xaxis_title="Date",
                yaxis_title="Price (INR ₹)",
                margin=dict(l=40, r=40, t=20, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_wf_p, use_container_width=True)

        with col_w2:
            st.markdown("#### Prediction Error Distribution")
            fig_err = go.Figure(go.Histogram(
                x=df_wf["Error"],
                nbinsx=30,
                marker_color="#c084fc",
                opacity=0.75
            ))
            fig_err.update_layout(
                template="plotly_dark",
                height=350,
                xaxis_title="Prediction Error (Predicted - Actual ₹)",
                yaxis_title="Frequency",
                margin=dict(l=40, r=40, t=20, b=40)
            )
            st.plotly_chart(fig_err, use_container_width=True)

        # PROMINENT BACKTEST DISCLAIMER
        st.warning(
            "⚠️ **Historical Backtest Disclaimer:** Historical backtest results are hypothetical simulations "
            "for academic research. They do not guarantee future performance and do not constitute investment advice. "
            "Simulations do not incorporate brokerage commissions, STT, or liquidity slippage."
        )
    else:
        st.info("Click **'▶ Run Walk-Forward Simulation'** above to compute chronological walk-forward out-of-sample validation.")


# -----------------------------------------------------------------------------
# TAB 4: METHODOLOGY & ABOUT
# -----------------------------------------------------------------------------
with tab_about:
    st.subheader("System Architecture & Quantitative Methodology")
    st.markdown(
        r"""
        ### 1. Project Overview
        This platform delivers an end-to-end Machine Learning and Quantitative Finance framework
        tailored exclusively for Indian National Stock Exchange (NSE) and Bombay Stock Exchange (BSE) equities.
        It features zero data leakage, chronological preprocessing, dual-model consensus forecasting (LSTM + XGBoost),
        and explainable multi-factor signal synthesis.

        ### 2. Live Data Architecture & Safe Pipeline
        The system separates tick data from historical training schemas:
        - **Historical Provider**: Yahoo Finance (`YFinanceProvider`) supplies daily OHLCV bars.
        - **Live Provider**: Upstox V3 Market Data Feed (`UpstoxProvider`) streams real-time LTPC ticks over WebSocket with Protobuf decoding.
        - **Live Price Pipeline**: Single ticks are not used to calculate daily indicators in isolation. Instead, the latest tick updates the current day's market state ($Close = LTP, High = \max(High, LTP), Low = \min(Low, LTP)$), and indicators are calculated causally across the entire historical series.

        ### 3. Strict Anti-Leakage Protocol
        Time-series prediction is vulnerable to lookahead bias. This system enforces:
        - **Monotonic Chronological Partitioning**: Training (70%), Validation (15%), and Out-of-Sample Testing (15%). No random shuffling.
        - **Train-Only Scaler Parameterization**: `StandardScaler` and `MinMaxScaler` parameters ($\mu, \sigma, \min, \max$) are computed exclusively on training records. Validation and Test splits are transformed using these pre-fitted parameters.
        - **Causal LSTM Sequence Formatting**: For lookback window $L=60$, sequences contain only bars $t-L \dots t-1$ to predict target $t+1$.
        - **Backward-Looking Indicators**: All moving averages, momentum oscillators (RSI, MACD), and volatility indicators use backward rolling windows.

        ### 4. Model Architectures
        - **Stacked LSTM**: 2-layer Recurrent Neural Network with Dropout (0.2) and Adam optimization minimizing Mean Squared Error (MSE), regularized by EarlyStopping on validation loss.
        - **XGBoost Regressor**: Gradient Boosted Decision Trees trained on tabular lag features, moving average spreads, and volatility bands with sub-sampling and col-sampling.

        ### 5. Educational Research Signal Engine
        Signals are not simple binary indicators. The engine executes a multi-factor confluence check:
        - **BUY Signal**: Predicted return $> +0.5\%$ **AND** MACD bullish **AND** Price $\ge$ EMA(20) **AND** RSI $< 70$ **AND** model directional agreement.
        - **SELL Signal**: Predicted return $< -0.5\%$ **AND** MACD bearish **AND** Price $\le$ EMA(20) **AND** RSI $> 30$ **AND** model directional agreement.
        - **HOLD Signal**: Neutral return band, model divergence, or conflicting momentum filters.

        ### 6. Academic Disclaimer
        **Educational and Research Purpose Only.** Historical simulations and machine learning forecasts
        do not guarantee future returns. Indian stock market equities are subject to volatility, macroeconomic shocks,
        and market risk. Never trade real capital based on research models.
        """
    )
