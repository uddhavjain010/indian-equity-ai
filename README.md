# Real-Time Indian Stock Price Prediction and Signal Dashboard using LSTM and XGBoost on NSE/BSE Data

An end-to-end Quantitative Finance, Machine Learning, and Streamlit platform built for academic research, algorithmic modeling, and market analysis on Indian equities listed on the **National Stock Exchange (NSE)** and **Bombay Stock Exchange (BSE)**.

---

## 📌 Executive Summary & Key Highlights

- **Pure Indian Equity Focus**: Nifty 50 constituents configured out of the box with seamless dual-exchange ticker resolution (`.NS` for NSE and `.BO` for BSE).
- **Dual Data Provider Architecture**:
  - **Default / Zero-Cost**: **Yahoo Finance** (`YFinanceProvider`) supplies historical daily OHLCV bars and latest available near-real-time market quotes.
  - **Live Streaming / Upstox V3**: **Upstox V3 Market Data Feed** (`UpstoxProvider`) interface supports authorized WebSocket streaming with binary Google Protocol Buffers (Protobuf) decoding via `MarketDataFeedV3_pb2`.
  - **Graceful Fallback**: Automatically falls back to yfinance if Upstox credentials are unconfigured or connections drop.
- **Strict Anti-Leakage Protocol**: Proven zero data leakage—features are backward-looking, datasets are partitioned strictly chronologically (Train 70% / Val 15% / Test 15%), and all scalers are fit **solely** on training records.
- **Dual Model Engine**:
  - **Stacked LSTM Neural Network**: Captures sequential temporal patterns over a 60-bar lookback window with Dropout regularization and EarlyStopping.
  - **XGBoost Regressor**: High-speed gradient boosted decision trees over 33 engineered technical and lag features with feature importance rankings.
- **Causal Walk-Forward Backtesting**: Simulates realistic forward rolling predictions with period re-fitting and cumulative return benchmarking against Buy & Hold.
- **Multi-Factor Consensus Signal Engine**: Transparent, explainable research signals (**BUY**, **HOLD**, **SELL**) combining forecast returns, RSI, MACD, and EMA-20 trend filters.
- **Modern Glassmorphic Streamlit Dashboard**: Dark glassmorphic interface featuring multi-pane Plotly candlestick charts, indicator overlays, model comparison tables, and walk-forward equity curves.

---

## 📐 System Architecture

```
                                  [ Indian Equities ]
                                    (NSE / BSE)
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         [ YFinance Provider ]                   [ Upstox V3 Provider ]
        (Default / Zero-Cost)                 (Live WebSocket / Protobuf)
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                            [ Data Validator & Sanitizer ]
                            • Impossible prices (H < L, <= 0)
                            • Monotonic chronology & deduplication
                            • Missing volume & holiday handling
                                         │
                                         ▼
                            [ Feature Engineering Engine ]
                            • 33 Backward-looking indicators
                            • Causal lag features (1, 2, 3, 5, 10)
                                         │
                                         ▼
                         [ Anti-Leakage Preprocessing ]
                            • Chronological Split (70/15/15)
                            • Scalers fitted ONLY on Train set
                            • Causal 3D LSTM Sequence windows
                            • Live Price Pipeline integration
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
          [ Stacked LSTM Model ]                    [ XGBoost Model ]
          • Lookback: 60 bars                       • 150 Estimators, Depth 5
          • 64 & 32 Units + Dropout                 • Gradient Boosted Trees
          • EarlyStopping on Val Loss               • Tabular Feature Matrix
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                          [ Model Evaluation & Metrics ]
                          • RMSE, MAE, MAPE
                          • Directional Accuracy (t vs t+1)
                          • Hold-Out vs Walk-Forward separation
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
        [ Walk-Forward Backtester ]               [ Signal Engine ]
        • Expanding historical windows            • Multi-indicator consensus
        • Step-by-step out-of-sample              • Return threshold + MACD + RSI + EMA
        • Cumulative return simulation            • Transparent reasoning output
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                          [ Streamlit Financial Dashboard ]
                          • Live Quotes & Interactive Charts
                          • Consensus Signals & Explainability
                          • Walk-Forward Performance & Equity Curve
```

---

## 🔌 Data Providers & Real-Time Streaming Architecture

### 1. Default Provider: Yahoo Finance (`yfinance`)
- **Role**: Supplies historical daily OHLCV series and latest available quotes.
- **Cost**: Zero cost; requires no API registration or paid data infrastructure.
- **Labeling Standard**: Labeled explicitly as **`"LATEST AVAILABLE — yfinance"`** (near-real-time / educational). It is never falsely labeled as exchange-grade low-latency data.

### 2. Live Provider: Upstox V3 Market Data Feed (`UpstoxProvider`)
- **Role**: Provides real-time WebSocket market data streaming for Indian equities.
- **Protocol**: Official **Upstox V3 Market Data Feed** (`MarketDataFeedV3`).
  1. Requests authorized one-time WebSocket redirect URL via `GET /v3/feed/market-data-feed/authorize`.
  2. Establishes binary WebSocket connection to `wss://api.upstox.com/v3/feed/market-data-feed`.
  3. Issues binary/JSON subscription message with target instrument keys (e.g. `NSE_EQ|INE002A01018` for Reliance).
  4. Decodes incoming binary ticks using compiled Google Protocol Buffers (`MarketDataFeedV3_pb2.py`).
- **Live Price Pipeline Architecture**:
  To prevent invalid calculation of daily indicators from a solitary tick, the system utilizes `integrate_live_tick_into_history`:
  $$\text{Live Price (LTP)} \longrightarrow \text{Update Current Day Bar } (Close = LTP, High = \max, Low = \min) \longrightarrow \text{Recalculate Indicators Causally}$$
  This safely feeds the model's 33-feature daily schema without corrupting historical intervals.

---

## 🏢 Indian Stock Universe & Ticker Conventions

The application targets Indian equities listed on the **National Stock Exchange of India (NSE)** and the **Bombay Stock Exchange (BSE)**.

- **NSE Ticker Format**: Symbol followed by `.NS` (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `SBIN.NS`).
- **BSE Ticker Format**: Scrip code or symbol followed by `.BO` (e.g., `500325.BO`, `532540.BO`).

Configured Nifty 50 constituents reside in [`config/stocks.py`](file:///Users/uddhavjain/AG%20pr3/config/stocks.py), including corporate names, sectors, and cross-exchange mappings:

```python
NIFTY_50_UNIVERSE = {
    "RELIANCE.NS": {
        "symbol": "RELIANCE.NS",
        "bse_symbol": "500325.BO",
        "company_name": "Reliance Industries Ltd",
        "exchange": "NSE",
        "sector": "Energy & Petrochemicals",
    },
    ...
}
```

---

## 📊 Feature Engineering (33 Indicators)

All features are calculated using strictly past and contemporaneous information ($t, t-1, \dots$). No future values are ever referenced:

1. **Price & Returns**:
   - `Daily_Return`: Percentage price change $\frac{Close_t - Close_{t-1}}{Close_{t-1}}$
   - `Log_Return`: $\ln(Close_t / Close_{t-1})$
2. **Trend & Moving Averages**:
   - Simple Moving Averages: `SMA_10`, `SMA_20`, `SMA_50`, `SMA_200`
   - Exponential Moving Averages: `EMA_10`, `EMA_20`, `EMA_50`
3. **Momentum**:
   - Relative Strength Index: `RSI_14` (14-period Wilder smoothing)
   - Moving Average Convergence Divergence: `MACD` (12, 26, 9), `MACD_Signal`, and `MACD_Hist`
4. **Volatility**:
   - Bollinger Bands: `BB_High`, `BB_Mid`, `BB_Low`, `BB_Width` (20-day, 2 standard deviations)
   - `Rolling_Vol_20`: 20-day rolling annualized return volatility
5. **Volume Dynamics**:
   - `Volume_Change`: Percentage volume change from previous trading session
   - `Volume_SMA_20`: 20-day moving average of traded volume
   - `Volume_Ratio`: $\frac{Volume_t}{Volume\_SMA_{20}}$
6. **Autoregressive Lag Features**:
   - Price lags: `Close_Lag_1`, `Close_Lag_2`, `Close_Lag_3`, `Close_Lag_5`, `Close_Lag_10`
   - Return lags: `Return_Lag_1`, `Return_Lag_5`

---

## 🎯 Target Definition

- **Continuous Price Target**:
  $$\text{Target\_Next\_Close}_t = Close_{t+1}$$
- **Directional Classification Target**:
  $$\text{Target\_Direction}_t = \begin{cases} 1 & \text{if } Close_{t+1} > Close_t \\ 0 & \text{otherwise} \end{cases}$$

---

## 🛡️ Data Leakage Prevention Guarantees

Time-series prediction is vulnerable to lookahead bias. This repository implements multiple mathematical and programmatic safeguards:

1. **Strict Chronological Splits**:
   - **Training Set (70%)**: $t = 0 \dots T_{\text{train}}$
   - **Validation Set (15%)**: $t = T_{\text{train}} \dots T_{\text{val}}$
   - **Test Set (15%)**: $t = T_{\text{val}} \dots T_{\text{end}}$
   *(Random shuffling is strictly prohibited).*
2. **Train-Only Scaling**:
   - `StandardScaler` ($\mu, \sigma$) and `MinMaxScaler` ($\min, \max$) are fit **exclusively** on `X_train` and `y_train`.
   - Validation and test splits are transformed using these frozen parameters.
3. **Causal LSTM Sequences**:
   - For any index $i$, sequence $X_{\text{seq}}[i] = X[i - L : i]$, predicting $y[i]$. The sequence strictly contains bars up to $i-1$.
4. **Mathematical Verification in Test Suite**:
   - [`tests/test_no_leakage.py`](file:///Users/uddhavjain/AG%20pr3/tests/test_no_leakage.py) proves:
     - Scaler parameters match training distribution and diverge from the full dataset.
     - Tampering with future data ($t+1 \dots$) produces zero change in feature values at time $t$.
     - Target shift is strictly $Close_{t+1}$.
     - Walk-forward backtesting isolates the expanding training window.

---

## 🤖 Models Architecture

### 1. Stacked LSTM (Deep Learning)
- **Framework**: TensorFlow 2.x / Keras
- **Architecture**:
  - `InputLayer(shape=(lookback=60, num_features=33))`
  - `LSTM(64 units, return_sequences=True)`
  - `Dropout(0.2)`
  - `LSTM(32 units, return_sequences=False)`
  - `Dropout(0.2)`
  - `Dense(16, activation='relu')`
  - `Dense(1, activation='linear')`
- **Optimization**: Adam (`lr=0.001`), Loss: Mean Squared Error (`mse`).
- **Regularization**: `EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)`.

### 2. XGBoost Regressor (Gradient Boosted Trees)
- **Framework**: XGBoost (`xgboost.XGBRegressor`)
- **Parameters**: `n_estimators=150`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Feature Importance**: Gain-based feature attribution shows top drivers (e.g. `Close`, `High`, `Close_Lag_1`, `EMA_20`).

---

## 📈 Evaluation Metrics

The dashboard and CLI report metrics across:
- **A. Hold-Out Test Evaluation**: Fixed 15% out-of-sample chronological test partition ($N=75$ bars).
- **B. Walk-Forward Out-Of-Sample Evaluation**: Sequential dynamic re-estimation ($N=380$ bars).

1. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
2. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$
3. **Mean Absolute Percentage Error (MAPE)**:
   $$\text{MAPE} = \frac{100\%}{N}\sum_{i=1}^N \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
4. **Directional Accuracy**:
   $$\text{Directional Accuracy} = \frac{1}{N}\sum_{i=1}^N \mathbf{1}_{(\hat{y}_i > Close_i) == (y_i > Close_i)} \times 100\%$$

---

## 🔄 Causal Walk-Forward Backtesting

Standard cross-validation is inappropriate for non-stationary market series. The [`WalkForwardBacktester`](file:///Users/uddhavjain/AG%20pr3/src/backtesting/walk_forward.py):
1. Initializes an expanding training window (e.g., 140 trading days).
2. Fits the feature scaler and model on data up to $t$.
3. Forecasts out-of-sample price for $t+1$.
4. Rolls the window forward by 1 bar and repeats until the dataset is exhausted.
5. Computes hypothetical educational strategy returns vs the Buy & Hold benchmark.

---

## 🚦 Rule-Based Quantitative Signal Engine

The signal engine computes transparent consensus decisions (**BUY**, **HOLD**, **SELL**) based on clear rules:

### BUY Signal Conditions (All must be satisfied):
- Predicted return $\frac{\hat{P}_{t+1} - P_t}{P_t} \ge +0.5\%$
- MACD is bullish (`MACD > MACD_Signal` or `MACD_Hist > 0`)
- Current price is trading above `EMA_20`
- `RSI_14 < 70` (not excessively overbought)
- Both models (if available) agree on upward direction

### SELL Signal Conditions (All must be satisfied):
- Predicted return $\frac{\hat{P}_{t+1} - P_t}{P_t} \le -0.5\%$
- MACD is bearish (`MACD < MACD_Signal` or `MACD_Hist < 0`)
- Current price is trading below `EMA_20`
- `RSI_14 > 30` (not excessively oversold)
- Both models (if available) agree on downward direction

### HOLD Signal:
- Triggered whenever returns reside within the neutral band ($\pm 0.5\%$), momentum conditions are conflicting, or LSTM and XGBoost directional predictions diverge.

---

## 🚀 Installation & Local Setup

### 1. Prerequisites
- Python 3.10 to 3.14.
- macOS, Linux, or Windows.

### 2. Setup Environment
```bash
git clone <repo-url> indian-stock-prediction
cd indian-stock-prediction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
python3 -m pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Contents of `.env`:
```ini
# Provider options: default, yfinance, upstox
YFINANCE_PROVIDER=default

# Optional Upstox V3 Market Data Feed Configuration
UPSTOX_CLIENT_ID=
UPSTOX_CLIENT_SECRET=
UPSTOX_ACCESS_TOKEN=
UPSTOX_REDIRECT_URI=https://127.0.0.1:5000/

LOG_LEVEL=INFO
DEFAULT_LOOKBACK=60
```

> **Note on Upstox Live Mode:** If Upstox credentials (`UPSTOX_ACCESS_TOKEN`) are not provided, the application automatically defaults to Yahoo Finance without errors. To use live streaming, obtain an access token from the [Upstox Developer Portal](https://upstox.com/developer/) and paste it into `.env`.

---

## 💻 CLI Commands & Training Scripts

### Download Historical Data
```bash
python scripts/download_data.py --symbol RELIANCE.NS --period 2y
```

### Train XGBoost Model
```bash
python scripts/train_xgboost.py --symbol RELIANCE.NS --period 2y
```

### Train LSTM Model
```bash
python scripts/train_lstm.py --symbol RELIANCE.NS --period 2y --epochs 15
```

### Run Walk-Forward Backtesting
```bash
python scripts/run_backtest.py --symbol RELIANCE.NS --period 2y --retrain_every 30
```

### Launch Streamlit Dashboard
```bash
streamlit run app.py
```
*The dashboard will open automatically in your browser at `http://localhost:8501`.*

---

## 🧪 Running Unit Tests

Run the complete 15-test automated test suite:
```bash
python3 -m pytest -v
```

---

## ⚠️ Limitations & Boundary Conditions

1. **Daily Bar Resolution**: Models evaluate daily EOD bars; intraday tick volatility and high-frequency order book microstructure are not captured in daily forecasts.
2. **Execution Realism**: Hypothetical backtest assumes fills at the official market close without incorporating slippage, transaction taxes (STT), brokerage, or exchange turnover fees.
3. **Upstox Live Feed Prerequisites**: Live WebSocket streaming mode requires an active Upstox developer account and a valid OAuth 2.0 access token generated on that trading day.

---

## ⚖️ Academic & Financial Disclaimer

> **DISCLAIMER:** This software and its generated signals (**BUY**, **HOLD**, **SELL**) are built strictly for **educational, academic, and research purposes**. No information in this repository constitutes investment, financial, tax, or legal advice. Market prices and equity values are subject to risk of capital loss. Never execute real monetary transactions based on model predictions.
