"""CLI Script: Train Deep Learning LSTM Model on Stock Sequences.

Usage:
    python scripts/train_lstm.py --symbol RELIANCE.NS --period 2y --epochs 20
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ModelConfig
from src.data.download_data import download_and_save_stock_data
from src.features.preprocessor import preprocess_stock_data
from src.models.lstm_model import StockLSTMModel
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import get_logger

logger = get_logger("script_train_lstm")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Deep Learning Model on Stock Data")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="Ticker symbol (e.g. RELIANCE.NS)")
    parser.add_argument("--period", type=str, default="2y", help="Historical data period (e.g. 2y, 5y)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lookback", type=int, default=60, help="Sequence window length")
    args = parser.parse_args()

    print("=" * 60)
    print(f" Training Stacked LSTM Network for {args.symbol}")
    print("=" * 60)

    # 1. Fetch & Validate Data
    print(f"[1/4] Fetching historical data ({args.period})...")
    df, report = download_and_save_stock_data(symbol=args.symbol, period=args.period)
    print(f"      Loaded {len(df)} validated records ({report['start_date'][:10]} to {report['end_date'][:10]})")

    # 2. Chronological Preprocessing & Sequence Creation
    print(f"[2/4] Constructing causal 3D sequences with lookback={args.lookback}...")
    cfg = ModelConfig(
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    dataset = preprocess_stock_data(df, config=cfg)
    print(f"      Training Sequences:   {dataset.X_train_seq.shape}")
    print(f"      Validation Sequences: {dataset.X_val_seq.shape}")
    print(f"      Test Sequences:       {dataset.X_test_seq.shape}")

    # 3. Train Model
    print(f"[3/4] Fitting LSTM with EarlyStopping on validation loss...")
    lstm_model = StockLSTMModel(config=cfg)
    lstm_model.train(
        dataset=dataset,
        symbol=args.symbol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )

    # 4. Evaluate on Test Sequences
    print("[4/4] Evaluating on hold-out Test sequences...")
    y_test_pred = lstm_model.predict_sequences(dataset.X_test_seq)
    
    # Inverse scale y_test_seq to get actual rupee prices
    y_test_actual = dataset.target_scaler.inverse_transform(
        dataset.y_test_seq.reshape(-1, 1)
    ).flatten()

    test_current_close = dataset.test_current_close.values

    metrics = calculate_metrics(
        y_true=y_test_actual,
        y_pred=y_test_pred,
        current_prices=test_current_close,
        model_name="LSTM"
    )

    print("\n" + "-" * 40)
    print(" TEST PARTITION PERFORMANCE (Out-of-Sample)")
    print("-" * 40)
    print(f" RMSE:                 ₹{metrics['RMSE']:.2f}")
    print(f" MAE:                  ₹{metrics['MAE']:.2f}")
    print(f" MAPE:                 {metrics['MAPE (%)']:.2f}%")
    print(f" Directional Accuracy: {metrics['Directional_Accuracy (%)']:.2f}%")
    print(f" Test Days Evaluated:  {metrics['Total_Predictions']}")
    print("-" * 40)

    # Save Model
    saved_path = lstm_model.save()
    print(f"\n✓ LSTM model, scalers, and metadata saved to:\n  {saved_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
