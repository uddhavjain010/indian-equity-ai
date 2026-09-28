"""CLI Script: Train XGBoost Model on Historical Stock Data.

Usage:
    python scripts/train_xgboost.py --symbol RELIANCE.NS --period 2y
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ModelConfig
from src.data.download_data import download_and_save_stock_data
from src.features.preprocessor import preprocess_stock_data
from src.models.xgboost_model import StockXGBoostModel
from src.evaluation.metrics import calculate_metrics
from src.utils.logger import get_logger

logger = get_logger("script_train_xgboost")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost Model on Indian Stock Data")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="Ticker symbol (e.g. RELIANCE.NS)")
    parser.add_argument("--period", type=str, default="2y", help="Historical data period (e.g. 2y, 5y)")
    parser.add_argument("--n_estimators", type=int, default=150, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum tree depth")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate")
    args = parser.parse_args()

    print("=" * 60)
    print(f" Training XGBoost for {args.symbol}")
    print("=" * 60)

    # 1. Fetch & Validate Data
    print(f"[1/4] Fetching historical data ({args.period})...")
    df, report = download_and_save_stock_data(symbol=args.symbol, period=args.period)
    print(f"      Loaded {len(df)} validated records ({report['start_date'][:10]} to {report['end_date'][:10]})")

    # 2. Chronological Preprocessing & Scaling
    print("[2/4] Engineering technical features & chronological partitioning...")
    cfg = ModelConfig(
        xgb_n_estimators=args.n_estimators,
        xgb_max_depth=args.max_depth,
        xgb_learning_rate=args.learning_rate
    )
    dataset = preprocess_stock_data(df, config=cfg)
    print(f"      Features: {len(dataset.feature_names)} features computed.")
    print(f"      Chronological splits: Train={len(dataset.X_train)}, Val={len(dataset.X_val)}, Test={len(dataset.X_test)}")

    # 3. Train Model
    print("[3/4] Fitting XGBoost Regressor with zero-leakage training...")
    xgb_model = StockXGBoostModel(config=cfg)
    xgb_model.train(dataset, symbol=args.symbol, verbose=False)

    # 4. Evaluate on Test Partition
    print("[4/4] Evaluating on hold-out Test partition...")
    y_test_pred = xgb_model.predict(dataset.X_test)
    y_test_actual = dataset.y_test.values
    test_current_close = dataset.X_test["Close"].values

    metrics = calculate_metrics(
        y_true=y_test_actual,
        y_pred=y_test_pred,
        current_prices=test_current_close,
        model_name="XGBoost"
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
    saved_path = xgb_model.save()
    print(f"\n✓ Model and metadata successfully saved to:\n  {saved_path}")

    # Top Features
    df_imp = xgb_model.get_feature_importances().head(5)
    print("\nTop 5 Most Informative Features:")
    for _, row in df_imp.iterrows():
        print(f"  - {row['Feature']:<20}: {row['Importance']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
