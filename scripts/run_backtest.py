"""CLI Script: Run Walk-Forward Backtesting.

Usage:
    python scripts/run_backtest.py --symbol RELIANCE.NS --period 2y
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.download_data import download_and_save_stock_data
from src.backtesting.walk_forward import WalkForwardBacktester
from src.utils.logger import get_logger

logger = get_logger("script_run_backtest")


def main():
    parser = argparse.ArgumentParser(description="Run Walk-Forward Backtest")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="Ticker symbol (e.g. RELIANCE.NS)")
    parser.add_argument("--period", type=str, default="2y", help="Historical data period")
    parser.add_argument("--initial_train", type=int, default=120, help="Initial training bars")
    parser.add_argument("--step_size", type=int, default=1, help="Walk-forward step size")
    parser.add_argument("--retrain_every", type=int, default=20, help="Retrain frequency in trading bars")
    args = parser.parse_args()

    print("=" * 60)
    print(f" Walk-Forward Backtesting: {args.symbol}")
    print("=" * 60)

    # 1. Fetch Data
    print(f"[1/3] Fetching historical data ({args.period})...")
    df, report = download_and_save_stock_data(symbol=args.symbol, period=args.period)
    print(f"      Loaded {len(df)} validated records ({report['start_date'][:10]} to {report['end_date'][:10]})")

    # 2. Run Backtest
    print(f"[2/3] Simulating causal walk-forward forecasting (step={args.step_size})...")
    backtester = WalkForwardBacktester(
        symbol=args.symbol,
        initial_train_size=args.initial_train,
        step_size=args.step_size,
        model_type="xgboost"
    )

    results_df, metrics = backtester.run(df, retrain_every=args.retrain_every)

    # 3. Report Results
    print("[3/3] Backtest Simulation Complete!")
    print("\n" + "=" * 50)
    print(" WALK-FORWARD PERFORMANCE REPORT")
    print("=" * 50)
    print(f" Symbol:                       {args.symbol}")
    print(f" Evaluation Window:            {metrics['Start_Date']} to {metrics['End_Date']}")
    print(f" Out-of-Sample Test Steps:     {metrics['Total_Predictions']}")
    print(f" RMSE:                         ₹{metrics['RMSE']:.2f}")
    print(f" MAE:                          ₹{metrics['MAE']:.2f}")
    print(f" MAPE:                         {metrics['MAPE (%)']:.2f}%")
    print(f" Directional Accuracy:         {metrics['Directional_Accuracy (%)']:.2f}%")
    print(f" Hypothetical Strategy Return: {metrics['Hypothetical_Strategy_Return (%)']:.2f}%")
    print(f" Benchmark Buy & Hold Return:  {metrics['Benchmark_Market_Return (%)']:.2f}%")
    print("=" * 50)
    print(" DISCLAIMER: Hypothetical backtest for academic research only.")
    print(" Past simulation results do not guarantee future performance.")
    print("=" * 50)

    saved_file = backtester.save_results()
    print(f"\n✓ Backtest details saved to: {saved_file}")


if __name__ == "__main__":
    main()
