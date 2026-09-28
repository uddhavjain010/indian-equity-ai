"""CLI Script: Download Historical NSE/BSE Equity Data.

Usage:
    python scripts/download_data.py --symbol RELIANCE.NS --period 2y --interval 1d
    python scripts/download_data.py --all  # Download top 5 Nifty 50 benchmark stocks
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.stocks import NIFTY_50_UNIVERSE
from src.data.download_data import download_and_save_stock_data
from src.utils.logger import get_logger

logger = get_logger("script_download")


def main():
    parser = argparse.ArgumentParser(description="Download Historical Stock Market Data")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="Ticker symbol (e.g. RELIANCE.NS)")
    parser.add_argument("--period", type=str, default="2y", choices=["1y", "2y", "5y", "10y"], help="Historical timeframe")
    parser.add_argument("--interval", type=str, default="1d", choices=["1d"], help="Data resolution")
    parser.add_argument("--all", action="store_true", help="Download top liquid Nifty 50 benchmark stocks")
    parser.add_argument("--force", action="store_true", help="Force fresh download ignoring local cache")

    args = parser.parse_args()

    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"] if args.all else [args.symbol]

    print("=" * 60)
    print(" Indian Stock Price Prediction: Historical Data Downloader")
    print("=" * 60)

    for sym in symbols:
        print(f"\n[+] Fetching {sym} (Period: {args.period}, Interval: {args.interval})...")
        try:
            df, report = download_and_save_stock_data(
                symbol=sym,
                period=args.period,
                interval=args.interval,
                force_refresh=args.force
            )
            print(f"    ✓ Successfully downloaded and validated {len(df)} records.")
            print(f"    ✓ Time range: {report['start_date'][:10]} to {report['end_date'][:10]}")
            print(f"    ✓ Validation: {report['impossible_price_rows_removed']} invalid prices removed, {report['duplicate_rows_removed']} duplicates removed.")
        except Exception as e:
            print(f"    ✗ Error fetching {sym}: {e}")

    print("\n" + "=" * 60)
    print(" All downloads complete! Datasets saved to data/raw/")
    print("=" * 60)


if __name__ == "__main__":
    main()
