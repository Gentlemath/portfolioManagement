from pathlib import Path
import sys

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsm import RegimeDetector  as rd

def sp500_example(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load S&P 500 price series for demonstration."""

    prices = yf.download("^GSPC", start=start, end=end, progress=False)[["Close"]]
    prices = prices.rename(columns={"Close": "SP500"})
    return prices

def main() -> None:
    start = pd.Timestamp("2020-05-12")
    end = pd.Timestamp("2026-05-12")
    prices = sp500_example(start, end)
    returns = (prices.pct_change().dropna() * 100)  # Convert to percentage

    print("=== Price summary ===")
    print(prices.tail())
    print("\n=== Return summary ===")
    print(returns.describe())

    # Visual exploration
    print("\n=== Plotting return and rolling volatility overview ===")
    detector = rd()
    detector.plot_regime_changes(returns["SP500"], window=30)


if __name__ == "__main__":
    main()