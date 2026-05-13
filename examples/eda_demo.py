"""Example exploratory data analysis script for portfolio management."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yfinance as yf

# Ensure src is on the import path when running from the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eda import DistributionAnalyzer, PlotAnalyzer


def sp500_example(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load S&P 500 price series for demonstration."""

    prices = yf.download("^GSPC", start=start, end=end, progress=False)[["Close"]]
    prices = prices.rename(columns={"Close": "SP500"})
    return prices


def main() -> None:
    start = pd.Timestamp("2025-05-12")
    end = pd.Timestamp("2026-05-12")
    prices = sp500_example(start, end)
    returns = PlotAnalyzer.compute_returns(prices, method="simple")

    print("=== Price summary ===")
    print(prices.tail())
    print("\n=== Return summary ===")
    print(returns.describe())

    # Visual exploration
    PlotAnalyzer.plot_overview(prices, returns, title="S&P 500 Portfolio Overview")

    # Distribution exploration
    print("\n=== Return distribution statistics ===")
    print(DistributionAnalyzer.describe_returns(returns))

    DistributionAnalyzer.plot_distribution_overview(returns, bins=40)


if __name__ == "__main__":
    main()
