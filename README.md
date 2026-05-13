# Portfolio Management

A portfolio management system for learning and development.

## Project Structure

- `data/` - Data storage and datasets
- `src/` - Source code modules
- `tests/` - Test cases

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Version Control

This project uses Git for version management.

```bash
git log          # View commit history
git status       # Check current status
git add .        # Stage changes
git commit -m "message"  # Commit changes
```

## Development

See individual module documentation in `src/` for more details.

## Data Sources

This starter supports multiple data providers:

- `yfinance` — price + simple fundamentals
- `Alpha Vantage` — fundamentals + price + global equities
- `FRED` — macroeconomic data
- `AKShare` — China market data
- `BaoStock` — China A-share backup data
- `Tushare` — China fundamentals

### Environment Variables

Set API keys for providers that require them:

```bash
export ALPHAVANTAGE_API_KEY="your_alpha_vantage_key"
export FRED_API_KEY="your_fred_key"
export TUSHARE_TOKEN="your_tushare_token"
```

### Example Usage

```python
from data.data_loader import create_data_loader

# load CSV data
csv_loader = create_data_loader("csv", data_dir="data")
df = csv_loader.load_csv("portfolio.csv")

# load yfinance price data
yfinance_loader = create_data_loader("yfinance")
price_df = yfinance_loader.history("AAPL", start="2024-01-01", end="2025-01-01")

# Alpha Vantage fundamentals
av_loader = create_data_loader("alpha_vantage")
overview = av_loader.get_fundamentals("MSFT")

# FRED macro data
fred_loader = create_data_loader("fred")
macros = fred_loader.get_series("GDP")
```

## Exploratory Data Analysis (EDA)

A small EDA demo is available at `examples/eda_demo.py`.

It shows how to:

- load S&P 500 price data using yfinance
- compute simple returns
- create a comprehensive overview plot with prices, returns, and cumulative returns in subplots
- inspect return distribution statistics
- plot combined histograms and Q-Q plots for distribution analysis

Run the demo with:

```bash
python examples/eda_demo.py
```
