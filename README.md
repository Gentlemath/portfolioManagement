# Portfolio Management

A comprehensive portfolio management system for financial data exploration, analysis, and modeling. Built for learning and practical development in quantitative finance.

## Project Structure

```
src/
  ├── dataloader/          # Data loading from multiple sources
  │   ├── data_loader.py   # Loaders: yfinance, Alpha Vantage, FRED, AKShare, BaoStock, Tushare
  │   └── __init__.py
  ├── eda/                 # Exploratory Data Analysis
  │   ├── __init__.py
  │   ├── plots.py         # Price, return, and cumulative return visualization
  │   ├── distribution.py  # Return distribution analysis
  │   └── tsa.py           # Time series analysis: stationarity, ACF/PACF, autocorrelation testing
  └── tsm/                 # Time Series Modeling
      ├── __init__.py
      └── prediction.py    # GARCH and ARIMA-GARCH models for volatility and return prediction
examples/
  ├── eda_demo.py          # Complete EDA demonstration
  └── garch_demo.py        # GARCH modeling demonstration
tests/                      # Unit tests
data/                       # Data storage directory
.github/workflows/
  └── ci.yml               # GitHub Actions CI configuration
```

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Version Control

This project uses Git for version management:

```bash
git log          # View commit history
git status       # Check current status
git add .        # Stage changes
git commit -m "message"  # Commit changes
```

## Data Loading

### Supported Data Sources

- **yfinance** — Historical prices + basic fundamentals
- **Alpha Vantage** — Global equity prices, fundamentals, symbol search
- **FRED** — US macroeconomic indicators
- **AKShare** — China market data
- **BaoStock** — China A-share backup data
- **Tushare** — China equity fundamentals

### Configuration

Set API keys for providers that require them:

```bash
export ALPHAVANTAGE_API_KEY="your_api_key"
export FRED_API_KEY="your_api_key"
export TUSHARE_TOKEN="your_token"
```

### Data Loader Examples

```python
from src.dataloader import create_data_loader

# CSV data
csv_loader = create_data_loader("csv", data_dir="data")
df = csv_loader.load_csv("portfolio.csv")

# yfinance: price data
yf_loader = create_data_loader("yfinance")
prices = yf_loader.history("AAPL", start="2024-01-01", end="2025-01-01")
info = yf_loader.get_simple_fundamentals("AAPL")

# Alpha Vantage: global equities + fundamentals
av_loader = create_data_loader("alpha_vantage")
price_data = av_loader.get_price("MSFT", interval="Daily")
fundamentals = av_loader.get_fundamentals("MSFT")

# FRED: macroeconomic data
fred_loader = create_data_loader("fred")
gdp = fred_loader.get_series("GDP", start_date="2020-01-01")
```

## Exploratory Data Analysis (EDA)

### Overview

The EDA module provides tools for initial data exploration and understanding of financial time series.

### EDA Components

**PlotAnalyzer**
- Price series visualization
- Return series visualization
- Cumulative return plotting
- Multi-asset overview plots with subplots

**DistributionAnalyzer**
- Return distribution statistics (mean, std, skew, kurtosis)
- Value at Risk (VaR) and Conditional VaR
- Return distribution histograms
- Q-Q plots for normality assessment

**TimeSeriesAnalyzer**
- Stationarity tests: ADF and KPSS
- Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots
- Significance testing for autocorrelation at individual lags
- Warning handling for edge cases

### Running the EDA Demo

```bash
cd examples
python eda_demo.py
```

The demo performs:

1. **Data Loading**: Real S&P 500 price data (via yfinance)
2. **Price Overview**: Plots prices, returns, and cumulative returns
3. **Return Statistics**: Computes mean, volatility, skew, kurtosis
4. **Distribution Analysis**: Histograms and Q-Q plots
5. **Stationarity Testing**: ADF-KPSS tests with interpretation
6. **Autocorrelation Analysis**: ACF/PACF with significance testing at α=0.01


## Time Series Modeling (TSM)

### Overview

The TSM module provides advanced time series models for volatility and return prediction.

### TSM Components

**GARCHPredictor**
- Fit GARCH(p,q) models for volatility modeling
- Predict next-day volatility and returns
- Model evaluation with standardized residuals and Ljung-Box tests
- Conditional volatility plotting

**ARIMAGARCHPredictor**
- Combined ARIMA-GARCH models for return prediction
- Handles both mean and volatility dynamics

### Running the GARCH Demo

```bash
cd examples
python garch_demo.py
```

The demo performs:

1. **Data Loading**: S&P 500 price data (via yfinance)
2. **Return Calculation**: Daily percentage returns
3. **GARCH Fitting**: GARCH(1,1) model estimation
4. **Parameter Analysis**: Model coefficients and diagnostics
5. **Volatility Prediction**: Next-day volatility forecast
6. **Model Evaluation**: Residual analysis and goodness-of-fit tests
7. **Visualization**: Conditional volatility plot


## Key Insights from EDA

### Stationarity
- Financial log-returns are typically stationary but with time-varying properties
- Use ADF and KPSS tests to confirm
- Non-stationarity → difference the series or use I(1) models

### Autocorrelation
- Short-term AR patterns may indicate mean reversion or microstructure effects
- Lag significance depends on data frequency and sample size
- Use stricter critical values (α=0.01) to filter noise and multiple testing artifacts

### Distribution
- Financial returns typically exhibit fat tails and negative skew
- Normal distribution assumption often violated
- Q-Q plots reveal deviations from normality

## Development & Testing

### Run Tests

```bash
pytest tests/ --maxfail=1 --disable-warnings -q
```

### Code Quality

```bash
flake8 src/ examples/
python -m py_compile src/**/*.py
```

### CI/CD

Automated testing runs on:
- Python 3.9, 3.10, 3.11
- On push and pull requests to `main` and `develop` branches

See `.github/workflows/ci.yml` for details.

## Next Steps

This starter provides a solid foundation for:

- **Advanced modeling**: GARCH, VAR, regime-switching models
- **Factor analysis**: PCA, factor models on multiple assets
- **Portfolio optimization**: Mean-variance, risk parity
- **Backtesting**: Strategy testing on historical data
- **Machine Learning**: Predictive models for returns/volatility

## References

- [Statsmodels Time Series Documentation](https://www.statsmodels.org/stable/tsa.html)
- [yfinance Documentation](https://yfinance.readthedocs.io/)
- [Financial Time Series Analysis Best Practices](https://en.wikipedia.org/wiki/Time_series)
