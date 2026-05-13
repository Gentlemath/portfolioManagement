# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Data Loading Module**: Comprehensive support for multiple financial data sources
  - yfinance: Historical price data and basic fundamentals
  - Alpha Vantage: Global equity prices, fundamentals, and symbol search
  - FRED: Macroeconomic time series data
  - AKShare: China market data and fundamentals
  - BaoStock: China A-share backup data
  - Tushare: China equity fundamentals and company profiles
- **Data Loader Factory**: `create_data_loader()` function for easy source selection
- **Exploratory Data Analysis (EDA) Module**: Tools for initial data exploration
  - PlotAnalyzer: Price, return, and cumulative return visualization
  - DistributionAnalyzer: Return distribution statistics and plots
- **EDA Demo Script**: `examples/eda_demo.py` demonstrating data loading and analysis
- **Project Infrastructure**:
  - GitHub Actions CI workflow with Python version matrix testing
  - Environment variable configuration for API keys
  - Comprehensive README with usage examples

### Changed
- Updated `requirements.txt` with new dependencies (matplotlib, scipy, financial data libraries)
- Enhanced project structure with organized modules in `src/`
- Improved documentation and setup instructions in README


  