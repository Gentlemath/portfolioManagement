# Changelog

Notable changes and version information.

## [Unreleased]

### Added
- Initial data loader support for multiple sources:
  - yfinance (price + simple fundamentals)
  - Alpha Vantage (global equities, price, fundamentals)
  - FRED (macro data)
  - AKShare (China market data)
  - BaoStock (China A-share backup)
  - Tushare (China fundamentals)
- `create_data_loader` factory for source selection

### Changed
- Project README and requirements updated for new data sources
  