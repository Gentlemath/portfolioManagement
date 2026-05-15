"""Portfolio Management Package"""

__version__ = "0.1.0"

# Import main modules for easy access
from .dataloader import create_data_loader
from .eda import PlotAnalyzer, DistributionAnalyzer, TimeSeriesAnalyzer
from .tsm import ARIMAGARCHPredictor, GARCHPredictor, MarkovSwitchingGARCHPredictor, RegimeDetector

__all__ = [
    "create_data_loader",
    "PlotAnalyzer",
    "DistributionAnalyzer",
    "TimeSeriesAnalyzer",
    "GARCHPredictor",
    "ARIMAGARCHPredictor",
    "MarkovSwitchingGARCHPredictor",
    "RegimeDetector",
]
