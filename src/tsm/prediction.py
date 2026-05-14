"""Time series modeling and prediction module."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union


class GARCHPredictor:
    """GARCH model for volatility and return prediction."""

    def __init__(self, p: int = 1, q: int = 1, mean_model: str = 'Constant'):
        """
        Initialize GARCH predictor.

        Args:
            p: GARCH order (volatility lags)
            q: ARCH order (squared residual lags)
            mean_model: Mean model type ('Constant', 'AR', 'Zero')
        """
        self.p = p
        self.q = q
        self.mean_model = mean_model
        self.model = None
        self.fitted_model = None
        self.residuals = None
        self.conditional_volatility = None

    def fit(self, returns: pd.Series, update_freq: int = 1) -> Dict[str, float]:
        """
        Fit GARCH model to returns data.

        Args:
            returns: Time series of returns
            update_freq: Frequency of parameter updates

        Returns:
            Dictionary with model summary statistics
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("arch library required for GARCH modeling. Install with `pip install arch`.")

        # Create the model
        self.model = arch_model(
            returns,
            mean=self.mean_model,
            vol='Garch',
            p=self.p,
            q=self.q,
        )

        # Fit with optional update frequency
        self.fitted_model = self.model.fit(disp='off', update_freq=update_freq)

        # Store residuals and conditional volatility
        self.residuals = self.fitted_model.resid
        self.conditional_volatility = self.fitted_model.conditional_volatility

        # Return summary statistics
        summary = {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'log_likelihood': self.fitted_model.loglikelihood,
            'convergence': getattr(self.fitted_model, 'convergence_flag', None),
            'iterations': getattr(self.fitted_model, 'iterations', None)
        }

        return summary

    def predict_volatility(self, horizon: int = 1) -> pd.Series:
        """
        Predict future conditional volatility.

        Args:
            horizon: Number of periods ahead to forecast

        Returns:
            Series of predicted volatilities
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction.")

        forecast = self.fitted_model.forecast(horizon=horizon)
        return forecast.variance.iloc[-1]

    def predict_return(self, horizon: int = 1, method: str = 'zero') -> pd.Series:
        """
        Predict future returns using GARCH volatility.

        Args:
            horizon: Number of periods ahead to forecast
            method: Prediction method ('zero' for mean return, 'historical' for historical mean)

        Returns:
            Series of predicted returns with confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction.")

        # Get volatility forecast
        vol_forecast = self.predict_volatility(horizon)

        # Get mean forecast based on method
        if method == 'zero':
            mean_forecast = 0.0
        elif method == 'historical':
            # Use the mean from the fitted model
            mean_forecast = self.fitted_model.params.get('mu', 0.0)
        else:
            raise ValueError(f"Unknown method: {method}")

        # For simplicity, return point forecast (mean) and volatility
        # In practice, you might want to return distribution parameters
        predictions = pd.Series({
            'predicted_return': mean_forecast,
            'predicted_volatility': np.sqrt(vol_forecast.iloc[0]),
            'confidence_interval_95_lower': mean_forecast - 1.96 * np.sqrt(vol_forecast.iloc[0]),
            'confidence_interval_95_upper': mean_forecast + 1.96 * np.sqrt(vol_forecast.iloc[0])
        })

        return predictions

    def get_model_summary(self) -> str:
        """Get detailed model summary."""
        if self.fitted_model is None:
            return "Model not fitted yet."

        return str(self.fitted_model.summary())

    def get_parameters(self) -> pd.Series:
        """Get fitted model parameters."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before getting parameters.")

        return self.fitted_model.params

    def plot_volatility(self) -> None:
        """Plot conditional volatility."""
        if self.conditional_volatility is None:
            raise ValueError("Model must be fitted before plotting.")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting.")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.conditional_volatility.index, self.conditional_volatility.values,
                label='Conditional Volatility', color='red', linewidth=1.5)
        ax.set_title('GARCH Conditional Volatility')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, test_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            test_returns: Optional test set for out-of-sample evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before evaluation.")

        evaluation = {}

        # In-sample evaluation
        evaluation['in_sample_log_likelihood'] = self.fitted_model.loglikelihood
        evaluation['in_sample_aic'] = self.fitted_model.aic
        evaluation['in_sample_bic'] = self.fitted_model.bic

        # Standardized residuals should be approximately white noise
        std_resid = self.fitted_model.resid / self.fitted_model.conditional_volatility
        evaluation['std_resid_mean'] = std_resid.mean()
        evaluation['std_resid_std'] = std_resid.std()
        evaluation['std_resid_skew'] = std_resid.skew()
        evaluation['std_resid_kurtosis'] = std_resid.kurtosis()

        # Ljung-Box test for autocorrelation in squared residuals
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_test = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
            evaluation['ljung_box_pvalue'] = lb_test['lb_pvalue'].iloc[0]
        except ImportError:
            evaluation['ljung_box_pvalue'] = None

        return evaluation


class ARIMAGARCHPredictor:
    """ARIMA-GARCH model for return prediction."""

    def __init__(self, arima_order: Tuple[int, int, int] = (1, 0, 1), garch_order: Tuple[int, int] = (1, 1)):
        """
        Initialize ARIMA-GARCH predictor.

        Args:
            arima_order: (p, d, q) for ARIMA mean model
            garch_order: (p, q) for GARCH volatility model
        """
        self.arima_order = arima_order
        self.garch_order = garch_order
        self.model = None
        self.fitted_model = None

    def fit(self, returns: pd.Series) -> Dict[str, float]:
        """Fit ARIMA-GARCH model."""
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("arch library required for ARIMA-GARCH modeling.")

        p, d, q = self.arima_order
        vol_p, vol_q = self.garch_order

        self.model = arch_model(
            returns,
            mean='ARX' if p > 0 or q > 0 else 'Constant',
            lags=p if p > 0 else None,
            vol='Garch',
            p=vol_p,
            q=vol_q,
            power=2.0 if d == 0 else 1.0  # Power for integrated models
        )

        self.fitted_model = self.model.fit(disp='off')

        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'log_likelihood': self.fitted_model.loglikelihood
        }

    def predict_return(self, horizon: int = 1) -> pd.Series:
        """Predict future returns."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction.")

        forecast = self.fitted_model.forecast(horizon=horizon)

        # Get mean forecast
        mean_forecast = forecast.mean.iloc[-1].iloc[0]

        # Get volatility forecast
        vol_forecast = np.sqrt(forecast.variance.iloc[-1].iloc[0])

        predictions = pd.Series({
            'predicted_return': mean_forecast,
            'predicted_volatility': vol_forecast,
            'confidence_interval_95_lower': mean_forecast - 1.96 * vol_forecast,
            'confidence_interval_95_upper': mean_forecast + 1.96 * vol_forecast
        })

        return predictions

