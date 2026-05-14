"""Example GARCH modeling script for volatility prediction."""

from pathlib import Path
import sys

import pandas as pd
import yfinance as yf

# Ensure src is on the import path when running from the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsm import GARCHPredictor


def main() -> None:
    # Load S&P 500 data
    print("Loading S&P 500 data...")
    prices = yf.download("^GSPC", start="2020-01-01", end="2026-05-12", progress=False)[["Close"]]
    prices = prices.rename(columns={"Close": "SP500"})

    # Calculate returns
    returns = (prices.pct_change().dropna() * 100)  # Convert to percentage

    print(f"Loaded {len(returns)} daily returns")
    print(f"Sample returns: {returns.head()}")

    # Fit GARCH(1,1) model
    print("\nFitting GARCH(1,1) model...")
    garch = GARCHPredictor(p=1, q=1, mean_model='Constant')
    summary = garch.fit(returns['SP500'])

    print("Model Summary:")
    print(f"AIC: {summary['aic']:.2f}")
    print(f"BIC: {summary['bic']:.2f}")
    print(f"Log Likelihood: {summary['log_likelihood']:.2f}")
    print(f"Convergence: {'Yes' if summary['convergence'] == 0 else 'No'}")

    # Get model parameters
    params = garch.get_parameters()
    print(f"\nModel Parameters:")
    print(params)

    # Predict next day volatility and return
    print("\nPredicting next day...")
    prediction = garch.predict_return(horizon=1, method='zero')
    print(f"Predicted Return: {prediction['predicted_return']:.4f}")
    print(f"Predicted Volatility: {prediction['predicted_volatility']:.4f}")
    print(f"95% Confidence Interval: [{prediction['confidence_interval_95_lower']:.4f}, {prediction['confidence_interval_95_upper']:.4f}]")

    # Evaluate model
    evaluation = garch.evaluate_model()
    print("Model Evaluation:")
    print(f"Standardized Residuals Mean: {evaluation['std_resid_mean']:.4f}")
    print(f"Standardized Residuals Std: {evaluation['std_resid_std']:.4f}")
    print(f"Standardized Residuals Skew: {evaluation['std_resid_skew']:.4f}")
    print(f"Standardized Residuals Kurtosis: {evaluation['std_resid_kurtosis']:.4f}")
    if evaluation['ljung_box_pvalue'] is not None:
        print(f"Ljung-Box p-value (autocorrelation in squared residuals): {evaluation['ljung_box_pvalue']:.4f}")

    # Plot conditional volatility
    print("\nPlotting conditional volatility...")
    garch.plot_volatility()


if __name__ == "__main__":
    main()
