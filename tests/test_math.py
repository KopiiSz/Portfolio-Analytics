import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from app.lib import metrics, factor_models, risk_decomposition

np.random.seed(42)
n = 1000
dates = pd.bdate_range("2020-01-01", periods=n)

# --- synthetic single asset return series ---
returns = pd.Series(np.random.normal(0.0005, 0.012, n), index=dates)
bench = pd.Series(np.random.normal(0.0003, 0.010, n), index=dates)

print("=== metrics.summary_table ===")
print(metrics.summary_table(returns))

print("\n=== rolling sharpe (tail) ===")
print(metrics.rolling_sharpe(returns, window=63).dropna().tail())

print("\n=== underwater series (tail) ===")
print(metrics.underwater_series(returns).tail())

# --- synthetic factor data for factor_models ---
factors = pd.DataFrame(
    {
        "Mkt-RF": np.random.normal(0.0004, 0.009, n),
        "SMB": np.random.normal(0.0001, 0.004, n),
        "HML": np.random.normal(0.0001, 0.004, n),
        "RMW": np.random.normal(0.0001, 0.003, n),
        "CMA": np.random.normal(0.0001, 0.003, n),
        "RF": np.full(n, 0.00005),
    },
    index=dates,
)

print("\n=== CAPM regression ===")
res = factor_models.run_factor_regression(returns, factors, "CAPM")
print({k: v for k, v in res.items() if k != "model_summary"})

print("\n=== FF5 regression ===")
res5 = factor_models.run_factor_regression(returns, factors, "FF5")
print({k: v for k, v in res5.items() if k != "model_summary"})

print("\n=== rolling CAPM betas (tail) ===")
rb = factor_models.rolling_factor_betas(returns, factors, "CAPM", window=126)
print(rb.tail())

# --- synthetic multi-asset portfolio for risk decomposition ---
assets = ["A", "B", "C", "D"]
asset_returns = pd.DataFrame(
    {
        "A": np.random.normal(0.0006, 0.020, n),   # high vol
        "B": np.random.normal(0.0004, 0.008, n),   # low vol
        "C": np.random.normal(0.0005, 0.012, n),
        "D": np.random.normal(0.0003, 0.006, n),
    },
    index=dates,
)
# Make A and C correlated to create an interesting risk decomposition
asset_returns["C"] = 0.6 * asset_returns["A"] + 0.4 * asset_returns["C"]

weights = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})  # equal weight

print("\n=== Euler risk contribution (equal weights, A&C correlated) ===")
euler = risk_decomposition.euler_risk_contribution(weights, asset_returns)
print(euler)
print("Sum of risk_contribution vs total portfolio vol:",
      euler["risk_contribution"].sum(),
      risk_decomposition.portfolio_volatility(weights.values, asset_returns[assets].cov().values))

print("\n=== Shapley risk contribution (equal weights, A&C correlated) ===")
shap = risk_decomposition.shapley_risk_contribution(weights, asset_returns)
print(shap)
print("Sum of shapley_contribution vs total portfolio vol:",
      shap["shapley_contribution"].sum(),
      risk_decomposition.portfolio_volatility(weights.values, asset_returns[assets].cov().values))

print("\nAll smoke tests completed without error.")
