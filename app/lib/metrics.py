"""
Core performance & risk metrics for a single return series.

Conventions
-----------
- All `returns` inputs are pandas Series of *simple periodic returns*
  (e.g. daily), indexed by date, NOT cumulative and NOT prices.
- `periods_per_year` defaults to 252 (daily trading days). Pass 52 for
  weekly, 12 for monthly data.
- Risk-free rate `rf` is annualized (e.g. 0.02 for 2%); it is converted
  internally to a per-period rate for excess-return calculations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rf_per_period(rf: float, periods_per_year: int) -> float:
    """Convert an annualized risk-free rate to a per-period rate (compounding)."""
    if rf == 0:
        return 0.0
    return (1 + rf) ** (1 / periods_per_year) - 1


def prices_to_returns(prices: pd.Series) -> pd.Series:
    """Simple periodic returns from a price series."""
    return prices.sort_index().pct_change().dropna()


def returns_to_cum_growth(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Cumulative growth-of-1 (or growth-of-start_value) equity curve from returns."""
    return start_value * (1 + returns).cumprod()


def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate implied by a return series."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    growth = (1 + returns).prod()
    n_years = len(returns) / periods_per_year
    if n_years <= 0 or growth <= 0:
        return np.nan
    return growth ** (1 / n_years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    rf_p = _rf_per_period(rf, periods_per_year)
    excess = returns - rf_p
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return (excess.mean() / std) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
    mar: float | None = None,
) -> float:
    """
    Sortino ratio. `mar` (minimum acceptable return, per-period) defaults to the
    per-period risk-free rate if not supplied, matching the Sharpe convention.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    rf_p = _rf_per_period(rf, periods_per_year)
    target = rf_p if mar is None else mar
    excess = returns - target
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.nan
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0 or np.isnan(downside_dev):
        return np.nan
    return (excess.mean() / downside_dev) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown, expressed as a negative fraction (e.g. -0.35)."""
    curve = returns_to_cum_growth(returns.dropna())
    if len(curve) == 0:
        return np.nan
    running_max = curve.cummax()
    dd = curve / running_max - 1
    return dd.min()


def underwater_series(returns: pd.Series) -> pd.Series:
    """Drawdown-from-peak at every point in time (for an underwater/drawdown plot)."""
    curve = returns_to_cum_growth(returns.dropna())
    running_max = curve.cummax()
    return curve / running_max - 1


def rolling_sharpe(
    returns: pd.Series, window: int = 63, rf: float = 0.0, periods_per_year: int = 252
) -> pd.Series:
    rf_p = _rf_per_period(rf, periods_per_year)
    excess = returns - rf_p

    def _sharpe(x):
        s = x.std(ddof=1)
        return np.nan if s == 0 or np.isnan(s) else (x.mean() / s) * np.sqrt(periods_per_year)

    return excess.rolling(window).apply(_sharpe, raw=False)


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical (empirical) Value at Risk at the given confidence level, for a
    single period, expressed as a negative fraction (a loss). E.g. -0.032 means
    a 3.2% loss is the threshold at this confidence level.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return np.percentile(returns, (1 - confidence) * 100)


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Variance-covariance (Gaussian) VaR for a single period."""
    from scipy.stats import norm

    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = norm.ppf(1 - confidence)
    return mu + z * sigma


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Expected Shortfall / CVaR: the average return in the tail beyond
    the VaR threshold. Expressed as a negative fraction.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    threshold = historical_var(returns, confidence)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return threshold
    return tail.mean()


def summary_table(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
    var_confidence: float = 0.95,
) -> pd.Series:
    """One-shot table of the headline metrics for a return series."""
    return pd.Series(
        {
            "CAGR": cagr(returns, periods_per_year),
            "Annualized Volatility": annualized_volatility(returns, periods_per_year),
            "Sharpe Ratio": sharpe_ratio(returns, rf, periods_per_year),
            "Sortino Ratio": sortino_ratio(returns, rf, periods_per_year),
            "Max Drawdown": max_drawdown(returns),
            f"Historical VaR ({int(var_confidence*100)}%)": historical_var(returns, var_confidence),
            f"Expected Shortfall ({int(var_confidence*100)}%)": expected_shortfall(returns, var_confidence),
        }
    )
