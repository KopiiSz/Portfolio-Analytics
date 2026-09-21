"""
Factor-model return decomposition: CAPM, Fama-French 3-factor, Fama-French 5-factor.

All regressions are run on *excess* returns (asset return minus the per-period
risk-free rate). Factor data (Mkt-RF, SMB, HML, RMW, CMA, RF) is expected as a
DataFrame with a DatetimeIndex and those column names, already in the same
return units (simple, per-period, e.g. daily) as the asset series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTOR_SETS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}


def _align(returns: pd.Series, factors: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """Inner-join asset returns with the requested factor columns + RF, drop NaNs."""
    needed = factor_cols + (["RF"] if "RF" in factors.columns else [])
    df = pd.concat([returns.rename("asset"), factors[needed]], axis=1, join="inner").dropna()
    return df


def run_factor_regression(
    returns: pd.Series, factors: pd.DataFrame, model: str = "CAPM"
) -> dict:
    """
    Run a single (full-sample) factor regression.

    Returns a dict with:
      - alpha (annualization left to caller; this is the per-period intercept)
      - alpha_annualized
      - betas: dict of factor -> coefficient
      - r_squared
      - n_obs
      - contribution: dict of factor -> average annualized return attributable
        to that factor (beta * mean factor return, annualized)
      - residual_contribution: annualized alpha, i.e. the part of the return
        not explained by the factors
      - model_summary: statsmodels RegressionResultsWrapper (for diagnostics)
    """
    if model not in FACTOR_SETS:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(FACTOR_SETS)}.")
    factor_cols = FACTOR_SETS[model]
    df = _align(returns, factors, factor_cols)
    if len(df) < len(factor_cols) + 2:
        raise ValueError("Not enough overlapping observations between returns and factor data.")

    rf_col = df["RF"] if "RF" in df.columns else 0.0
    y = df["asset"] - rf_col
    X = sm.add_constant(df[factor_cols])
    fit = sm.OLS(y, X).fit()

    periods_per_year = _infer_periods_per_year(df.index)
    alpha = fit.params["const"]
    betas = {f: fit.params[f] for f in factor_cols}

    contribution = {f: betas[f] * df[f].mean() * periods_per_year for f in factor_cols}
    alpha_annualized = (1 + alpha) ** periods_per_year - 1

    return {
        "alpha": alpha,
        "alpha_annualized": alpha_annualized,
        "betas": betas,
        "r_squared": fit.rsquared,
        "n_obs": int(fit.nobs),
        "contribution": contribution,
        "residual_contribution": alpha_annualized,
        "pvalues": dict(fit.pvalues),
        "model_summary": fit,
    }


def rolling_factor_betas(
    returns: pd.Series, factors: pd.DataFrame, model: str = "CAPM", window: int = 126
) -> pd.DataFrame:
    """
    Rolling-window factor regression. Returns a DataFrame indexed by date with
    one column per factor beta plus 'alpha' and 'r_squared', each computed on
    the trailing `window` observations ending at that date.
    """
    factor_cols = FACTOR_SETS[model]
    df = _align(returns, factors, factor_cols)
    rf_col = df["RF"] if "RF" in df.columns else pd.Series(0.0, index=df.index)
    y_full = df["asset"] - rf_col
    X_full = sm.add_constant(df[factor_cols])

    records = []
    idx = []
    for end in range(window, len(df) + 1):
        y = y_full.iloc[end - window : end]
        X = X_full.iloc[end - window : end]
        fit = sm.OLS(y, X).fit()
        row = {"alpha": fit.params["const"], "r_squared": fit.rsquared}
        for f in factor_cols:
            row[f] = fit.params[f]
        records.append(row)
        idx.append(df.index[end - 1])

    return pd.DataFrame(records, index=pd.Index(idx, name=df.index.name))


def _infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Rough inference of sampling frequency from the median day gap."""
    if len(index) < 3:
        return 252
    diffs = pd.Series(index).diff().dropna().dt.days
    median_gap = diffs.median()
    if median_gap <= 1.5:
        return 252
    if median_gap <= 9:
        return 52
    return 12
