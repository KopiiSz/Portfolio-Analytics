"""
Multi-asset portfolio risk decomposition.

Two approaches are provided:

1. `euler_risk_contribution` — the standard closed-form "component VaR" /
   marginal-contribution-to-risk decomposition. Fast (O(N^2)), exact under
   the assumption that portfolio risk = volatility of weighted sum, and
   satisfies additivity (contributions sum exactly to total portfolio risk).
   This is the industry-standard method.

2. `shapley_risk_contribution` — a game-theoretic decomposition that treats
   each asset as a "player" and portfolio volatility (given the *actual*
   weights, with non-participating assets zeroed out) as the value of a
   coalition. The Shapley value is the average marginal contribution across
   all possible orderings in which assets are "added" to the portfolio. It
   also sums exactly to total risk, but unlike the Euler method it doesn't
   assume linearity, so it can reveal a different (and arguably fairer) split
   when assets interact non-trivially through correlation. Cost is O(2^N),
   so it's only run for reasonably small numbers of assets (<= ~18).
"""
from __future__ import annotations

from itertools import combinations
from math import comb, factorial

import numpy as np
import pandas as pd


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    variance = weights @ cov @ weights
    return float(np.sqrt(max(variance, 0.0)))


def euler_risk_contribution(weights: pd.Series, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Closed-form marginal contribution to risk (Euler / component VaR method).

    weights: Series indexed by asset name, summing to 1 (or any total).
    returns: DataFrame of periodic returns, one column per asset, same index
             (dates) as used to estimate the covariance matrix.

    Returns a DataFrame with columns:
      weight, marginal_contribution (per unit of weight), risk_contribution
      (weight * marginal), pct_of_risk (risk_contribution / total risk),
      pct_of_weight (for comparison against pct_of_risk).
    """
    assets = list(weights.index)
    cov = returns[assets].cov().values
    w = weights.values.astype(float)

    port_vol = portfolio_volatility(w, cov)
    if port_vol == 0:
        marginal = np.zeros_like(w)
    else:
        marginal = (cov @ w) / port_vol  # d(port_vol)/d(w_i)

    risk_contribution = w * marginal
    pct_of_risk = risk_contribution / port_vol if port_vol != 0 else np.full_like(w, np.nan)
    pct_of_weight = w / w.sum() if w.sum() != 0 else np.full_like(w, np.nan)

    return pd.DataFrame(
        {
            "weight": w,
            "pct_of_weight": pct_of_weight,
            "marginal_contribution": marginal,
            "risk_contribution": risk_contribution,
            "pct_of_risk": pct_of_risk,
        },
        index=assets,
    )


def shapley_risk_contribution(
    weights: pd.Series, returns: pd.DataFrame, annualize_periods: int | None = None
) -> pd.DataFrame:
    """
    Shapley-value decomposition of portfolio volatility.

    The characteristic function v(S) for a coalition S of assets is the
    volatility of the portfolio built from ONLY those assets' actual weights
    (other assets zeroed out) — i.e. v(S) = vol(sum_{i in S} w_i * r_i).
    v(empty set) = 0.

    Cost is O(2^N); intended for portfolios of up to ~15-18 holdings.

    Returns a DataFrame with columns: weight, shapley_contribution,
    pct_of_risk, pct_of_weight — directly comparable to
    `euler_risk_contribution`'s output.
    """
    assets = list(weights.index)
    n = len(assets)
    if n > 18:
        raise ValueError(
            f"Shapley decomposition is O(2^N); {n} assets is too many to run exactly. "
            "Use euler_risk_contribution for larger portfolios."
        )

    w = weights.astype(float)
    rets = returns[assets]

    # Precompute v(S) for every subset.
    value_cache: dict[frozenset, float] = {frozenset(): 0.0}
    for size in range(1, n + 1):
        for combo in combinations(assets, size):
            subset = frozenset(combo)
            weighted = rets[list(subset)].mul(w[list(subset)], axis=1).sum(axis=1)
            value_cache[subset] = float(weighted.std(ddof=1))

    shapley = {a: 0.0 for a in assets}
    all_others = set(assets)
    for i in assets:
        others = list(all_others - {i})
        m = len(others)
        for size in range(0, m + 1):
            weight_factor = factorial(size) * factorial(m - size) / factorial(m + 1)
            for combo in combinations(others, size):
                S = frozenset(combo)
                S_with_i = S | {i}
                marginal = value_cache[S_with_i] - value_cache[S]
                shapley[i] += weight_factor * marginal

    total_risk = value_cache[frozenset(assets)]
    shap_series = pd.Series(shapley)
    pct_of_risk = shap_series / total_risk if total_risk != 0 else np.nan
    pct_of_weight = w / w.sum() if w.sum() != 0 else np.nan

    result = pd.DataFrame(
        {
            "weight": w,
            "pct_of_weight": pct_of_weight,
            "shapley_contribution": shap_series,
            "pct_of_risk": pct_of_risk,
        }
    )
    if annualize_periods:
        result["shapley_contribution_annualized"] = shap_series * np.sqrt(annualize_periods)
    return result
