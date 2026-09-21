"""Plotly chart builders. Kept separate from Streamlit so they're reusable
if the UI layer is ever swapped out (e.g. for a FastAPI + React frontend)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import metrics


def equity_curve_fig(returns: pd.Series, benchmark_returns: pd.Series | None = None,
                      title: str = "Equity Curve") -> go.Figure:
    curve = metrics.returns_to_cum_growth(returns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name="Portfolio", mode="lines"))
    if benchmark_returns is not None:
        bench_curve = metrics.returns_to_cum_growth(benchmark_returns)
        fig.add_trace(go.Scatter(x=bench_curve.index, y=bench_curve.values, name="Benchmark",
                                  mode="lines", line=dict(dash="dot")))
    fig.update_layout(title=title, yaxis_title="Growth of 1", xaxis_title="Date",
                       hovermode="x unified", legend=dict(orientation="h", y=1.02))
    return fig


def underwater_fig(returns: pd.Series, benchmark_returns: pd.Series | None = None,
                    title: str = "Drawdown from Peak (Underwater Plot)") -> go.Figure:
    uw = metrics.underwater_series(returns)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=uw.index, y=uw.values * 100, name="Portfolio", fill="tozeroy",
                              mode="lines"))
    if benchmark_returns is not None:
        uw_b = metrics.underwater_series(benchmark_returns)
        fig.add_trace(go.Scatter(x=uw_b.index, y=uw_b.values * 100, name="Benchmark",
                                  mode="lines", line=dict(dash="dot")))
    fig.update_layout(title=title, yaxis_title="Drawdown (%)", xaxis_title="Date",
                       hovermode="x unified", legend=dict(orientation="h", y=1.02))
    return fig


def rolling_sharpe_fig(returns: pd.Series, window: int = 63, rf: float = 0.0,
                        periods_per_year: int = 252,
                        benchmark_returns: pd.Series | None = None,
                        title: str | None = None) -> go.Figure:
    rs = metrics.rolling_sharpe(returns, window, rf, periods_per_year)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs.index, y=rs.values, name="Portfolio", mode="lines"))
    if benchmark_returns is not None:
        rs_b = metrics.rolling_sharpe(benchmark_returns, window, rf, periods_per_year)
        fig.add_trace(go.Scatter(x=rs_b.index, y=rs_b.values, name="Benchmark",
                                  mode="lines", line=dict(dash="dot")))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=title or f"Rolling {window}-period Sharpe Ratio",
                       yaxis_title="Sharpe Ratio", xaxis_title="Date",
                       hovermode="x unified", legend=dict(orientation="h", y=1.02))
    return fig


def rolling_betas_fig(rolling_betas_df: pd.DataFrame, factor_cols: list[str],
                       title: str = "Rolling Factor Betas") -> go.Figure:
    fig = go.Figure()
    for f in factor_cols:
        fig.add_trace(go.Scatter(x=rolling_betas_df.index, y=rolling_betas_df[f], name=f, mode="lines"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=title, yaxis_title="Beta", xaxis_title="Date",
                       hovermode="x unified", legend=dict(orientation="h", y=1.02))
    return fig


def factor_contribution_fig(contribution: dict, residual: float,
                             title: str = "Return Decomposition") -> go.Figure:
    labels = list(contribution.keys()) + ["Alpha (unexplained)"]
    values = list(contribution.values()) + [residual]
    fig = go.Figure(go.Bar(x=labels, y=[v * 100 for v in values]))
    fig.update_layout(title=title, yaxis_title="Annualized contribution (%)")
    return fig


def risk_contribution_fig(contrib_df: pd.DataFrame, weight_col: str, risk_col: str,
                           title: str = "Weight vs. Risk Contribution") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=contrib_df.index, y=contrib_df[weight_col] * 100, name="% of Weight"))
    fig.add_trace(go.Bar(x=contrib_df.index, y=contrib_df[risk_col] * 100, name="% of Risk"))
    fig.update_layout(title=title, barmode="group", yaxis_title="Percent (%)",
                       legend=dict(orientation="h", y=1.02))
    return fig
